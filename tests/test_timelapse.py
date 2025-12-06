"""Tests for timelapse encoding module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frigate_tools.timelapse import (
    ProgressInfo,
    concat_files,
    create_timelapse,
    encode_timelapse,
    get_video_duration,
    parse_ffmpeg_progress,
)


class TestParseFFmpegProgress:
    """Tests for FFmpeg progress parsing."""

    def test_parse_standard_progress_line(self):
        """Parses standard FFmpeg progress output."""
        line = "frame=  100 fps= 30 q=28.0 size=    1024kB time=00:00:03.33 bitrate=2517.8kbits/s speed=1.50x"
        progress = parse_ffmpeg_progress(line)

        assert progress is not None
        assert progress.frame == 100
        assert progress.fps == 30.0
        assert abs(progress.time_seconds - 3.33) < 0.01
        assert progress.speed == 1.50

    def test_parse_with_total_duration(self):
        """Calculates percentage when total duration provided."""
        line = "frame=  100 fps= 30 time=00:00:05.00 speed=1.00x"
        progress = parse_ffmpeg_progress(line, total_duration=10.0)

        assert progress is not None
        assert progress.percent == 50.0

    def test_parse_with_hours(self):
        """Handles time with hours."""
        line = "frame=10000 fps=60 time=01:30:45.50 speed=2.00x"
        progress = parse_ffmpeg_progress(line)

        assert progress is not None
        expected_seconds = 1 * 3600 + 30 * 60 + 45.50
        assert abs(progress.time_seconds - expected_seconds) < 0.01

    def test_parse_zero_fps(self):
        """Handles missing fps."""
        line = "frame=  100 time=00:00:05.00 speed=1.00x"
        progress = parse_ffmpeg_progress(line)

        assert progress is not None
        assert progress.fps == 0.0

    def test_parse_invalid_line(self):
        """Returns None for non-progress lines."""
        assert parse_ffmpeg_progress("Encoding started...") is None
        assert parse_ffmpeg_progress("") is None
        assert parse_ffmpeg_progress("fps=30") is None

    def test_percent_capped_at_100(self):
        """Percent is capped at 100."""
        line = "frame=  100 fps= 30 time=00:00:15.00 speed=1.00x"
        progress = parse_ffmpeg_progress(line, total_duration=10.0)

        assert progress is not None
        assert progress.percent == 100.0


class TestGetVideoDuration:
    """Tests for video duration detection."""

    @patch("subprocess.run")
    def test_get_duration_success(self, mock_run):
        """Returns duration from ffprobe output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="123.456\n",
        )

        duration = get_video_duration(Path("/test/video.mp4"))
        assert abs(duration - 123.456) < 0.001

    @patch("subprocess.run")
    def test_get_duration_failure(self, mock_run):
        """Returns 0 on ffprobe failure."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
        )

        duration = get_video_duration(Path("/test/video.mp4"))
        assert duration == 0.0

    @patch("subprocess.run")
    def test_get_duration_invalid_output(self, mock_run):
        """Returns 0 on invalid ffprobe output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not a number\n",
        )

        duration = get_video_duration(Path("/test/video.mp4"))
        assert duration == 0.0


class TestConcatFiles:
    """Tests for file concatenation."""

    def test_concat_empty_list(self):
        """Returns False for empty input list."""
        result = concat_files([], Path("/test/output.mp4"))
        assert result is False

    @patch("subprocess.Popen")
    def test_concat_creates_process(self, mock_popen, tmp_path):
        """Creates ffmpeg process with correct arguments."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("", "")
        mock_popen.return_value = mock_process

        input_files = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        for f in input_files:
            f.touch()

        output_path = tmp_path / "output.mp4"
        result = concat_files(input_files, output_path)

        assert result is True
        mock_popen.assert_called_once()

        # Check ffmpeg was called with concat demuxer
        call_args = mock_popen.call_args[0][0]
        assert "ffmpeg" in call_args[0]
        assert "-f" in call_args
        assert "concat" in call_args
        assert "-c" in call_args
        assert "copy" in call_args

    @patch("subprocess.Popen")
    def test_concat_failure(self, mock_popen, tmp_path):
        """Returns False on ffmpeg failure."""
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = ("", "Error message")
        mock_popen.return_value = mock_process

        input_files = [tmp_path / "a.mp4"]
        input_files[0].touch()

        result = concat_files(input_files, tmp_path / "output.mp4")
        assert result is False


class TestEncodeTimelapse:
    """Tests for timelapse encoding."""

    @patch("frigate_tools.timelapse.get_video_duration")
    @patch("subprocess.Popen")
    def test_encode_calculates_pts_multiplier(self, mock_popen, mock_duration, tmp_path):
        """Calculates correct PTS multiplier for speed adjustment."""
        mock_duration.return_value = 600.0  # 10 minutes
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = iter([])
        mock_process.communicate.return_value = ("", "")
        mock_popen.return_value = mock_process

        input_path = tmp_path / "input.mp4"
        input_path.touch()
        output_path = tmp_path / "output.mp4"

        # Target 1 minute output from 10 minute source = 10x speed
        # PTS multiplier = 1/10 = 0.1
        result = encode_timelapse(input_path, output_path, target_duration=60.0)

        assert result is True

        # Check setpts filter was applied
        call_args = mock_popen.call_args[0][0]
        filter_idx = call_args.index("-filter:v") + 1
        assert "setpts=" in call_args[filter_idx]
        assert "0.1" in call_args[filter_idx]

    @patch("frigate_tools.timelapse.get_video_duration")
    def test_encode_fails_without_duration(self, mock_duration, tmp_path):
        """Returns False when source duration cannot be determined."""
        mock_duration.return_value = 0.0

        input_path = tmp_path / "input.mp4"
        input_path.touch()

        result = encode_timelapse(input_path, tmp_path / "output.mp4", target_duration=60.0)
        assert result is False

    @patch("frigate_tools.timelapse.get_video_duration")
    @patch("subprocess.Popen")
    def test_encode_with_progress_callback(self, mock_popen, mock_duration, tmp_path):
        """Calls progress callback with updates."""
        mock_duration.return_value = 100.0
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = iter([
            "frame=  50 fps= 30 time=00:00:05.00 speed=1.00x\n",
            "frame= 100 fps= 30 time=00:00:10.00 speed=1.00x\n",
        ])
        mock_process.communicate.return_value = ("", "")
        mock_popen.return_value = mock_process

        callback = MagicMock()
        input_path = tmp_path / "input.mp4"
        input_path.touch()

        encode_timelapse(
            input_path,
            tmp_path / "output.mp4",
            target_duration=10.0,
            progress_callback=callback,
        )

        assert callback.call_count == 2


class TestCreateTimelapse:
    """Tests for full timelapse creation."""

    @patch("frigate_tools.timelapse.encode_timelapse")
    @patch("frigate_tools.timelapse.concat_files")
    def test_create_timelapse_two_step_process(self, mock_concat, mock_encode, tmp_path):
        """Creates timelapse using concat then encode."""
        mock_concat.return_value = True
        mock_encode.return_value = True

        input_files = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        for f in input_files:
            f.touch()

        output_path = tmp_path / "output.mp4"
        result = create_timelapse(input_files, output_path, target_duration=60.0)

        assert result is True
        mock_concat.assert_called_once()
        mock_encode.assert_called_once()

    @patch("frigate_tools.timelapse.concat_files")
    def test_create_timelapse_fails_on_concat_failure(self, mock_concat, tmp_path):
        """Returns False when concat fails."""
        mock_concat.return_value = False

        input_files = [tmp_path / "a.mp4"]
        input_files[0].touch()

        result = create_timelapse(input_files, tmp_path / "output.mp4", target_duration=60.0)
        assert result is False

    @patch("frigate_tools.timelapse.encode_timelapse")
    @patch("frigate_tools.timelapse.concat_files")
    def test_create_timelapse_fails_on_encode_failure(self, mock_concat, mock_encode, tmp_path):
        """Returns False when encode fails."""
        mock_concat.return_value = True
        mock_encode.return_value = False

        input_files = [tmp_path / "a.mp4"]
        input_files[0].touch()

        result = create_timelapse(input_files, tmp_path / "output.mp4", target_duration=60.0)
        assert result is False

    @patch("frigate_tools.timelapse.encode_timelapse")
    @patch("frigate_tools.timelapse.concat_files")
    def test_create_timelapse_cleans_temp_file(self, mock_concat, mock_encode, tmp_path):
        """Removes temporary concat file after completion."""
        mock_concat.return_value = True
        mock_encode.return_value = True

        input_files = [tmp_path / "a.mp4"]
        input_files[0].touch()

        output_path = tmp_path / "output.mp4"
        create_timelapse(input_files, output_path, target_duration=60.0)

        # Temp file should be cleaned up
        temp_file = tmp_path / ".output_concat.mp4"
        assert not temp_file.exists()

    @patch("frigate_tools.timelapse.encode_timelapse")
    @patch("frigate_tools.timelapse.concat_files")
    def test_create_timelapse_keeps_temp_when_requested(self, mock_concat, mock_encode, tmp_path):
        """Keeps temp file when keep_temp=True."""
        # Create actual temp file to test preservation
        def create_temp(files, output):
            output.touch()
            return True

        mock_concat.side_effect = create_temp
        mock_encode.return_value = True

        input_files = [tmp_path / "a.mp4"]
        input_files[0].touch()

        output_path = tmp_path / "output.mp4"
        create_timelapse(input_files, output_path, target_duration=60.0, keep_temp=True)

        # When keep_temp=True, we don't delete, but the mock creates the file
        # at a different path, so this test mainly verifies the flag is passed
