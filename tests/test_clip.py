"""Tests for clip file selection and concatenation."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frigate_tools.clip import (
    ClipProgress,
    concat_clip,
    create_clip,
    create_multi_camera_clip,
    find_overlapping_segments,
)


class TestFindOverlappingSegments:
    """Tests for segment finding."""

    @pytest.fixture
    def mock_frigate_dir(self, tmp_path):
        """Create a mock Frigate directory structure.

        Frigate structure: recordings/YYYY-MM-DD/HH/camera/MM.SS.mp4
        """
        recordings_path = tmp_path / "recordings"

        # Create segments for Dec 5, 2025
        hour12 = recordings_path / "2025-12-05" / "12" / "front"
        hour12.mkdir(parents=True)
        (hour12 / "00.00.mp4").touch()
        (hour12 / "15.00.mp4").touch()
        (hour12 / "30.00.mp4").touch()
        (hour12 / "45.00.mp4").touch()

        hour13 = recordings_path / "2025-12-05" / "13" / "front"
        hour13.mkdir(parents=True)
        (hour13 / "00.00.mp4").touch()
        (hour13 / "15.00.mp4").touch()

        return tmp_path

    def test_finds_all_segments_in_range(self, mock_frigate_dir):
        """Finds all segments within time range."""
        files = find_overlapping_segments(
            instance_path=mock_frigate_dir,
            camera="front",
            start=datetime(2025, 12, 5, 12, 0, 0),
            end=datetime(2025, 12, 5, 14, 0, 0),
        )
        assert len(files) == 6

    def test_partial_range(self, mock_frigate_dir):
        """Finds segments in partial range."""
        files = find_overlapping_segments(
            instance_path=mock_frigate_dir,
            camera="front",
            start=datetime(2025, 12, 5, 12, 20, 0),
            end=datetime(2025, 12, 5, 12, 50, 0),
        )
        # Should get 12:30 and 12:45
        assert len(files) == 2

    def test_no_calendar_filtering(self, mock_frigate_dir):
        """Clips don't apply calendar filtering."""
        # Even on a Saturday, all files should be included
        files = find_overlapping_segments(
            instance_path=mock_frigate_dir,
            camera="front",
            start=datetime(2025, 12, 5, 12, 0, 0),  # This is a Friday
            end=datetime(2025, 12, 5, 13, 30, 0),
        )
        # All segments should be found
        assert len(files) == 6

    def test_nonexistent_camera(self, mock_frigate_dir):
        """Returns empty for nonexistent camera."""
        files = find_overlapping_segments(
            instance_path=mock_frigate_dir,
            camera="nonexistent",
            start=datetime(2025, 12, 5, 12, 0, 0),
            end=datetime(2025, 12, 5, 13, 0, 0),
        )
        assert files == []


class TestConcatClip:
    """Tests for clip concatenation."""

    def test_empty_input_list(self):
        """Returns False for empty input."""
        result = concat_clip([], Path("/test/output.mp4"))
        assert result is False

    @patch("subprocess.Popen")
    def test_stream_copy_by_default(self, mock_popen, tmp_path):
        """Uses stream copy by default."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("", "")
        mock_popen.return_value = mock_process

        input_file = tmp_path / "a.mp4"
        input_file.touch()

        concat_clip([input_file], tmp_path / "output.mp4")

        call_args = mock_popen.call_args[0][0]
        assert "-c" in call_args
        assert "copy" in call_args

    @patch("subprocess.Popen")
    def test_reencode_option(self, mock_popen, tmp_path):
        """Uses re-encoding when requested."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("", "")
        mock_popen.return_value = mock_process

        input_file = tmp_path / "a.mp4"
        input_file.touch()

        concat_clip([input_file], tmp_path / "output.mp4", reencode=True)

        call_args = mock_popen.call_args[0][0]
        assert "-preset" in call_args
        # Should not have -c copy
        assert "-c" not in call_args or "copy" not in call_args

    @patch("subprocess.Popen")
    def test_failure_returns_false(self, mock_popen, tmp_path):
        """Returns False on ffmpeg failure."""
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = ("", "Error")
        mock_popen.return_value = mock_process

        input_file = tmp_path / "a.mp4"
        input_file.touch()

        result = concat_clip([input_file], tmp_path / "output.mp4")
        assert result is False


class TestCreateClip:
    """Tests for clip creation."""

    @pytest.fixture
    def mock_frigate_dir(self, tmp_path):
        """Create mock Frigate directory.

        Frigate structure: recordings/YYYY-MM-DD/HH/camera/MM.SS.mp4
        """
        recordings_path = tmp_path / "recordings"
        camera_path = recordings_path / "2025-12-05" / "12" / "front"
        camera_path.mkdir(parents=True)
        (camera_path / "00.00.mp4").touch()
        (camera_path / "30.00.mp4").touch()
        return tmp_path

    @patch("frigate_tools.clip.concat_clip")
    def test_creates_clip(self, mock_concat, mock_frigate_dir, tmp_path):
        """Creates clip from found segments."""
        mock_concat.return_value = True

        result = create_clip(
            instance_path=mock_frigate_dir,
            camera="front",
            start=datetime(2025, 12, 5, 12, 0, 0),
            end=datetime(2025, 12, 5, 13, 0, 0),
            output_path=tmp_path / "output.mp4",
        )

        assert result is True
        mock_concat.assert_called_once()

        # Verify correct files were passed
        call_args = mock_concat.call_args
        files = call_args[1]["input_files"]
        assert len(files) == 2

    @patch("frigate_tools.clip.concat_clip")
    def test_returns_false_no_files(self, mock_concat, tmp_path):
        """Returns False when no files found."""
        result = create_clip(
            instance_path=tmp_path,  # Empty directory
            camera="front",
            start=datetime(2025, 12, 5, 12, 0, 0),
            end=datetime(2025, 12, 5, 13, 0, 0),
            output_path=tmp_path / "output.mp4",
        )

        assert result is False
        mock_concat.assert_not_called()

    @patch("frigate_tools.clip.concat_clip")
    def test_progress_callback(self, mock_concat, mock_frigate_dir, tmp_path):
        """Calls progress callback at different stages."""
        mock_concat.return_value = True
        callback = MagicMock()

        create_clip(
            instance_path=mock_frigate_dir,
            camera="front",
            start=datetime(2025, 12, 5, 12, 0, 0),
            end=datetime(2025, 12, 5, 13, 0, 0),
            output_path=tmp_path / "output.mp4",
            progress_callback=callback,
        )

        # Should be called at least for finding and concatenating stages
        assert callback.call_count >= 2

        stages = [call[0][0].stage for call in callback.call_args_list]
        assert "finding" in stages


class TestCreateMultiCameraClip:
    """Tests for multi-camera clip creation."""

    @pytest.fixture
    def mock_multi_camera_dir(self, tmp_path):
        """Create mock Frigate directory with multiple cameras.

        Frigate structure: recordings/YYYY-MM-DD/HH/camera/MM.SS.mp4
        """
        recordings_path = tmp_path / "recordings"
        for camera in ["front", "back"]:
            camera_path = recordings_path / "2025-12-05" / "12" / camera
            camera_path.mkdir(parents=True)
            (camera_path / "00.00.mp4").touch()
        return tmp_path

    @patch("frigate_tools.clip.create_clip")
    def test_separate_creates_individual_files(
        self, mock_create_clip, mock_multi_camera_dir, tmp_path
    ):
        """Separate mode creates one file per camera."""
        mock_create_clip.return_value = True
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = create_multi_camera_clip(
            instance_path=mock_multi_camera_dir,
            cameras=["front", "back"],
            start=datetime(2025, 12, 5, 12, 0, 0),
            end=datetime(2025, 12, 5, 13, 0, 0),
            output_dir=output_dir,
            separate=True,
        )

        assert result is not None
        assert isinstance(result, dict)
        assert "front" in result
        assert "back" in result
        assert mock_create_clip.call_count == 2

    @patch("frigate_tools.clip.create_clip")
    def test_separate_returns_none_on_failure(
        self, mock_create_clip, mock_multi_camera_dir, tmp_path
    ):
        """Returns None if any camera fails."""
        mock_create_clip.side_effect = [True, False]  # First succeeds, second fails
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = create_multi_camera_clip(
            instance_path=mock_multi_camera_dir,
            cameras=["front", "back"],
            start=datetime(2025, 12, 5, 12, 0, 0),
            end=datetime(2025, 12, 5, 13, 0, 0),
            output_dir=output_dir,
            separate=True,
        )

        assert result is None

    @patch("frigate_tools.grid.create_grid_video")
    def test_grid_mode_uses_grid_module(
        self, mock_grid, mock_multi_camera_dir, tmp_path
    ):
        """Grid mode uses grid video creation."""
        mock_grid.return_value = True
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = create_multi_camera_clip(
            instance_path=mock_multi_camera_dir,
            cameras=["front", "back"],
            start=datetime(2025, 12, 5, 12, 0, 0),
            end=datetime(2025, 12, 5, 13, 0, 0),
            output_dir=output_dir,
            separate=False,
        )

        assert result is not None
        mock_grid.assert_called_once()
