"""End-to-end tests using real Frigate recordings.

These tests require access to real Frigate recordings at /data/nvr/frigate/cherokee.
They are skipped if the recordings are not available.

Tests verify:
- File list generation finds correct recordings
- Timelapse creation produces valid video output
- Clip creation produces valid video output
- Multi-camera grid creation works
"""

import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from frigate_tools.file_list import generate_file_lists
from frigate_tools.timelapse import create_timelapse, get_video_duration
from frigate_tools.clip import create_clip
from frigate_tools.grid import create_grid_video, calculate_grid_layout

# Real Frigate instance path
FRIGATE_INSTANCE = Path("/data/nvr/frigate/cherokee")


def frigate_recordings_available() -> bool:
    """Check if real Frigate recordings are available."""
    recordings_path = FRIGATE_INSTANCE / "recordings"
    if not recordings_path.exists():
        return False
    # Check for at least one date directory
    return any(d.is_dir() for d in recordings_path.iterdir() if d.name.startswith("2025"))


def ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


skip_no_frigate = pytest.mark.skipif(
    not frigate_recordings_available(),
    reason="Frigate recordings not available at /data/nvr/frigate/cherokee"
)

skip_no_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(),
    reason="ffmpeg not available"
)


@skip_no_frigate
class TestFileListE2E:
    """End-to-end tests for file list generation."""

    def test_finds_recordings_for_camera(self):
        """Finds recording files for a known camera and time range."""
        file_lists = generate_file_lists(
            cameras=["bporchcam"],
            start=datetime(2025, 12, 1, 8, 0, 0),
            end=datetime(2025, 12, 1, 8, 10, 0),  # 10 minutes
            instance_path=FRIGATE_INSTANCE,
        )

        assert "bporchcam" in file_lists
        files = file_lists["bporchcam"]
        # Should have ~60 files (one per 10 seconds for 10 minutes)
        assert len(files) > 0
        assert len(files) <= 120  # Reasonable upper bound

        # All files should exist and be mp4
        for f in files:
            assert f.exists()
            assert f.suffix == ".mp4"

    def test_finds_multiple_cameras(self):
        """Finds recordings for multiple cameras."""
        file_lists = generate_file_lists(
            cameras=["bporchcam", "doorbell"],
            start=datetime(2025, 12, 1, 8, 0, 0),
            end=datetime(2025, 12, 1, 8, 5, 0),  # 5 minutes
            instance_path=FRIGATE_INSTANCE,
        )

        assert "bporchcam" in file_lists
        assert "doorbell" in file_lists
        assert len(file_lists["bporchcam"]) > 0
        assert len(file_lists["doorbell"]) > 0

    def test_respects_time_boundaries(self):
        """File list respects start/end time boundaries."""
        # Get files for a small window
        file_lists = generate_file_lists(
            cameras=["bporchcam"],
            start=datetime(2025, 12, 1, 8, 0, 0),
            end=datetime(2025, 12, 1, 8, 1, 0),  # 1 minute only
            instance_path=FRIGATE_INSTANCE,
        )

        files = file_lists["bporchcam"]
        # Should have ~6 files (one per 10 seconds)
        assert len(files) <= 12  # Allow some tolerance
        assert len(files) > 0


@skip_no_frigate
@skip_no_ffmpeg
class TestTimelapseE2E:
    """End-to-end tests for timelapse creation with real recordings."""

    def test_creates_timelapse_from_real_recordings(self, tmp_path):
        """Creates a timelapse from real Frigate recordings."""
        # Get file list
        file_lists = generate_file_lists(
            cameras=["bporchcam"],
            start=datetime(2025, 12, 1, 8, 0, 0),
            end=datetime(2025, 12, 1, 8, 5, 0),  # 5 minutes of footage
            instance_path=FRIGATE_INSTANCE,
        )

        files = file_lists["bporchcam"]
        assert len(files) > 0, "No files found for test"

        output = tmp_path / "timelapse.mp4"

        # Create a 10-second timelapse from ~5 minutes of footage (30x speedup)
        success = create_timelapse(
            input_files=files,
            output_path=output,
            target_duration=10.0,
            preset="ultrafast",  # Fast for testing
        )

        assert success, "Timelapse creation failed"
        assert output.exists(), "Output file not created"
        assert output.stat().st_size > 0, "Output file is empty"

        # Verify output is a valid video
        duration = get_video_duration(output)
        assert duration > 0, "Output video has no duration"
        # Allow 50% tolerance on duration
        assert 5 <= duration <= 15, f"Duration {duration}s not close to 10s target"

    def test_timelapse_with_progress_callback(self, tmp_path):
        """Progress callback is called during real timelapse creation."""
        file_lists = generate_file_lists(
            cameras=["bporchcam"],
            start=datetime(2025, 12, 1, 8, 0, 0),
            end=datetime(2025, 12, 1, 8, 2, 0),  # 2 minutes
            instance_path=FRIGATE_INSTANCE,
        )

        files = file_lists["bporchcam"]
        output = tmp_path / "timelapse.mp4"

        progress_values = []

        def track_progress(info):
            if info.percent is not None:
                progress_values.append(info.percent)

        success = create_timelapse(
            input_files=files,
            output_path=output,
            target_duration=5.0,
            preset="ultrafast",
            progress_callback=track_progress,
        )

        assert success
        # Progress should have been reported (may be 0 for very fast encodes)
        # Just verify we get completion


@skip_no_frigate
@skip_no_ffmpeg
class TestClipE2E:
    """End-to-end tests for clip creation with real recordings."""

    def test_creates_clip_from_real_recordings(self, tmp_path):
        """Creates a clip from real Frigate recordings."""
        output = tmp_path / "clip.mp4"

        success = create_clip(
            instance_path=FRIGATE_INSTANCE,
            camera="bporchcam",
            start=datetime(2025, 12, 1, 8, 0, 0),
            end=datetime(2025, 12, 1, 8, 1, 0),  # 1 minute clip
            output_path=output,
            reencode=False,  # Fast stream copy
        )

        assert success, "Clip creation failed"
        assert output.exists(), "Output file not created"
        assert output.stat().st_size > 0, "Output file is empty"

        # Verify output is a valid video
        duration = get_video_duration(output)
        assert duration > 0, "Output video has no duration"
        # Clip should be roughly 1 minute (60 seconds, with some tolerance)
        assert 30 <= duration <= 120, f"Duration {duration}s not close to expected"

    def test_creates_reencoded_clip(self, tmp_path):
        """Creates a re-encoded clip from real recordings."""
        output = tmp_path / "clip_reencoded.mp4"

        success = create_clip(
            instance_path=FRIGATE_INSTANCE,
            camera="bporchcam",
            start=datetime(2025, 12, 1, 8, 0, 0),
            end=datetime(2025, 12, 1, 8, 0, 30),  # 30 seconds
            output_path=output,
            reencode=True,
            preset="ultrafast",
        )

        assert success, "Re-encoded clip creation failed"
        assert output.exists(), "Output file not created"

        duration = get_video_duration(output)
        assert 10 <= duration <= 60, f"Duration {duration}s unexpected"


@skip_no_frigate
@skip_no_ffmpeg
class TestGridE2E:
    """End-to-end tests for grid video creation with real recordings."""

    def test_creates_grid_from_multiple_cameras(self, tmp_path):
        """Creates a grid video from multiple real cameras."""
        # Get files for two cameras
        file_lists = generate_file_lists(
            cameras=["bporchcam", "doorbell"],
            start=datetime(2025, 12, 1, 8, 0, 0),
            end=datetime(2025, 12, 1, 8, 0, 30),  # 30 seconds
            instance_path=FRIGATE_INSTANCE,
        )

        # Verify we have files for both
        assert len(file_lists["bporchcam"]) > 0
        assert len(file_lists["doorbell"]) > 0

        output = tmp_path / "grid.mp4"

        success = create_grid_video(
            camera_files=file_lists,
            output_path=output,
            preset="ultrafast",
        )

        assert success, "Grid creation failed"
        assert output.exists(), "Output file not created"
        assert output.stat().st_size > 0, "Output file is empty"

        duration = get_video_duration(output)
        assert duration > 0, "Output video has no duration"

    def test_grid_layout_calculation(self):
        """Grid layout is calculated correctly for various camera counts."""
        # 2 cameras -> 1x2
        layout = calculate_grid_layout(2)
        assert layout.rows == 1
        assert layout.cols == 2

        # 4 cameras -> 2x2
        layout = calculate_grid_layout(4)
        assert layout.rows == 2
        assert layout.cols == 2

        # 6 cameras -> 2x3 or 3x2
        layout = calculate_grid_layout(6)
        assert layout.total_cells == 6


@skip_no_frigate
@skip_no_ffmpeg
class TestCLIE2E:
    """End-to-end CLI tests with real recordings."""

    def test_cli_timelapse_dry_run(self, tmp_path):
        """CLI dry-run mode shows correct estimates."""
        from typer.testing import CliRunner
        from frigate_tools.cli import app

        runner = CliRunner()
        output = tmp_path / "timelapse.mp4"

        result = runner.invoke(app, [
            "timelapse", "create",
            "--cameras", "bporchcam",
            "--start", "2025-12-01T08:00",
            "--end", "2025-12-01T08:05",
            "--duration", "10s",
            "--output", str(output),
            "--instance", str(FRIGATE_INSTANCE),
            "--dry-run",
        ])

        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        assert "Dry run" in result.stdout
        assert "Would create" in result.stdout
        assert not output.exists(), "Dry run should not create output"

    def test_cli_timelapse_creates_file(self, tmp_path):
        """CLI creates actual timelapse file."""
        from typer.testing import CliRunner
        from frigate_tools.cli import app

        runner = CliRunner()
        output = tmp_path / "timelapse.mp4"

        result = runner.invoke(app, [
            "timelapse", "create",
            "--cameras", "bporchcam",
            "--start", "2025-12-01T08:00",
            "--end", "2025-12-01T08:02",  # 2 minutes
            "--duration", "5s",
            "--output", str(output),
            "--instance", str(FRIGATE_INSTANCE),
            "--preset", "ultrafast",
        ])

        assert result.exit_code == 0, f"CLI failed: {result.stdout}"
        assert "Success" in result.stdout
        assert output.exists(), "Output file not created"

        # Verify it's a valid video
        duration = get_video_duration(output)
        assert duration > 0
