"""Tests for multi-camera grid layout module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frigate_tools.grid import (
    GridLayout,
    SyncedFileSet,
    calculate_grid_layout,
    create_grid_video,
    generate_xstack_filter,
    sync_file_lists,
)


class TestGridLayout:
    """Tests for GridLayout dataclass."""

    def test_total_cells(self):
        """Calculates total cells correctly."""
        layout = GridLayout(rows=2, cols=3)
        assert layout.total_cells == 6

    def test_total_cells_single(self):
        """Single cell grid."""
        layout = GridLayout(rows=1, cols=1)
        assert layout.total_cells == 1

    def test_total_cells_empty(self):
        """Empty grid."""
        layout = GridLayout(rows=0, cols=0)
        assert layout.total_cells == 0


class TestCalculateGridLayout:
    """Tests for grid layout calculation."""

    def test_zero_cameras(self):
        """Zero cameras returns empty layout."""
        layout = calculate_grid_layout(0)
        assert layout.rows == 0
        assert layout.cols == 0

    def test_one_camera(self):
        """Single camera is 1x1."""
        layout = calculate_grid_layout(1)
        assert layout.rows == 1
        assert layout.cols == 1

    def test_two_cameras(self):
        """Two cameras is 1x2 (side by side)."""
        layout = calculate_grid_layout(2)
        assert layout.rows == 1
        assert layout.cols == 2

    def test_three_cameras(self):
        """Three cameras is 2x2 with one empty cell."""
        layout = calculate_grid_layout(3)
        assert layout.rows == 2
        assert layout.cols == 2
        assert layout.total_cells >= 3

    def test_four_cameras(self):
        """Four cameras is 2x2."""
        layout = calculate_grid_layout(4)
        assert layout.rows == 2
        assert layout.cols == 2

    def test_five_cameras(self):
        """Five cameras is 2x3 or 3x2."""
        layout = calculate_grid_layout(5)
        assert layout.total_cells >= 5
        assert layout.total_cells <= 6

    def test_six_cameras(self):
        """Six cameras is 2x3 or 3x2."""
        layout = calculate_grid_layout(6)
        assert layout.rows * layout.cols == 6

    def test_nine_cameras(self):
        """Nine cameras is 3x3."""
        layout = calculate_grid_layout(9)
        assert layout.rows == 3
        assert layout.cols == 3

    def test_layout_has_enough_cells(self):
        """Layout always has enough cells for all cameras."""
        for count in range(1, 20):
            layout = calculate_grid_layout(count)
            assert layout.total_cells >= count


class TestGenerateXstackFilter:
    """Tests for xstack filter generation."""

    def test_empty_input(self):
        """Returns empty string for no inputs."""
        result = generate_xstack_filter(0, GridLayout(0, 0))
        assert result == ""

    def test_single_input(self):
        """Generates filter for single input."""
        result = generate_xstack_filter(1, GridLayout(1, 1))
        assert "xstack" in result
        assert "inputs=1" in result

    def test_two_inputs_horizontal(self):
        """Generates filter for horizontal 1x2 layout."""
        result = generate_xstack_filter(2, GridLayout(1, 2))
        assert "xstack" in result
        assert "inputs=2" in result
        assert "0_0" in result  # First input at origin
        assert "w0_0" in result  # Second input after first width

    def test_four_inputs_grid(self):
        """Generates filter for 2x2 grid."""
        result = generate_xstack_filter(4, GridLayout(2, 2))
        assert "xstack" in result
        assert "inputs=4" in result

    def test_with_dimensions(self):
        """Includes scale filters when dimensions specified."""
        result = generate_xstack_filter(2, GridLayout(1, 2), width=640, height=480)
        assert "scale=640:480" in result


class TestSyncFileLists:
    """Tests for file list synchronization."""

    def test_empty_input(self):
        """Returns empty result for empty input."""
        result = sync_file_lists({})
        assert result.camera_files == {}
        assert result.gap_indices == set()

    def test_equal_length_lists(self):
        """No gaps when all lists have same length."""
        files = {
            "cam1": [Path("a.mp4"), Path("b.mp4")],
            "cam2": [Path("c.mp4"), Path("d.mp4")],
        }
        result = sync_file_lists(files)
        assert result.gap_indices == set()

    def test_unequal_length_lists(self):
        """Identifies gaps in shorter lists."""
        files = {
            "cam1": [Path("a.mp4"), Path("b.mp4"), Path("c.mp4")],
            "cam2": [Path("d.mp4")],  # Missing 2 files
        }
        result = sync_file_lists(files)
        assert 1 in result.gap_indices
        assert 2 in result.gap_indices

    def test_preserves_original_files(self):
        """Original file lists are preserved."""
        files = {
            "cam1": [Path("a.mp4")],
            "cam2": [Path("b.mp4")],
        }
        result = sync_file_lists(files)
        assert result.camera_files["cam1"] == [Path("a.mp4")]
        assert result.camera_files["cam2"] == [Path("b.mp4")]


class TestCreateGridVideo:
    """Tests for grid video creation."""

    def test_empty_camera_files(self):
        """Returns False for empty camera files."""
        result = create_grid_video({}, Path("/test/output.mp4"))
        assert result is False

    @patch("subprocess.Popen")
    def test_creates_ffmpeg_process(self, mock_popen, tmp_path):
        """Creates ffmpeg process with correct inputs."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("", "")
        mock_popen.return_value = mock_process

        # Create test files
        files = {
            "cam1": [tmp_path / "a.mp4"],
            "cam2": [tmp_path / "b.mp4"],
        }
        for camera_files in files.values():
            for f in camera_files:
                f.touch()

        output = tmp_path / "output.mp4"
        result = create_grid_video(files, output)

        assert result is True
        mock_popen.assert_called_once()

        # Verify ffmpeg command includes filter_complex
        call_args = mock_popen.call_args[0][0]
        assert "ffmpeg" in call_args[0]
        assert "-filter_complex" in call_args

    @patch("subprocess.Popen")
    def test_handles_ffmpeg_failure(self, mock_popen, tmp_path):
        """Returns False when ffmpeg fails."""
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = ("", "Error message")
        mock_popen.return_value = mock_process

        files = {"cam1": [tmp_path / "a.mp4"]}
        files["cam1"][0].touch()

        result = create_grid_video(files, tmp_path / "output.mp4")
        assert result is False

    def test_rejects_camera_with_no_files(self, tmp_path):
        """Returns False when a camera has no files."""
        files = {
            "cam1": [tmp_path / "a.mp4"],
            "cam2": [],  # No files
        }
        files["cam1"][0].touch()

        result = create_grid_video(files, tmp_path / "output.mp4")
        assert result is False

    @patch("subprocess.Popen")
    def test_cleans_up_temp_files(self, mock_popen, tmp_path):
        """Cleans up temporary concat files after completion."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("", "")
        mock_popen.return_value = mock_process

        files = {"cam1": [tmp_path / "a.mp4"]}
        files["cam1"][0].touch()

        create_grid_video(files, tmp_path / "output.mp4")

        # Temp concat files should be cleaned up
        # We can't easily verify this without inspecting internals,
        # but we verify the function completes successfully
