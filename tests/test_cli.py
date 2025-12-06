"""Tests for CLI module."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from frigate_tools.cli import app, parse_duration, find_frigate_instance, DEFAULT_FRIGATE_PATHS

runner = CliRunner()


def test_app_shows_help_with_no_args() -> None:
    """App shows help when invoked with no arguments (exit code 2 is expected for no_args_is_help)."""
    result = runner.invoke(app)
    # Typer uses exit code 2 for no_args_is_help
    assert result.exit_code in (0, 2)
    assert "Usage" in result.stdout or "timelapse" in result.stdout


def test_app_help_flag() -> None:
    """App responds to --help flag."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "timelapse" in result.stdout
    assert "clip" in result.stdout


def test_timelapse_subcommand_exists() -> None:
    """Timelapse subcommand is available."""
    result = runner.invoke(app, ["timelapse", "--help"])
    assert result.exit_code == 0
    assert "timelapse" in result.stdout.lower() or "Generate" in result.stdout


def test_clip_subcommand_exists() -> None:
    """Clip subcommand is available."""
    result = runner.invoke(app, ["clip", "--help"])
    assert result.exit_code == 0
    assert "clip" in result.stdout.lower() or "Export" in result.stdout


class TestParseDuration:
    """Tests for parse_duration function."""

    def test_parse_minutes(self):
        """Parses minutes correctly."""
        assert parse_duration("5m") == 300.0
        assert parse_duration("30m") == 1800.0

    def test_parse_hours(self):
        """Parses hours correctly."""
        assert parse_duration("1h") == 3600.0
        assert parse_duration("2h") == 7200.0

    def test_parse_seconds(self):
        """Parses seconds correctly."""
        assert parse_duration("90s") == 90.0
        assert parse_duration("30s") == 30.0

    def test_parse_combined(self):
        """Parses combined durations."""
        assert parse_duration("1h30m") == 5400.0
        assert parse_duration("2h30m15s") == 9015.0
        assert parse_duration("1h15s") == 3615.0

    def test_parse_uppercase(self):
        """Parses uppercase letters."""
        assert parse_duration("5M") == 300.0
        assert parse_duration("1H") == 3600.0

    def test_parse_with_spaces(self):
        """Handles leading/trailing spaces."""
        assert parse_duration("  5m  ") == 300.0

    def test_parse_invalid_format(self):
        """Raises ValueError for invalid format."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("invalid")

    def test_parse_empty_string(self):
        """Raises ValueError for empty string."""
        with pytest.raises(ValueError, match="Invalid duration format"):
            parse_duration("")

    def test_parse_zero_duration(self):
        """Raises ValueError for zero duration."""
        with pytest.raises(ValueError, match="Duration must be greater than 0"):
            parse_duration("0m")


class TestFindFrigateInstance:
    """Tests for find_frigate_instance function."""

    def test_finds_direct_recordings_dir(self, tmp_path):
        """Finds instance when recordings/ is directly under path."""
        # Create structure
        recordings = tmp_path / "recordings"
        recordings.mkdir()

        with patch(
            "frigate_tools.cli.DEFAULT_FRIGATE_PATHS",
            [tmp_path]
        ):
            result = find_frigate_instance()
            assert result == tmp_path

    def test_finds_named_instance(self, tmp_path):
        """Finds named instance subdirectory."""
        # Create structure: tmp_path/cherokee/recordings
        instance = tmp_path / "cherokee"
        recordings = instance / "recordings"
        recordings.mkdir(parents=True)

        with patch(
            "frigate_tools.cli.DEFAULT_FRIGATE_PATHS",
            [tmp_path]
        ):
            result = find_frigate_instance()
            assert result == instance

    def test_returns_none_when_not_found(self):
        """Returns None when no instance found."""
        with patch(
            "frigate_tools.cli.DEFAULT_FRIGATE_PATHS",
            [Path("/nonexistent/path")]
        ):
            result = find_frigate_instance()
            assert result is None


class TestTimelapseCreateCommand:
    """Tests for timelapse create CLI command."""

    def test_help_shows_options(self):
        """Help shows all expected options."""
        result = runner.invoke(app, ["timelapse", "create", "--help"])
        assert result.exit_code == 0
        assert "--cameras" in result.stdout
        assert "--start" in result.stdout
        assert "--end" in result.stdout
        assert "--duration" in result.stdout
        assert "--output" in result.stdout
        assert "--instance" in result.stdout
        assert "--skip-days" in result.stdout
        assert "--skip-hours" in result.stdout
        assert "--preset" in result.stdout

    def test_requires_cameras(self):
        """Requires cameras option."""
        result = runner.invoke(app, [
            "timelapse", "create",
            "--start", "2025-12-01T08:00",
            "--end", "2025-12-01T12:00",
            "--duration", "5m",
            "--output", "test.mp4",
        ])
        # Should fail with missing required option
        assert result.exit_code != 0

    def test_requires_all_options(self):
        """Requires all mandatory options."""
        result = runner.invoke(app, [
            "timelapse", "create",
            "--cameras", "front",
        ])
        assert result.exit_code != 0

    @patch("frigate_tools.cli.find_frigate_instance")
    @patch("frigate_tools.cli.generate_file_lists")
    @patch("frigate_tools.cli.create_timelapse")
    def test_creates_single_camera_timelapse(
        self, mock_create, mock_file_lists, mock_find, tmp_path
    ):
        """Creates timelapse for single camera."""
        # Setup mocks
        mock_find.return_value = tmp_path
        mock_file_lists.return_value = {
            "front": [tmp_path / "file1.mp4", tmp_path / "file2.mp4"]
        }
        mock_create.return_value = True

        output = tmp_path / "output.mp4"
        # Create output file for size check
        output.write_bytes(b"x" * 1000)

        result = runner.invoke(app, [
            "timelapse", "create",
            "--cameras", "front",
            "--start", "2025-12-01T08:00",
            "--end", "2025-12-01T12:00",
            "--duration", "5m",
            "--output", str(output),
        ])

        # Check result
        assert result.exit_code == 0, result.stdout
        assert "Success" in result.stdout

    @patch("frigate_tools.cli.find_frigate_instance")
    @patch("frigate_tools.cli.generate_file_lists")
    @patch("frigate_tools.cli.encode_timelapse")
    @patch("frigate_tools.cli.create_grid_video")
    def test_creates_multi_camera_grid(
        self, mock_grid, mock_encode, mock_file_lists, mock_find, tmp_path
    ):
        """Creates grid timelapse for multiple cameras."""
        mock_find.return_value = tmp_path
        mock_file_lists.return_value = {
            "front": [tmp_path / "file1.mp4"],
            "back": [tmp_path / "file2.mp4"],
        }
        mock_grid.return_value = True
        mock_encode.return_value = True

        output = tmp_path / "output.mp4"
        output.write_bytes(b"x" * 1000)

        result = runner.invoke(app, [
            "timelapse", "create",
            "--cameras", "front,back",
            "--start", "2025-12-01T08:00",
            "--end", "2025-12-01T12:00",
            "--duration", "5m",
            "--output", str(output),
        ])

        assert result.exit_code == 0, result.stdout
        assert "Grid layout" in result.stdout
        # Verify both steps were called
        mock_grid.assert_called_once()
        mock_encode.assert_called_once()

    @patch("frigate_tools.cli.find_frigate_instance")
    def test_fails_when_no_instance_found(self, mock_find, tmp_path):
        """Fails gracefully when no Frigate instance found."""
        mock_find.return_value = None

        result = runner.invoke(app, [
            "timelapse", "create",
            "--cameras", "front",
            "--start", "2025-12-01T08:00",
            "--end", "2025-12-01T12:00",
            "--duration", "5m",
            "--output", str(tmp_path / "output.mp4"),
        ])

        assert result.exit_code == 1
        assert "Could not auto-detect" in result.stdout

    @patch("frigate_tools.cli.find_frigate_instance")
    @patch("frigate_tools.cli.generate_file_lists")
    def test_fails_when_no_files_found(self, mock_file_lists, mock_find, tmp_path):
        """Fails gracefully when no recording files found."""
        mock_find.return_value = tmp_path
        mock_file_lists.return_value = {"front": []}

        result = runner.invoke(app, [
            "timelapse", "create",
            "--cameras", "front",
            "--start", "2025-12-01T08:00",
            "--end", "2025-12-01T12:00",
            "--duration", "5m",
            "--output", str(tmp_path / "output.mp4"),
        ])

        assert result.exit_code == 1
        assert "No recording files found" in result.stdout

    def test_invalid_duration_format(self, tmp_path):
        """Reports error for invalid duration format."""
        result = runner.invoke(app, [
            "timelapse", "create",
            "--cameras", "front",
            "--start", "2025-12-01T08:00",
            "--end", "2025-12-01T12:00",
            "--duration", "invalid",
            "--output", str(tmp_path / "output.mp4"),
            "--instance", str(tmp_path),
        ])

        assert result.exit_code == 1
        assert "Invalid duration format" in result.stdout
