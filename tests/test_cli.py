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
