"""Tests for file list generation with calendar filtering."""

from datetime import datetime
from pathlib import Path

import pytest

from frigate_tools.file_list import (
    HourRange,
    find_recording_files,
    generate_file_lists,
    parse_file_timestamp,
    parse_skip_days,
    parse_skip_hours,
    should_skip_timestamp,
)


class TestHourRange:
    """Tests for HourRange class."""

    def test_normal_range_contains(self):
        """Normal range like 9-17 contains hours within bounds."""
        r = HourRange(9, 17)
        assert r.contains(9) is True
        assert r.contains(12) is True
        assert r.contains(16) is True
        assert r.contains(17) is False  # end is exclusive
        assert r.contains(8) is False
        assert r.contains(18) is False

    def test_wrapping_range_contains(self):
        """Wrapping range like 16-8 spans midnight."""
        r = HourRange(16, 8)
        # After 16:00
        assert r.contains(16) is True
        assert r.contains(20) is True
        assert r.contains(23) is True
        # Before 08:00
        assert r.contains(0) is True
        assert r.contains(5) is True
        assert r.contains(7) is True
        # Outside range
        assert r.contains(8) is False
        assert r.contains(12) is False
        assert r.contains(15) is False


class TestParseSkipDays:
    """Tests for parse_skip_days function."""

    def test_short_day_names(self):
        """Parses short day names."""
        result = parse_skip_days(["sat", "sun"])
        assert result == {5, 6}

    def test_full_day_names(self):
        """Parses full day names."""
        result = parse_skip_days(["saturday", "sunday"])
        assert result == {5, 6}

    def test_mixed_case(self):
        """Handles mixed case."""
        result = parse_skip_days(["SAT", "Sun", "MONDAY"])
        assert result == {0, 5, 6}

    def test_all_weekdays(self):
        """Parses all weekdays."""
        result = parse_skip_days(["mon", "tue", "wed", "thu", "fri"])
        assert result == {0, 1, 2, 3, 4}

    def test_invalid_days_ignored(self):
        """Invalid day names are ignored."""
        result = parse_skip_days(["sat", "invalid", "sun"])
        assert result == {5, 6}

    def test_empty_list(self):
        """Empty list returns empty set."""
        result = parse_skip_days([])
        assert result == set()


class TestParseSkipHours:
    """Tests for parse_skip_hours function."""

    def test_normal_range(self):
        """Parses normal hour range."""
        result = parse_skip_hours(["9-17"])
        assert len(result) == 1
        assert result[0].start == 9
        assert result[0].end == 17

    def test_wrapping_range(self):
        """Parses wrapping hour range."""
        result = parse_skip_hours(["16-8"])
        assert len(result) == 1
        assert result[0].start == 16
        assert result[0].end == 8

    def test_multiple_ranges(self):
        """Parses multiple hour ranges."""
        result = parse_skip_hours(["0-6", "18-24"])
        assert len(result) == 1  # 18-24 is invalid (24 > 23)
        assert result[0].start == 0
        assert result[0].end == 6

    def test_invalid_format_ignored(self):
        """Invalid formats are ignored."""
        result = parse_skip_hours(["invalid", "9-17", "abc-def"])
        assert len(result) == 1
        assert result[0].start == 9

    def test_empty_list(self):
        """Empty list returns empty list."""
        result = parse_skip_hours([])
        assert result == []


class TestParseFileTimestamp:
    """Tests for parse_file_timestamp function."""

    def test_valid_timestamp(self):
        """Parses valid Frigate filename structure."""
        ts = parse_file_timestamp("2025-12-01", "14", "30.45.mp4")
        assert ts == datetime(2025, 12, 1, 14, 30, 45)

    def test_midnight(self):
        """Parses midnight timestamp."""
        ts = parse_file_timestamp("2025-01-15", "00", "00.00.mp4")
        assert ts == datetime(2025, 1, 15, 0, 0, 0)

    def test_end_of_day(self):
        """Parses end of day timestamp."""
        ts = parse_file_timestamp("2025-06-30", "23", "59.59.mp4")
        assert ts == datetime(2025, 6, 30, 23, 59, 59)

    def test_invalid_filename_pattern(self):
        """Returns None for invalid filename pattern."""
        assert parse_file_timestamp("2025-12-01", "14", "invalid.mp4") is None
        assert parse_file_timestamp("2025-12-01", "14", "30-45.mp4") is None
        assert parse_file_timestamp("2025-12-01", "14", "30.45.txt") is None

    def test_invalid_date_format(self):
        """Returns None for invalid date format."""
        assert parse_file_timestamp("invalid", "14", "30.45.mp4") is None
        assert parse_file_timestamp("2025/12/01", "14", "30.45.mp4") is None


class TestShouldSkipTimestamp:
    """Tests for should_skip_timestamp function."""

    def test_skip_weekend_saturday(self):
        """Skips Saturday when weekend days are filtered."""
        ts = datetime(2025, 12, 6, 12, 0, 0)  # Saturday
        assert should_skip_timestamp(ts, {5, 6}, []) is True

    def test_skip_weekend_sunday(self):
        """Skips Sunday when weekend days are filtered."""
        ts = datetime(2025, 12, 7, 12, 0, 0)  # Sunday
        assert should_skip_timestamp(ts, {5, 6}, []) is True

    def test_keep_weekday(self):
        """Keeps weekday when weekend days are filtered."""
        ts = datetime(2025, 12, 5, 12, 0, 0)  # Friday
        assert should_skip_timestamp(ts, {5, 6}, []) is False

    def test_skip_hours_normal_range(self):
        """Skips hours in normal range."""
        ts = datetime(2025, 12, 5, 20, 0, 0)  # 8pm Friday
        assert should_skip_timestamp(ts, set(), [HourRange(18, 22)]) is True

    def test_keep_hours_outside_range(self):
        """Keeps hours outside range."""
        ts = datetime(2025, 12, 5, 12, 0, 0)  # noon
        assert should_skip_timestamp(ts, set(), [HourRange(18, 22)]) is False

    def test_skip_hours_wrapping_range(self):
        """Skips hours in wrapping range (overnight)."""
        # 16-8 means skip 4pm to 8am
        ts_evening = datetime(2025, 12, 5, 20, 0, 0)  # 8pm
        ts_early = datetime(2025, 12, 5, 6, 0, 0)  # 6am
        assert should_skip_timestamp(ts_evening, set(), [HourRange(16, 8)]) is True
        assert should_skip_timestamp(ts_early, set(), [HourRange(16, 8)]) is True

    def test_keep_hours_inside_wrapping_gap(self):
        """Keeps hours inside gap of wrapping range."""
        # 16-8 keeps 8am to 4pm
        ts = datetime(2025, 12, 5, 12, 0, 0)  # noon
        assert should_skip_timestamp(ts, set(), [HourRange(16, 8)]) is False

    def test_combined_filters(self):
        """Skips when either day or hour filter matches."""
        skip_days = {5, 6}  # weekend
        skip_hours = [HourRange(16, 8)]  # 4pm-8am

        # Saturday noon - skipped by day
        assert should_skip_timestamp(datetime(2025, 12, 6, 12, 0, 0), skip_days, skip_hours) is True

        # Friday 8pm - skipped by hour
        assert should_skip_timestamp(datetime(2025, 12, 5, 20, 0, 0), skip_days, skip_hours) is True

        # Friday noon - kept
        assert should_skip_timestamp(datetime(2025, 12, 5, 12, 0, 0), skip_days, skip_hours) is False


class TestFindRecordingFiles:
    """Tests for find_recording_files function."""

    @pytest.fixture
    def mock_frigate_dir(self, tmp_path):
        """Create a mock Frigate directory structure."""
        # Create structure: instance/recordings/camera/YYYY-MM-DD/HH/MM.SS.mp4
        camera_path = tmp_path / "recordings" / "front"

        # Dec 5, 2025 (Friday) - multiple hours
        dec5 = camera_path / "2025-12-05"
        (dec5 / "08").mkdir(parents=True)
        (dec5 / "08" / "00.00.mp4").touch()
        (dec5 / "08" / "30.00.mp4").touch()

        (dec5 / "12").mkdir(parents=True)
        (dec5 / "12" / "00.00.mp4").touch()
        (dec5 / "12" / "30.00.mp4").touch()

        (dec5 / "20").mkdir(parents=True)
        (dec5 / "20" / "00.00.mp4").touch()

        # Dec 6, 2025 (Saturday)
        dec6 = camera_path / "2025-12-06"
        (dec6 / "10").mkdir(parents=True)
        (dec6 / "10" / "00.00.mp4").touch()
        (dec6 / "10" / "30.00.mp4").touch()

        return tmp_path

    def test_find_all_files_in_range(self, mock_frigate_dir):
        """Finds all files in time range without filters."""
        files = find_recording_files(
            instance_path=mock_frigate_dir,
            camera="front",
            start=datetime(2025, 12, 5, 0, 0, 0),
            end=datetime(2025, 12, 7, 0, 0, 0),
        )
        assert len(files) == 7

    def test_find_files_partial_range(self, mock_frigate_dir):
        """Finds files in partial time range."""
        files = find_recording_files(
            instance_path=mock_frigate_dir,
            camera="front",
            start=datetime(2025, 12, 5, 10, 0, 0),
            end=datetime(2025, 12, 5, 15, 0, 0),
        )
        # Should only get noon files (12:00 and 12:30)
        assert len(files) == 2

    def test_filter_by_skip_days(self, mock_frigate_dir):
        """Filters out weekend days."""
        files = find_recording_files(
            instance_path=mock_frigate_dir,
            camera="front",
            start=datetime(2025, 12, 5, 0, 0, 0),
            end=datetime(2025, 12, 7, 0, 0, 0),
            skip_days={5, 6},  # Skip Saturday and Sunday
        )
        # Only Friday files (Dec 5)
        assert len(files) == 5

    def test_filter_by_skip_hours(self, mock_frigate_dir):
        """Filters out specific hours."""
        files = find_recording_files(
            instance_path=mock_frigate_dir,
            camera="front",
            start=datetime(2025, 12, 5, 0, 0, 0),
            end=datetime(2025, 12, 7, 0, 0, 0),
            skip_hours=[HourRange(16, 8)],  # Skip 4pm to 8am (keep 8am-4pm)
        )
        # Only files from 8am-4pm:
        # Dec 5: 08:00, 08:30, 12:00, 12:30 (4 files)
        # Dec 6: 10:00, 10:30 (2 files)
        # Dec 5 20:00 is skipped (8pm)
        assert len(files) == 6

    def test_nonexistent_camera(self, mock_frigate_dir):
        """Returns empty list for nonexistent camera."""
        files = find_recording_files(
            instance_path=mock_frigate_dir,
            camera="nonexistent",
            start=datetime(2025, 12, 5, 0, 0, 0),
            end=datetime(2025, 12, 7, 0, 0, 0),
        )
        assert files == []

    def test_files_are_sorted(self, mock_frigate_dir):
        """Files are returned in sorted order."""
        files = find_recording_files(
            instance_path=mock_frigate_dir,
            camera="front",
            start=datetime(2025, 12, 5, 0, 0, 0),
            end=datetime(2025, 12, 7, 0, 0, 0),
        )
        file_strs = [str(f) for f in files]
        assert file_strs == sorted(file_strs)


class TestGenerateFileLists:
    """Tests for generate_file_lists function."""

    @pytest.fixture
    def mock_multi_camera_dir(self, tmp_path):
        """Create mock Frigate dir with multiple cameras."""
        for camera in ["front", "back"]:
            camera_path = tmp_path / "recordings" / camera / "2025-12-05" / "12"
            camera_path.mkdir(parents=True)
            (camera_path / "00.00.mp4").touch()
            (camera_path / "30.00.mp4").touch()
        return tmp_path

    def test_generate_lists_multiple_cameras(self, mock_multi_camera_dir):
        """Generates file lists for multiple cameras."""
        result = generate_file_lists(
            cameras=["front", "back"],
            start=datetime(2025, 12, 5, 0, 0, 0),
            end=datetime(2025, 12, 6, 0, 0, 0),
            instance_path=mock_multi_camera_dir,
        )

        assert "front" in result
        assert "back" in result
        assert len(result["front"]) == 2
        assert len(result["back"]) == 2

    def test_generate_lists_with_skip_days(self, mock_multi_camera_dir):
        """Generates file lists with day filtering."""
        result = generate_file_lists(
            cameras=["front"],
            start=datetime(2025, 12, 5, 0, 0, 0),  # Friday
            end=datetime(2025, 12, 6, 0, 0, 0),
            instance_path=mock_multi_camera_dir,
            skip_days=["fri"],  # Skip Friday
        )

        assert len(result["front"]) == 0

    def test_generate_lists_with_skip_hours(self, mock_multi_camera_dir):
        """Generates file lists with hour filtering."""
        result = generate_file_lists(
            cameras=["front"],
            start=datetime(2025, 12, 5, 0, 0, 0),
            end=datetime(2025, 12, 6, 0, 0, 0),
            instance_path=mock_multi_camera_dir,
            skip_hours=["10-14"],  # Skip 10am-2pm (includes noon)
        )

        assert len(result["front"]) == 0
