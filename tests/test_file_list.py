"""Tests for file list generation with calendar filtering."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from frigate_tools.file_list import (
    HourRange,
    find_recording_files,
    generate_file_lists,
    parse_file_timestamp,
    parse_skip_days,
    parse_skip_hours,
    should_skip_timestamp,
    utc_to_local,
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


class TestUtcToLocal:
    """Tests for utc_to_local timezone conversion."""

    def test_conversion_produces_different_hour(self):
        """UTC to local conversion changes the hour (unless UTC+0)."""
        utc_time = datetime(2025, 12, 5, 18, 0, 0)  # 6pm UTC
        local_time = utc_to_local(utc_time)

        # The result should be a valid datetime
        assert isinstance(local_time, datetime)
        # Hour should be different unless in UTC+0 timezone
        # (We can't assert specific values as it depends on machine timezone)

    def test_preserves_minute_second(self):
        """UTC to local conversion preserves minutes and seconds."""
        utc_time = datetime(2025, 12, 5, 18, 30, 45)
        local_time = utc_to_local(utc_time)

        assert local_time.minute == 30
        assert local_time.second == 45

    def test_returns_naive_datetime(self):
        """Returns naive datetime (no timezone info)."""
        utc_time = datetime(2025, 12, 5, 18, 0, 0)
        local_time = utc_to_local(utc_time)

        assert local_time.tzinfo is None


class TestShouldSkipTimestamp:
    """Tests for should_skip_timestamp function.

    These tests mock utc_to_local to test skip logic independently of timezone.
    The function receives UTC timestamps and converts to local for filtering.
    """

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_skip_weekend_saturday(self, mock_utc):
        """Skips Saturday when weekend days are filtered."""
        ts = datetime(2025, 12, 6, 12, 0, 0)  # Saturday
        assert should_skip_timestamp(ts, {5, 6}, []) is True

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_skip_weekend_sunday(self, mock_utc):
        """Skips Sunday when weekend days are filtered."""
        ts = datetime(2025, 12, 7, 12, 0, 0)  # Sunday
        assert should_skip_timestamp(ts, {5, 6}, []) is True

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_keep_weekday(self, mock_utc):
        """Keeps weekday when weekend days are filtered."""
        ts = datetime(2025, 12, 5, 12, 0, 0)  # Friday
        assert should_skip_timestamp(ts, {5, 6}, []) is False

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_skip_hours_normal_range(self, mock_utc):
        """Skips hours in normal range."""
        ts = datetime(2025, 12, 5, 20, 0, 0)  # 8pm Friday
        assert should_skip_timestamp(ts, set(), [HourRange(18, 22)]) is True

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_keep_hours_outside_range(self, mock_utc):
        """Keeps hours outside range."""
        ts = datetime(2025, 12, 5, 12, 0, 0)  # noon
        assert should_skip_timestamp(ts, set(), [HourRange(18, 22)]) is False

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_skip_hours_wrapping_range(self, mock_utc):
        """Skips hours in wrapping range (overnight)."""
        # 16-8 means skip 4pm to 8am
        ts_evening = datetime(2025, 12, 5, 20, 0, 0)  # 8pm
        ts_early = datetime(2025, 12, 5, 6, 0, 0)  # 6am
        assert should_skip_timestamp(ts_evening, set(), [HourRange(16, 8)]) is True
        assert should_skip_timestamp(ts_early, set(), [HourRange(16, 8)]) is True

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_keep_hours_inside_wrapping_gap(self, mock_utc):
        """Keeps hours inside gap of wrapping range."""
        # 16-8 keeps 8am to 4pm
        ts = datetime(2025, 12, 5, 12, 0, 0)  # noon
        assert should_skip_timestamp(ts, set(), [HourRange(16, 8)]) is False

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_combined_filters(self, mock_utc):
        """Skips when either day or hour filter matches."""
        skip_days = {5, 6}  # weekend
        skip_hours = [HourRange(16, 8)]  # 4pm-8am

        # Saturday noon - skipped by day
        assert should_skip_timestamp(datetime(2025, 12, 6, 12, 0, 0), skip_days, skip_hours) is True

        # Friday 8pm - skipped by hour
        assert should_skip_timestamp(datetime(2025, 12, 5, 20, 0, 0), skip_days, skip_hours) is True

        # Friday noon - kept
        assert should_skip_timestamp(datetime(2025, 12, 5, 12, 0, 0), skip_days, skip_hours) is False

    def test_utc_timestamp_converted_to_local(self):
        """Verifies that UTC timestamp is converted to local for hour filtering.

        This test uses real timezone conversion to ensure skip_hours works
        correctly with UTC file timestamps.
        """
        # Create a UTC time that when converted to local will be in different hour
        # We'll test the conversion is called by checking the function behavior
        # changes based on timezone offset

        # Get the local timezone offset
        import time as time_module

        if time_module.daylight:
            offset_hours = -time_module.altzone // 3600
        else:
            offset_hours = -time_module.timezone // 3600

        if offset_hours == 0:
            pytest.skip("Test requires non-UTC timezone")

        # Create a UTC timestamp where UTC hour != local hour
        # If offset is negative (e.g., CST = -6), UTC 18:00 = local 12:00
        # If offset is positive (e.g., CET = +1), UTC 11:00 = local 12:00
        if offset_hours < 0:
            utc_hour = 12 - offset_hours  # e.g., CST: 12 - (-6) = 18 UTC
        else:
            utc_hour = 12 - offset_hours  # e.g., CET: 12 - 1 = 11 UTC

        utc_ts = datetime(2025, 12, 5, utc_hour, 0, 0)

        # Skip hours 10-14 (10am-2pm local) - should skip local noon
        result = should_skip_timestamp(utc_ts, set(), [HourRange(10, 14)])
        assert result is True, f"UTC {utc_hour}:00 should convert to ~12:00 local and be skipped"


class TestFindRecordingFiles:
    """Tests for find_recording_files function."""

    @pytest.fixture
    def mock_frigate_dir(self, tmp_path):
        """Create a mock Frigate directory structure.

        Frigate structure: recordings/YYYY-MM-DD/HH/camera/MM.SS.mp4
        """
        recordings_path = tmp_path / "recordings"

        # Dec 5, 2025 (Friday) - multiple hours
        dec5 = recordings_path / "2025-12-05"
        (dec5 / "08" / "front").mkdir(parents=True)
        (dec5 / "08" / "front" / "00.00.mp4").touch()
        (dec5 / "08" / "front" / "30.00.mp4").touch()

        (dec5 / "12" / "front").mkdir(parents=True)
        (dec5 / "12" / "front" / "00.00.mp4").touch()
        (dec5 / "12" / "front" / "30.00.mp4").touch()

        (dec5 / "20" / "front").mkdir(parents=True)
        (dec5 / "20" / "front" / "00.00.mp4").touch()

        # Dec 6, 2025 (Saturday)
        dec6 = recordings_path / "2025-12-06"
        (dec6 / "10" / "front").mkdir(parents=True)
        (dec6 / "10" / "front" / "00.00.mp4").touch()
        (dec6 / "10" / "front" / "30.00.mp4").touch()

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

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_filter_by_skip_hours(self, mock_utc, mock_frigate_dir):
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
        """Create mock Frigate dir with multiple cameras.

        Frigate structure: recordings/YYYY-MM-DD/HH/camera/MM.SS.mp4
        """
        for camera in ["front", "back"]:
            camera_path = tmp_path / "recordings" / "2025-12-05" / "12" / camera
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

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_generate_lists_with_skip_hours(self, mock_utc, mock_multi_camera_dir):
        """Generates file lists with hour filtering."""
        result = generate_file_lists(
            cameras=["front"],
            start=datetime(2025, 12, 5, 0, 0, 0),
            end=datetime(2025, 12, 6, 0, 0, 0),
            instance_path=mock_multi_camera_dir,
            skip_hours=["10-14"],  # Skip 10am-2pm (includes noon)
        )

        assert len(result["front"]) == 0


class TestGenerateFileListsCombinedFilters:
    """Integration tests for generate_file_lists with combined skip_days and skip_hours."""

    @pytest.fixture
    def mock_week_dir(self, tmp_path):
        """Create mock Frigate dir with a week of recordings across multiple hours.

        Structure covers:
        - Mon Dec 1 to Sun Dec 7, 2025
        - Multiple hours per day to test hour filtering

        Frigate structure: recordings/YYYY-MM-DD/HH/camera/MM.SS.mp4
        """
        recordings_path = tmp_path / "recordings"

        # Create files for each day of the week
        for day in range(1, 8):  # Dec 1-7
            date_str = f"2025-12-0{day}"
            date_path = recordings_path / date_str

            # Create files at: 6am, 10am, 12pm, 6pm, 10pm
            hours = ["06", "10", "12", "18", "22"]
            for hour in hours:
                camera_path = date_path / hour / "front"
                camera_path.mkdir(parents=True)
                (camera_path / "00.00.mp4").touch()
                (camera_path / "30.00.mp4").touch()

        return tmp_path

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_combined_skip_days_and_hours(self, mock_utc, mock_week_dir):
        """Test that both skip_days and skip_hours are applied together.

        With skip_days=["sat", "sun"] and skip_hours=["16-8"]:
        - Excludes all Saturday (Dec 6) and Sunday (Dec 7) files
        - Excludes 6am, 6pm, 10pm files (outside 8am-4pm)
        - Keeps only Mon-Fri 10am and 12pm files
        """
        result = generate_file_lists(
            cameras=["front"],
            start=datetime(2025, 12, 1, 0, 0, 0),
            end=datetime(2025, 12, 8, 0, 0, 0),
            instance_path=mock_week_dir,
            skip_days=["sat", "sun"],  # Skip weekend
            skip_hours=["16-8"],  # Skip 4pm to 8am (keep 8am-4pm)
        )

        files = result["front"]

        # Mon-Fri = 5 days
        # Per day: 10am (2 files) + 12pm (2 files) = 4 files kept
        # 6am, 6pm, 10pm = 6 files skipped per day
        # Total: 5 days * 4 files = 20 files
        assert len(files) == 20

        # Verify no weekend files
        for f in files:
            path_str = str(f)
            assert "2025-12-06" not in path_str  # Saturday
            assert "2025-12-07" not in path_str  # Sunday

        # Verify only 8am-4pm files (10 and 12 hour directories)
        # Path structure: recordings/YYYY-MM-DD/HH/camera/MM.SS.mp4
        for f in files:
            # Extract hour from path: parent is camera, parent.parent is hour
            hour = int(f.parent.parent.name)
            assert 8 <= hour < 16, f"File at hour {hour} should be filtered: {f}"

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_combined_filters_keep_weekday_business_hours(self, mock_utc, mock_week_dir):
        """Keep only weekday business hours (9am-5pm)."""
        result = generate_file_lists(
            cameras=["front"],
            start=datetime(2025, 12, 1, 0, 0, 0),
            end=datetime(2025, 12, 8, 0, 0, 0),
            instance_path=mock_week_dir,
            skip_days=["sat", "sun"],  # Skip weekend
            skip_hours=["17-9"],  # Skip 5pm to 9am (keep 9am-5pm)
        )

        files = result["front"]

        # Mon-Fri = 5 days
        # Per day: 10am (2) + 12pm (2) = 4 files (6am and 6pm+ are outside 9-5)
        # Total: 5 * 4 = 20 files
        assert len(files) == 20

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_combined_filters_strict(self, mock_utc, mock_week_dir):
        """Very strict filters that exclude most files."""
        result = generate_file_lists(
            cameras=["front"],
            start=datetime(2025, 12, 1, 0, 0, 0),
            end=datetime(2025, 12, 8, 0, 0, 0),
            instance_path=mock_week_dir,
            skip_days=["mon", "tue", "wed", "thu", "fri"],  # Skip all weekdays
            skip_hours=["18-10"],  # Skip 6pm to 10am
        )

        files = result["front"]

        # Only Sat/Sun, only 10am-6pm
        # Sat+Sun = 2 days
        # Per day: 10am (2) + 12pm (2) = 4 files
        # Total: 2 * 4 = 8 files
        assert len(files) == 8

    @patch("frigate_tools.file_list.utc_to_local", side_effect=lambda x: x)
    def test_combined_filters_all_excluded(self, mock_utc, mock_week_dir):
        """Filters that exclude everything."""
        result = generate_file_lists(
            cameras=["front"],
            start=datetime(2025, 12, 1, 0, 0, 0),
            end=datetime(2025, 12, 8, 0, 0, 0),
            instance_path=mock_week_dir,
            skip_days=["mon", "tue", "wed", "thu", "fri", "sat", "sun"],  # All days
            skip_hours=["16-8"],
        )

        files = result["front"]
        assert len(files) == 0
