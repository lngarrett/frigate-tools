"""File list generation with calendar filtering for Frigate recordings.

Frigate stores recordings in the structure:
    {instance}/recordings/{camera}/{YYYY-MM-DD}/{HH}/{MM}.{SS}.mp4

This module finds and filters recording files by time range and calendar rules.
"""

import re
import time as time_module
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from frigate_tools.observability import get_logger, traced_operation

# Frigate filename pattern: MM.SS.mp4
FILENAME_PATTERN = re.compile(r"^(\d{2})\.(\d{2})\.mp4$")

# Day name mapping for skip_days
DAY_NAMES = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tuesday": 1,
    "wed": 2,
    "wednesday": 2,
    "thu": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


@dataclass
class HourRange:
    """Represents a range of hours to skip (can wrap around midnight)."""

    start: int  # 0-23
    end: int  # 0-23

    def contains(self, hour: int) -> bool:
        """Check if hour falls within this skip range (inclusive on both ends)."""
        if self.start <= self.end:
            # Normal range like 9-17 means skip hours 9 through 17 inclusive
            return self.start <= hour <= self.end
        else:
            # Wrapping range like 22-6 (skip 10pm to 6am inclusive)
            return hour >= self.start or hour <= self.end


def parse_skip_days(skip_days: list[str]) -> set[int]:
    """Parse day names into weekday numbers (0=Monday, 6=Sunday)."""
    result = set()
    for day in skip_days:
        day_lower = day.lower().strip()
        if day_lower in DAY_NAMES:
            result.add(DAY_NAMES[day_lower])
    return result


def parse_skip_hours(skip_hours: list[str]) -> list[HourRange]:
    """Parse hour range strings like '16-8' into HourRange objects."""
    logger = get_logger()
    result = []
    for range_str in skip_hours:
        if "-" in range_str:
            parts = range_str.split("-")
            if len(parts) == 2:
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                    if 0 <= start <= 23 and 0 <= end <= 23:
                        result.append(HourRange(start, end))
                    else:
                        logger.debug(
                            "Invalid skip_hours range - hours must be 0-23",
                            range=range_str,
                            start=start,
                            end=end,
                        )
                except ValueError:
                    logger.debug(
                        "Invalid skip_hours format - could not parse integers",
                        range=range_str,
                    )
        else:
            logger.debug(
                "Invalid skip_hours format - expected 'start-end'",
                range=range_str,
            )
    return result


def parse_file_timestamp(
    date_dir: str,
    hour_dir: str,
    filename: str,
) -> datetime | None:
    """Extract timestamp from Frigate directory/file structure.

    Args:
        date_dir: Directory name like "2025-12-01"
        hour_dir: Directory name like "14"
        filename: File name like "30.00.mp4"

    Returns:
        datetime or None if parsing fails
    """
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return None

    try:
        minute = int(match.group(1))
        second = int(match.group(2))
        year, month, day = map(int, date_dir.split("-"))
        hour = int(hour_dir)

        return datetime(year, month, day, hour, minute, second)
    except (ValueError, AttributeError):
        return None


def utc_to_local(dt: datetime) -> datetime:
    """Convert naive UTC datetime to naive local datetime.

    Used for skip_hours filtering: file timestamps are UTC but user-specified
    skip hours are in local time.

    Args:
        dt: Naive datetime in UTC

    Returns:
        Naive datetime in local time
    """
    # Make the datetime UTC-aware
    utc_aware = dt.replace(tzinfo=timezone.utc)

    # Get local timezone offset for this specific time (accounting for DST)
    local_timestamp = utc_aware.timestamp()
    local_time = time_module.localtime(local_timestamp)

    if local_time.tm_isdst > 0:
        utc_offset = -time_module.altzone
    else:
        utc_offset = -time_module.timezone

    local_tz = timezone(timedelta(seconds=utc_offset))
    local_aware = utc_aware.astimezone(local_tz)
    return local_aware.replace(tzinfo=None)


def should_skip_timestamp(
    ts: datetime,
    skip_days: set[int],
    skip_hours: list[HourRange],
) -> bool:
    """Check if a timestamp should be skipped based on calendar rules.

    Args:
        ts: UTC timestamp from file path
        skip_days: Days of week to skip (0=Monday, 6=Sunday) - in local time
        skip_hours: Hour ranges to skip - in local time
    """
    # Convert UTC timestamp to local time for calendar filtering
    # User specifies skip rules in local time (e.g., "skip 4pm-8am" means local time)
    local_ts = utc_to_local(ts)

    # Check day of week (in local time)
    if local_ts.weekday() in skip_days:
        return True

    # Check hour ranges (in local time)
    for hour_range in skip_hours:
        if hour_range.contains(local_ts.hour):
            return True

    return False


def find_recording_files(
    instance_path: Path,
    camera: str,
    start: datetime,
    end: datetime,
    skip_days: set[int] | None = None,
    skip_hours: list[HourRange] | None = None,
) -> list[Path]:
    """Find recording files for a camera within a time range.

    Frigate stores recordings as: recordings/YYYY-MM-DD/HH/camera/MM.SS.mp4

    Args:
        instance_path: Path to Frigate instance (contains recordings/ subdirectory)
        camera: Camera name
        start: Start of time range (inclusive)
        end: End of time range (exclusive)
        skip_days: Set of weekday numbers to skip (0=Monday, 6=Sunday)
        skip_hours: List of hour ranges to skip

    Returns:
        Sorted list of file paths
    """
    skip_days = skip_days or set()
    skip_hours = skip_hours or []

    recordings_path = instance_path / "recordings"
    if not recordings_path.exists():
        return []

    files = []

    # Iterate through date directories
    current_date = start.date()
    end_date = end.date()

    while current_date <= end_date:
        date_dir = current_date.strftime("%Y-%m-%d")
        date_path = recordings_path / date_dir

        if date_path.exists():
            # Check each hour directory
            for hour_path in sorted(date_path.iterdir()):
                if not hour_path.is_dir():
                    continue

                try:
                    hour = int(hour_path.name)
                except ValueError:
                    continue

                # Camera is inside the hour directory
                camera_path = hour_path / camera
                if not camera_path.exists():
                    continue

                # Find .mp4 files in this camera directory
                for file_path in sorted(camera_path.glob("*.mp4")):
                    ts = parse_file_timestamp(date_dir, hour_path.name, file_path.name)
                    if ts is None:
                        continue

                    # Check time range
                    if ts < start or ts >= end:
                        continue

                    # Check calendar filters
                    if should_skip_timestamp(ts, skip_days, skip_hours):
                        continue

                    files.append(file_path)

        current_date += timedelta(days=1)

    return files


def generate_file_lists(
    cameras: list[str],
    start: datetime,
    end: datetime,
    instance_path: Path,
    skip_days: list[str] | None = None,
    skip_hours: list[str] | None = None,
) -> dict[str, list[Path]]:
    """Generate file lists for multiple cameras with calendar filtering.

    Args:
        cameras: List of camera names
        start: Start of time range
        end: End of time range
        instance_path: Path to Frigate instance
        skip_days: Day names to skip (e.g., ["sat", "sun"])
        skip_hours: Hour ranges to skip (e.g., ["16-8"])

    Returns:
        Dict mapping camera name to sorted list of file paths
    """
    logger = get_logger()

    parsed_skip_days = parse_skip_days(skip_days or [])
    parsed_skip_hours = parse_skip_hours(skip_hours or [])

    with traced_operation(
        "generate_file_lists",
        {
            "cameras": ",".join(cameras),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "skip_days": ",".join(skip_days or []),
            "skip_hours": ",".join(skip_hours or []),
        },
    ):
        result = {}

        for camera in cameras:
            files = find_recording_files(
                instance_path=instance_path,
                camera=camera,
                start=start,
                end=end,
                skip_days=parsed_skip_days,
                skip_hours=parsed_skip_hours,
            )
            result[camera] = files
            logger.info(
                "Found files for camera",
                camera=camera,
                file_count=len(files),
            )

        return result
