"""File list generation with calendar filtering for Frigate recordings.

Frigate stores recordings in the structure:
    {instance}/recordings/{camera}/{YYYY-MM-DD}/{HH}/{MM}.{SS}.mp4

This module finds and filters recording files by time range and calendar rules.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
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
        """Check if hour falls within this skip range."""
        if self.start <= self.end:
            # Normal range like 9-17
            return self.start <= hour < self.end
        else:
            # Wrapping range like 16-8 (skip 4pm to 8am)
            return hour >= self.start or hour < self.end


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
                except ValueError:
                    pass
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


def should_skip_timestamp(
    ts: datetime,
    skip_days: set[int],
    skip_hours: list[HourRange],
) -> bool:
    """Check if a timestamp should be skipped based on calendar rules."""
    # Check day of week
    if ts.weekday() in skip_days:
        return True

    # Check hour ranges
    for hour_range in skip_hours:
        if hour_range.contains(ts.hour):
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

    recordings_path = instance_path / "recordings" / camera
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

                # Find .mp4 files in this hour
                for file_path in sorted(hour_path.glob("*.mp4")):
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
