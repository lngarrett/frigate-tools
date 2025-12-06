"""Single camera timelapse encoding.

Two-step approach:
1. Concat input files with -c copy to temp file (fast, no re-encoding)
2. Encode with setpts filter to speed up video

Speed multiplier = source_duration / target_duration
"""

import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from frigate_tools.observability import get_logger, traced_operation


@dataclass
class ProgressInfo:
    """FFmpeg encoding progress information."""

    frame: int
    fps: float
    time_seconds: float
    speed: float
    percent: float | None = None  # Set when total duration is known


def parse_ffmpeg_progress(line: str, total_duration: float | None = None) -> ProgressInfo | None:
    """Parse FFmpeg progress line.

    FFmpeg outputs progress like:
    frame=  123 fps= 30 ... time=00:00:04.10 ... speed=1.23x
    """
    frame_match = re.search(r"frame=\s*(\d+)", line)
    fps_match = re.search(r"fps=\s*([\d.]+)", line)
    time_match = re.search(r"time=(\d+):(\d+):([\d.]+)", line)
    speed_match = re.search(r"speed=\s*([\d.]+)x", line)

    if not (frame_match and time_match):
        return None

    frame = int(frame_match.group(1))
    fps = float(fps_match.group(1)) if fps_match else 0.0

    hours, minutes, seconds = time_match.groups()
    time_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    speed = float(speed_match.group(1)) if speed_match else 0.0

    percent = None
    if total_duration and total_duration > 0:
        percent = min(100.0, (time_seconds / total_duration) * 100)

    return ProgressInfo(
        frame=frame,
        fps=fps,
        time_seconds=time_seconds,
        speed=speed,
        percent=percent,
    )


def get_video_duration(file_path: Path) -> float:
    """Get duration of a video file in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def concat_files(
    input_files: list[Path],
    output_path: Path,
    progress_callback: Callable[[ProgressInfo], None] | None = None,
) -> bool:
    """Concatenate video files using ffmpeg concat demuxer.

    Uses -c copy for fast concatenation without re-encoding.

    Args:
        input_files: List of video files to concatenate
        output_path: Output file path
        progress_callback: Optional callback for progress updates

    Returns:
        True if successful, False otherwise
    """
    logger = get_logger()

    if not input_files:
        logger.error("No input files provided")
        return False

    with traced_operation("concat_files", {"file_count": len(input_files)}):
        # Create concat file list
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            concat_file = Path(f.name)
            for file_path in input_files:
                # FFmpeg concat requires escaped paths
                escaped = str(file_path).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        try:
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(output_path),
            ]

            logger.info("Starting concat", file_count=len(input_files))

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            _, stderr = process.communicate()

            if process.returncode != 0:
                logger.error("Concat failed", stderr=stderr)
                return False

            logger.info("Concat complete", output=str(output_path))
            return True

        finally:
            concat_file.unlink(missing_ok=True)


def encode_timelapse(
    input_path: Path,
    output_path: Path,
    target_duration: float,
    preset: str = "fast",
    progress_callback: Callable[[ProgressInfo], None] | None = None,
) -> bool:
    """Encode video with timelapse effect using setpts filter.

    Args:
        input_path: Input video file
        output_path: Output file path
        target_duration: Desired output duration in seconds
        preset: FFmpeg preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        progress_callback: Optional callback(ProgressInfo) for progress updates

    Returns:
        True if successful, False otherwise
    """
    logger = get_logger()

    with traced_operation(
        "encode_timelapse",
        {"target_duration": target_duration, "preset": preset},
    ):
        # Get source duration to calculate speed multiplier
        source_duration = get_video_duration(input_path)
        if source_duration <= 0:
            logger.error("Could not determine source duration")
            return False

        # Calculate PTS multiplier (inverse of speed)
        # speed = source_duration / target_duration
        # pts_multiplier = 1 / speed = target_duration / source_duration
        pts_multiplier = target_duration / source_duration

        logger.info(
            "Encoding timelapse",
            source_duration=source_duration,
            target_duration=target_duration,
            speed=source_duration / target_duration,
            pts_multiplier=pts_multiplier,
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-filter:v", f"setpts={pts_multiplier}*PTS",
            "-an",  # Remove audio (doesn't make sense for timelapse)
            "-preset", preset,
            "-progress", "pipe:1",
            str(output_path),
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Parse progress from stdout
        if progress_callback and process.stdout:
            for line in process.stdout:
                progress = parse_ffmpeg_progress(line.strip(), target_duration)
                if progress:
                    progress_callback(progress)

        _, stderr = process.communicate()

        if process.returncode != 0:
            logger.error("Encoding failed", stderr=stderr)
            return False

        logger.info("Encoding complete", output=str(output_path))
        return True


def create_timelapse(
    input_files: list[Path],
    output_path: Path,
    target_duration: float,
    preset: str = "fast",
    progress_callback: Callable[[ProgressInfo], None] | None = None,
    keep_temp: bool = False,
) -> bool:
    """Create a timelapse video from input files.

    Two-step process:
    1. Concatenate all input files (fast, no re-encoding)
    2. Encode with speed adjustment to target duration

    Args:
        input_files: List of video files
        output_path: Output file path
        target_duration: Desired output duration in seconds
        preset: FFmpeg encoding preset
        progress_callback: Optional callback for progress updates
        keep_temp: Keep temporary concatenated file (for debugging)

    Returns:
        True if successful, False otherwise
    """
    logger = get_logger()

    with traced_operation(
        "create_timelapse",
        {"file_count": len(input_files), "target_duration": target_duration},
    ):
        # Create temp file for concatenated video
        temp_dir = output_path.parent
        temp_file = temp_dir / f".{output_path.stem}_concat.mp4"

        try:
            # Step 1: Concatenate
            if not concat_files(input_files, temp_file):
                return False

            # Step 2: Encode with timelapse effect
            if not encode_timelapse(
                temp_file,
                output_path,
                target_duration,
                preset=preset,
                progress_callback=progress_callback,
            ):
                return False

            logger.info(
                "Timelapse created",
                output=str(output_path),
                file_count=len(input_files),
            )
            return True

        finally:
            if not keep_temp:
                temp_file.unlink(missing_ok=True)
