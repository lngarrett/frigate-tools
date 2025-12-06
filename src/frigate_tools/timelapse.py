"""Single camera timelapse encoding.

Two-step approach:
1. Concat input files with -c copy to temp file (fast, no re-encoding)
2. Encode with setpts filter to create timelapse

Uses setpts filter with frame dropping for efficient timelapse creation.
The setpts approach is faster than select filter because ffmpeg only
decodes and encodes frames that will be in the output.
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


def get_video_info(file_path: Path) -> tuple[float, float]:
    """Get duration and frame rate of a video file using ffprobe.

    Returns:
        Tuple of (duration_seconds, fps)
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration:stream=r_frame_rate",
        "-of", "json",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0, 0.0

    import json
    try:
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))

        # Parse frame rate (e.g., "30/1" or "30000/1001")
        fps_str = data.get("streams", [{}])[0].get("r_frame_rate", "0/1")
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den else 0.0

        return duration, fps
    except (ValueError, KeyError, IndexError, json.JSONDecodeError):
        return 0.0, 0.0


def get_video_duration(file_path: Path) -> float:
    """Get duration of a video file in seconds using ffprobe."""
    duration, _ = get_video_info(file_path)
    return duration


@dataclass
class ConcatProgress:
    """Concatenation progress information."""

    files_total: int
    files_processed: int
    bytes_written: int
    elapsed_seconds: float
    percent: float | None = None


def parse_concat_progress(line: str, total_files: int, start_time: float) -> ConcatProgress | None:
    """Parse ffmpeg progress output during concatenation.

    FFmpeg -progress outputs key=value pairs. We track out_time_ms for progress.
    """
    import time

    # Look for out_time_ms which shows output duration in microseconds
    if line.startswith("out_time_ms="):
        try:
            out_time_us = int(line.split("=")[1])
            elapsed = time.time() - start_time
            # We can't know exact file count from time, but we report what we have
            return ConcatProgress(
                files_total=total_files,
                files_processed=0,  # Unknown during concat
                bytes_written=0,
                elapsed_seconds=elapsed,
                percent=None,  # Can't calculate without knowing total duration
            )
        except (ValueError, IndexError):
            pass

    # Look for total_size which shows bytes written
    if line.startswith("total_size="):
        try:
            import time
            bytes_written = int(line.split("=")[1])
            elapsed = time.time() - start_time
            return ConcatProgress(
                files_total=total_files,
                files_processed=0,
                bytes_written=bytes_written,
                elapsed_seconds=elapsed,
                percent=None,
            )
        except (ValueError, IndexError):
            pass

    return None


def concat_files(
    input_files: list[Path],
    output_path: Path,
    progress_callback: Callable[[ConcatProgress], None] | None = None,
) -> bool:
    """Concatenate video files using ffmpeg concat demuxer.

    Uses -c copy for fast concatenation without re-encoding.

    Args:
        input_files: List of video files to concatenate
        output_path: Output file path
        progress_callback: Optional callback for progress updates (receives ConcatProgress)

    Returns:
        True if successful, False otherwise
    """
    import time

    logger = get_logger()

    if not input_files:
        logger.error("No input files provided")
        return False

    with traced_operation("concat_files", {"file_count": len(input_files)}):
        # Calculate total input size for progress estimation
        total_size = sum(f.stat().st_size for f in input_files if f.exists())

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
            ]

            # Add progress output if callback provided
            if progress_callback:
                cmd.extend(["-progress", "pipe:1"])

            cmd.append(str(output_path))

            logger.info("Starting concat", file_count=len(input_files), total_size=total_size)

            start_time = time.time()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if progress_callback and process.stdout:
                last_bytes = 0
                for line in process.stdout:
                    line = line.strip()
                    # Parse total_size for progress
                    if line.startswith("total_size="):
                        try:
                            bytes_written = int(line.split("=")[1])
                            elapsed = time.time() - start_time
                            # Estimate percent based on bytes written vs expected
                            percent = min(99.0, (bytes_written / total_size) * 100) if total_size > 0 else None
                            progress_callback(ConcatProgress(
                                files_total=len(input_files),
                                files_processed=0,
                                bytes_written=bytes_written,
                                elapsed_seconds=elapsed,
                                percent=percent,
                            ))
                            last_bytes = bytes_written
                        except (ValueError, IndexError):
                            pass

                stderr = process.stderr.read() if process.stderr else ""
                process.wait()
            else:
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
    output_fps: float = 30.0,
    preset: str = "fast",
    progress_callback: Callable[[ProgressInfo], None] | None = None,
) -> bool:
    """Encode video with timelapse effect using setpts filter.

    Uses setpts filter to speed up video by adjusting presentation timestamps.
    FFmpeg automatically drops frames to match output framerate, which is
    much faster than the select filter approach (which decodes all frames).

    Args:
        input_path: Input video file
        output_path: Output file path
        target_duration: Desired output duration in seconds
        output_fps: Output frame rate (default 30)
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
        # Get source duration
        source_duration, source_fps = get_video_info(input_path)
        if source_duration <= 0:
            logger.error("Could not determine source duration")
            return False

        if source_fps <= 0:
            source_fps = 30.0  # Assume 30fps if detection fails

        # Calculate speed factor
        speed = source_duration / target_duration

        logger.info(
            "Encoding timelapse",
            source_duration=source_duration,
            source_fps=source_fps,
            target_duration=target_duration,
            output_fps=output_fps,
            speed=speed,
        )

        # Use setpts to speed up video - PTS/N makes video N times faster
        # FFmpeg will drop frames to match output framerate, which is efficient
        filter_complex = f"setpts=PTS/{speed}"

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_path),
            "-vf", filter_complex,
            "-r", str(output_fps),
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
        # We read stdout completely before waiting for the process.
        # This is safe because we consume the entire stream, then wait().
        stderr_output = ""
        if process.stdout:
            for line in process.stdout:
                if progress_callback:
                    progress = parse_ffmpeg_progress(line.strip(), target_duration)
                    if progress:
                        progress_callback(progress)

        # Read any remaining stderr and wait for process
        if process.stderr:
            stderr_output = process.stderr.read()
        process.wait()

        if process.returncode != 0:
            logger.error("Encoding failed", stderr=stderr_output)
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
