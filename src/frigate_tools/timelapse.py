"""Single camera timelapse encoding.

Two-step approach:
1. Concat input files with -c copy to temp file (fast, no re-encoding)
2. Encode with setpts filter to create timelapse

Uses setpts filter with frame dropping for efficient timelapse creation.
Supports hardware acceleration (Intel QSV, VAAPI) when available.
"""

import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from frigate_tools.observability import get_logger, traced_operation


class HWAccel(Enum):
    """Hardware acceleration types."""
    NONE = "none"
    QSV = "qsv"       # Intel Quick Sync Video
    VAAPI = "vaapi"   # Video Acceleration API (Linux)


def detect_hwaccel() -> HWAccel:
    """Detect available hardware acceleration.

    Checks for Intel QSV first (faster), then VAAPI.
    Returns HWAccel.NONE if no hardware acceleration is available.
    """
    logger = get_logger()

    # Check for render device (needed for both QSV and VAAPI)
    render_device = Path("/dev/dri/renderD128")
    if not render_device.exists():
        logger.debug("No render device found, using software encoding")
        return HWAccel.NONE

    # Check for QSV support by testing if h264_qsv encoder is available
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "h264_qsv" in result.stdout:
            # Verify QSV actually works with a quick test
            test_result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-v", "error",
                    "-init_hw_device", "qsv=qsv:hw",
                    "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                    "-c:v", "h264_qsv", "-f", "null", "-"
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if test_result.returncode == 0:
                logger.info("Using Intel QSV hardware acceleration")
                return HWAccel.QSV

        # Check for VAAPI support
        if "h264_vaapi" in result.stdout:
            test_result = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-v", "error",
                    "-vaapi_device", "/dev/dri/renderD128",
                    "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                    "-vf", "format=nv12,hwupload",
                    "-c:v", "h264_vaapi", "-f", "null", "-"
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if test_result.returncode == 0:
                logger.info("Using VAAPI hardware acceleration")
                return HWAccel.VAAPI

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    logger.debug("No hardware acceleration available, using software encoding")
    return HWAccel.NONE


# Cache hardware detection result
_hwaccel_cache: HWAccel | None = None


def get_hwaccel() -> HWAccel:
    """Get cached hardware acceleration type."""
    global _hwaccel_cache
    if _hwaccel_cache is None:
        _hwaccel_cache = detect_hwaccel()
    return _hwaccel_cache


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

    Supports two formats:
    1. Classic progress format: frame=123 fps=30 ... time=00:00:04.10 ... speed=1.23x
       All fields on one line - requires frame AND time to be present.
    2. -progress pipe:1 format: out_time=00:00:04.100000 (key=value on separate lines)
       Only needs out_time line for progress calculation.
    """
    # Try to match classic format (all on one line) or -progress format (separate lines)
    frame_match = re.search(r"frame=\s*(\d+)", line)
    fps_match = re.search(r"fps=\s*([\d.]+)", line)
    speed_match = re.search(r"speed=\s*([\d.]+)x", line)

    # Check for out_time (from -progress pipe:1) - this is on its own line
    out_time_match = re.search(r"^out_time=(\d+):(\d+):([\d.]+)", line)
    # Check for classic time= format (on same line as frame=)
    classic_time_match = re.search(r"time=(\d+):(\d+):([\d.]+)", line)

    # For -progress format: just need out_time line to report progress
    if out_time_match:
        hours, minutes, seconds = out_time_match.groups()
        time_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

        percent = None
        if total_duration and total_duration > 0:
            percent = min(100.0, (time_seconds / total_duration) * 100)

        return ProgressInfo(
            frame=0,  # Frame count not available on this line
            fps=0.0,
            time_seconds=time_seconds,
            speed=0.0,
            percent=percent,
        )

    # For classic format: need both frame and time on same line
    if frame_match and classic_time_match:
        frame = int(frame_match.group(1))
        fps = float(fps_match.group(1)) if fps_match else 0.0

        hours, minutes, seconds = classic_time_match.groups()
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

    return None


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


def concat_files(
    input_files: list[Path],
    output_path: Path,
    progress_callback: Callable[[ConcatProgress], None] | None = None,
    batch_size: int = 100,
) -> bool:
    """Concatenate video files using ffmpeg concat demuxer.

    Uses -c copy for fast concatenation without re-encoding.
    Processes files in batches to avoid issues with a very large number of files.

    Args:
        input_files: List of video files to concatenate
        output_path: Output file path
        progress_callback: Optional callback for progress updates (receives ConcatProgress)
        batch_size: Number of files to concatenate in each batch.

    Returns:
        True if successful, False otherwise
    """
    import time
    import tempfile

    logger = get_logger()

    if not input_files:
        logger.error("No input files provided")
        return False

    with traced_operation("concat_files", {"file_count": len(input_files), "batch_size": batch_size}):
        total_files = len(input_files)
        total_batches = (total_files + batch_size - 1) // batch_size
        intermediate_files = []
        temp_dir = Path(tempfile.mkdtemp(prefix="frigate_concat_"))
        
        try:
            files_processed_count = 0
            for i in range(total_batches):
                batch_start_index = i * batch_size
                batch_end_index = min((i + 1) * batch_size, total_files)
                batch = input_files[batch_start_index:batch_end_index]
                batch_output_path = temp_dir / f"intermediate_{i}.mp4"
                
                logger.info("Processing batch", batch_num=i+1, of=total_batches, file_count=len(batch))
                
                if not _concat_batch(batch, batch_output_path):
                    logger.error(f"Failed to concatenate batch {i+1}")
                    return False
                
                intermediate_files.append(batch_output_path)
                
                files_processed_count += len(batch)
                if progress_callback:
                    percent = min(100.0, (files_processed_count / total_files) * 100)
                    progress_callback(ConcatProgress(
                        files_total=total_files,
                        files_processed=files_processed_count,
                        bytes_written=0, # Not easily available for file-based progress
                        elapsed_seconds=0, # Not easily available for file-based progress
                        percent=percent,
                    ))

            # Finally, concatenate the intermediate files.
            # This step also needs to report progress if there's more than one intermediate file.
            final_concat_successful = _concat_batch(intermediate_files, output_path)
            
            if final_concat_successful and progress_callback:
                progress_callback(ConcatProgress(
                    files_total=total_files,
                    files_processed=total_files,
                    bytes_written=0,
                    elapsed_seconds=0,
                    percent=100.0,
                ))
            return final_concat_successful

        finally:
            # Clean up the temporary directory.
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def _concat_batch(
    input_files: list[Path],
    output_path: Path,
    progress_callback: Callable[[ConcatProgress], None] | None = None,
) -> bool:
    """Helper function to concatenate a single batch of files."""
    import time

    logger = get_logger()
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file_list_path = Path(f.name)
        for file_path in input_files:
            # Resolve to absolute path to ensure ffmpeg can find the files
            abs_path = file_path.resolve()
            escaped = str(abs_path).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file_list_path),
            "-c", "copy",
        ]
        
        # Add progress output if callback provided (only if re-encoding, copy doesn't give good progress)
        # For concat -c copy, ffmpeg doesn't output reliable progress updates via pipe:1
        # if progress_callback:
        #     cmd.extend(["-progress", "pipe:1"])
            
        cmd.append(str(output_path))
        
        # Calculate total input size for logging and potential future progress estimation
        total_size = sum(f.stat().st_size for f in input_files if f.exists())
        logger.info("Starting batch concat", file_count=len(input_files), total_size=total_size)
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # For -c copy, ffmpeg typically doesn't send progress to stdout via -progress pipe:1
        # Just consume stdout/stderr after process finishes
        stdout, stderr = process.communicate()


        if process.returncode != 0:
            logger.error("Batch concat failed", stderr=stderr)
            return False
            
        logger.info("Batch concat complete", output=str(output_path))
        return True

    finally:
        concat_file_list_path.unlink(missing_ok=True)



def encode_timelapse(
    input_path: Path,
    output_path: Path,
    target_duration: float,
    output_fps: float = 30.0,
    preset: str = "fast",
    progress_callback: Callable[[ProgressInfo], None] | None = None,
    hwaccel: HWAccel | None = None,
) -> bool:
    """Encode video with timelapse effect using setpts filter.

    Uses setpts filter to speed up video by adjusting presentation timestamps.
    Supports hardware acceleration (Intel QSV, VAAPI) for faster encoding.

    Args:
        input_path: Input video file
        output_path: Output file path
        target_duration: Desired output duration in seconds
        output_fps: Output frame rate (default 30)
        preset: FFmpeg preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        progress_callback: Optional callback(ProgressInfo) for progress updates
        hwaccel: Hardware acceleration to use (auto-detected if None)

    Returns:
        True if successful, False otherwise
    """
    logger = get_logger()

    # Auto-detect hardware acceleration if not specified
    if hwaccel is None:
        hwaccel = get_hwaccel()

    with traced_operation(
        "encode_timelapse",
        {"target_duration": target_duration, "preset": preset, "hwaccel": hwaccel.value},
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
            hwaccel=hwaccel.value,
        )

        # Build command based on hardware acceleration type
        cmd = ["ffmpeg", "-y"]

        if hwaccel == HWAccel.QSV:
            # Intel QSV: Use hardware decoding and encoding
            cmd.extend([
                "-hwaccel", "qsv",
                "-hwaccel_output_format", "qsv",
                "-i", str(input_path),
                # QSV filter chain: scale for setpts equivalent, then encode
                "-vf", f"setpts=PTS/{speed}",
                "-r", str(output_fps),
                "-c:v", "h264_qsv",
                "-preset", _qsv_preset(preset),
                "-global_quality", "23",  # Quality level (lower = better, 18-28 typical)
            ])
        elif hwaccel == HWAccel.VAAPI:
            # VAAPI: Use hardware encoding with -vaapi_device
            cmd.extend([
                "-vaapi_device", "/dev/dri/renderD128",
                "-i", str(input_path),
                "-vf", f"setpts=PTS/{speed},format=nv12,hwupload",
                "-r", str(output_fps),
                "-c:v", "h264_vaapi",
                "-qp", "23",  # Quality parameter
            ])
        else:
            # Software encoding (default)
            cmd.extend([
                "-i", str(input_path),
                "-vf", f"setpts=PTS/{speed}",
                "-r", str(output_fps),
                "-preset", preset,
            ])

        # Common options
        cmd.extend([
            "-an",  # Remove audio (doesn't make sense for timelapse)
            "-progress", "pipe:1",
            str(output_path),
        ])

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
            logger.error("Encoding failed", stderr=stderr_output, hwaccel=hwaccel.value)
            # Fall back to software encoding if hardware failed
            if hwaccel != HWAccel.NONE:
                logger.info("Falling back to software encoding")
                return encode_timelapse(
                    input_path,
                    output_path,
                    target_duration,
                    output_fps,
                    preset,
                    progress_callback,
                    hwaccel=HWAccel.NONE,
                )
            return False

        logger.info("Encoding complete", output=str(output_path), hwaccel=hwaccel.value)
        return True


def _qsv_preset(preset: str) -> str:
    """Map software preset to QSV preset.

    QSV presets: veryfast, faster, fast, medium, slow, slower, veryslow
    """
    # QSV has fewer presets, map appropriately
    mapping = {
        "ultrafast": "veryfast",
        "superfast": "veryfast",
        "veryfast": "veryfast",
        "faster": "faster",
        "fast": "fast",
        "medium": "medium",
        "slow": "slow",
        "slower": "slower",
        "veryslow": "veryslow",
    }
    return mapping.get(preset, "fast")


def create_timelapse(
    input_files: list[Path],
    output_path: Path,
    target_duration: float,
    preset: str = "fast",
    progress_callback: Callable[[ProgressInfo], None] | None = None,
    keep_temp: bool = False,
    hwaccel: HWAccel | None = None,
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
        hwaccel: Hardware acceleration to use (auto-detected if None)

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
                hwaccel=hwaccel,
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
