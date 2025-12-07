"""Single camera timelapse encoding.

Keyframe extraction approach:
1. Calculate which timestamps/files contain the frames we need
2. Extract keyframes directly from source files (no concat)
3. Encode extracted frames to output

Uses -skip_frame nokey for efficient keyframe-only decoding.
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
            # Intel QSV: Full hardware pipeline with frame selection
            # select filter picks frames on GPU, avoiding CPU transfer for setpts
            # For speed=N, select every Nth frame: 'not(mod(n,N))'
            select_expr = f"not(mod(n,{int(speed)}))" if speed >= 2 else None
            if select_expr:
                # High speedup: use select filter (processes fewer frames)
                cmd.extend([
                    "-hwaccel", "qsv",
                    "-hwaccel_output_format", "qsv",
                    "-i", str(input_path),
                    "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
                    "-r", str(output_fps),
                    "-c:v", "h264_qsv",
                    "-preset", _qsv_preset(preset),
                    "-global_quality", "23",
                ])
            else:
                # Low speedup: use setpts (smoother)
                cmd.extend([
                    "-hwaccel", "qsv",
                    "-hwaccel_output_format", "qsv",
                    "-i", str(input_path),
                    "-vf", f"setpts=PTS/{speed}",
                    "-r", str(output_fps),
                    "-c:v", "h264_qsv",
                    "-preset", _qsv_preset(preset),
                    "-global_quality", "23",
                ])
        elif hwaccel == HWAccel.VAAPI:
            # VAAPI: Full hardware pipeline with frame selection
            # select filter picks frames on GPU, scale_vaapi processes on GPU
            select_expr = f"not(mod(n,{int(speed)}))" if speed >= 2 else None
            if select_expr:
                # High speedup: use select filter (processes fewer frames)
                cmd.extend([
                    "-hwaccel", "vaapi",
                    "-hwaccel_output_format", "vaapi",
                    "-hwaccel_device", "/dev/dri/renderD128",
                    "-i", str(input_path),
                    "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB,scale_vaapi=format=nv12",
                    "-r", str(output_fps),
                    "-c:v", "h264_vaapi",
                    "-qp", "23",
                ])
            else:
                # Low speedup: use setpts (smoother)
                cmd.extend([
                    "-hwaccel", "vaapi",
                    "-hwaccel_output_format", "vaapi",
                    "-hwaccel_device", "/dev/dri/renderD128",
                    "-i", str(input_path),
                    "-vf", f"setpts=PTS/{speed},scale_vaapi=format=nv12",
                    "-r", str(output_fps),
                    "-c:v", "h264_vaapi",
                    "-qp", "23",
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
    output_fps: float = 30.0,
    preset: str = "fast",
    progress_callback: Callable[[ProgressInfo], None] | None = None,
    keep_temp: bool = False,
    hwaccel: HWAccel | None = None,
) -> bool:
    """Create a timelapse video from input files.

    Uses adaptive strategy based on speedup ratio:
    - Low speedup (<30x): Concat + encode with select filter (more frames needed)
    - High speedup (>=30x): Two-pass BSF approach (no decode/encode, much faster)

    For high speedups, uses bitstream filters to operate at the packet level
    without decoding. This provides 5-10x performance improvement.

    Args:
        input_files: List of video files (typically Frigate 10-second segments)
        output_path: Output file path
        target_duration: Desired output duration in seconds
        output_fps: Output frame rate (default 30)
        preset: FFmpeg encoding preset (for low speedups)
        progress_callback: Optional callback for progress updates
        keep_temp: Keep temporary files (for debugging)
        hwaccel: Hardware acceleration to use (for low speedups)

    Returns:
        True if successful, False otherwise
    """
    logger = get_logger()

    if hwaccel is None:
        hwaccel = get_hwaccel()

    with traced_operation(
        "create_timelapse",
        {"file_count": len(input_files), "target_duration": target_duration},
    ):
        # Get actual source duration by sampling files
        sample_count = min(5, len(input_files))
        sample_durations = [get_video_duration(f) for f in input_files[:sample_count]]
        avg_file_duration = sum(sample_durations) / len(sample_durations) if sample_durations else 5.0
        source_duration = len(input_files) * avg_file_duration

        speedup = source_duration / target_duration

        # BSF approach works when speedup >= 30x
        # (assuming ~1 keyframe/second input and 30fps output)
        # Below that threshold, we need more frames than keyframes available
        use_bsf = speedup >= 30.0

        if use_bsf:
            return _create_timelapse_bsf(
                input_files=input_files,
                output_path=output_path,
                target_duration=target_duration,
                source_duration=source_duration,
                output_fps=output_fps,
                progress_callback=progress_callback,
                keep_temp=keep_temp,
            )
        else:
            return _create_timelapse_concat(
                input_files=input_files,
                output_path=output_path,
                target_duration=target_duration,
                preset=preset,
                progress_callback=progress_callback,
                keep_temp=keep_temp,
                hwaccel=hwaccel,
            )


def _create_timelapse_concat(
    input_files: list[Path],
    output_path: Path,
    target_duration: float,
    preset: str = "fast",
    progress_callback: Callable[[ProgressInfo], None] | None = None,
    keep_temp: bool = False,
    hwaccel: HWAccel | None = None,
) -> bool:
    """Create timelapse using concat + encode approach.

    For lower speedups where we need more frames than keyframes available.
    Two-step process:
    1. Concatenate all input files (fast, -c copy)
    2. Encode with select filter to target duration
    """
    logger = get_logger()

    logger.info(
        "Creating timelapse with concat approach (low speedup)",
        file_count=len(input_files),
        target_duration=f"{target_duration:.0f}s",
        hwaccel=hwaccel.value if hwaccel else "none",
    )

    # Create temp file for concatenated video
    temp_file = output_path.parent / f".{output_path.stem}_concat.mp4"

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


def check_ffmpeg_bsf_support() -> bool:
    """Check if FFmpeg supports the noise bitstream filter.

    The noise BSF with drop parameter is required for the two-pass BSF approach.
    Available in FFmpeg 4.4+.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-bsfs"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "noise" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def estimate_keyframes(file_count: int, avg_duration: float = 10.0) -> int:
    """Estimate keyframe count from file count.

    Frigate recordings have ~1 keyframe per second (GOP=30 at 30fps).
    Each 10-second segment has ~10 keyframes.

    Args:
        file_count: Number of input files
        avg_duration: Average file duration in seconds (Frigate default: 10s)

    Returns:
        Estimated number of keyframes
    """
    return int(file_count * avg_duration)


def _bsf_pass1_concat(
    input_files: list[Path],
    output_path: Path,
    progress_callback: Callable[[ProgressInfo], None] | None = None,
) -> bool:
    """Pass 1: Fast concatenation with stream copy (no decode/encode).

    Args:
        input_files: List of video files to concatenate
        output_path: Output file path
        progress_callback: Optional callback for progress updates

    Returns:
        True if successful
    """
    import time

    logger = get_logger()

    # Estimate expected output size for progress reporting
    # Sample first few files to get average size
    sample_sizes = [f.stat().st_size for f in input_files[:min(5, len(input_files))]]
    avg_file_size = sum(sample_sizes) / len(sample_sizes) if sample_sizes else 10 * 1024 * 1024
    expected_size = len(input_files) * avg_file_size

    # Create concat list file
    concat_list = output_path.parent / f".{output_path.stem}_list.txt"
    with open(concat_list, "w") as f:
        for file_path in input_files:
            escaped = str(file_path.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            "-an",
            str(output_path),
        ]

        logger.debug("BSF Pass 1 command", cmd=" ".join(cmd))

        # Run concat with progress monitoring via output file size
        stderr_file = output_path.parent / f".{output_path.stem}_p1_stderr.log"
        try:
            with open(stderr_file, "w") as stderr_fh:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=stderr_fh,
                    text=True,
                )

                # Monitor progress by checking output file size
                # Progress range: 5% to 85% (leave room for Pass 2)
                last_report_time = time.time()
                while process.poll() is None:
                    time.sleep(2)  # Check every 2 seconds

                    if progress_callback and output_path.exists():
                        current_size = output_path.stat().st_size
                        # Scale progress: 5% to 85%
                        size_ratio = min(1.0, current_size / expected_size)
                        percent = 5 + (size_ratio * 80)

                        # Calculate estimated files processed
                        files_processed = int(size_ratio * len(input_files))

                        progress_callback(ProgressInfo(
                            frame=files_processed,
                            fps=0,
                            time_seconds=0,
                            speed=0,
                            percent=percent,
                        ))
                        last_report_time = time.time()

            stderr_output = ""
            if process.returncode != 0:
                stderr_output = stderr_file.read_text()
        finally:
            stderr_file.unlink(missing_ok=True)

        if process.returncode != 0:
            logger.error("BSF Pass 1 (concat) failed", stderr=stderr_output)
            return False

        if not output_path.exists():
            logger.error("BSF Pass 1 (concat) failed - output file not created")
            return False

        logger.info(
            "BSF Pass 1 complete",
            output_size_mb=output_path.stat().st_size / (1024 * 1024),
        )
        return True

    finally:
        concat_list.unlink(missing_ok=True)


def _bsf_pass2_timelapse(
    input_path: Path,
    output_path: Path,
    packet_interval: int,
    output_fps: float,
) -> bool:
    """Pass 2: Apply bitstream filter to create timelapse (no decode/encode).

    Uses noise BSF to drop packets and setts BSF to retime.

    Args:
        input_path: Concatenated video file
        output_path: Output timelapse file
        packet_interval: Keep every Nth packet
        output_fps: Output frame rate

    Returns:
        True if successful
    """
    logger = get_logger()

    # Build BSF expression
    # noise=drop keeps packets where expression is 0 (false)
    # We keep every Nth packet starting at 0
    drop_expr = f"mod(n\\,{packet_interval})"
    setts_expr = f"N/{output_fps}/TB_OUT"

    cmd = [
        "ffmpeg", "-y",
        "-discard", "nokey",  # Only read keyframe packets
        "-i", str(input_path),
        "-c", "copy",
        "-an",
        "-bsf:v", f"noise=drop='{drop_expr}',setts=ts='{setts_expr}'",
        str(output_path),
    ]

    logger.debug("BSF Pass 2 command", cmd=" ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error("BSF Pass 2 (timelapse) failed", stderr=result.stderr)
        return False

    if not output_path.exists():
        logger.error("BSF Pass 2 (timelapse) failed - output file not created")
        return False

    logger.info(
        "BSF Pass 2 complete",
        output_size_mb=output_path.stat().st_size / (1024 * 1024),
    )
    return True


def _create_timelapse_bsf(
    input_files: list[Path],
    output_path: Path,
    target_duration: float,
    source_duration: float,
    output_fps: float = 30.0,
    progress_callback: Callable[[ProgressInfo], None] | None = None,
    keep_temp: bool = False,
) -> bool:
    """Create timelapse using two-pass bitstream filter approach.

    This approach operates at the packet level without decoding:
    - Pass 1: Concatenate all files with stream copy
    - Pass 2: Apply BSF to select keyframes and retime

    Much faster than decode+encode for high speedups (5-10x improvement).

    Args:
        input_files: List of video files
        output_path: Output file path
        target_duration: Desired output duration in seconds
        source_duration: Total source duration in seconds
        output_fps: Output frame rate
        progress_callback: Optional callback for progress updates
        keep_temp: Keep intermediate files for debugging

    Returns:
        True if successful
    """
    logger = get_logger()

    # Calculate BSF parameters
    estimated_keyframes = estimate_keyframes(len(input_files))
    frames_needed = int(target_duration * output_fps)
    packet_interval = max(1, estimated_keyframes // frames_needed)
    speedup = source_duration / target_duration

    logger.info(
        "Creating timelapse with BSF approach",
        file_count=len(input_files),
        source_duration=f"{source_duration:.0f}s",
        target_duration=f"{target_duration:.0f}s",
        speedup=f"{speedup:.0f}x",
        estimated_keyframes=estimated_keyframes,
        frames_needed=frames_needed,
        packet_interval=packet_interval,
    )

    # Temp file for concatenated video
    concat_output = output_path.parent / f".{output_path.stem}_concat.mp4"

    try:
        # Pass 1: Concatenate (with progress monitoring)
        if not _bsf_pass1_concat(input_files, concat_output, progress_callback):
            return False

        # Signal Pass 1 complete, Pass 2 starting
        if progress_callback:
            progress_callback(ProgressInfo(
                frame=len(input_files), fps=0, time_seconds=0, speed=0,
                percent=88.0,  # Concat done, BSF starting
            ))

        # Pass 2: BSF timelapse
        if not _bsf_pass2_timelapse(
            concat_output, output_path, packet_interval, output_fps
        ):
            return False

        if progress_callback:
            progress_callback(ProgressInfo(
                frame=frames_needed, fps=output_fps,
                time_seconds=target_duration, speed=0,
                percent=100.0,
            ))

        logger.info(
            "BSF timelapse created",
            output=str(output_path),
            file_count=len(input_files),
        )
        return True

    finally:
        if not keep_temp:
            concat_output.unlink(missing_ok=True)
