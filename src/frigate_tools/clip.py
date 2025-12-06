"""Clip file selection and concatenation for Frigate recordings.

Finds recording segments overlapping a time range and concatenates them
into a single output file.
"""

import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from frigate_tools.file_list import find_recording_files
from frigate_tools.observability import get_logger, traced_operation


@dataclass
class ClipProgress:
    """Clip creation progress information."""

    stage: str  # "finding", "concatenating", "encoding"
    percent: float | None = None
    message: str | None = None


def find_overlapping_segments(
    instance_path: Path,
    camera: str,
    start: datetime,
    end: datetime,
) -> list[Path]:
    """Find recording segments that overlap with the given time range.

    Unlike generate_file_lists, this doesn't apply calendar filtering -
    clips should include all content in the time range.

    Args:
        instance_path: Path to Frigate instance
        camera: Camera name
        start: Start of clip (inclusive)
        end: End of clip (exclusive)

    Returns:
        Sorted list of file paths
    """
    # Use existing file list logic without calendar filtering
    return find_recording_files(
        instance_path=instance_path,
        camera=camera,
        start=start,
        end=end,
        skip_days=None,
        skip_hours=None,
    )


def concat_clip(
    input_files: list[Path],
    output_path: Path,
    reencode: bool = False,
    preset: str = "fast",
) -> bool:
    """Concatenate video segments into a single clip.

    Args:
        input_files: List of video files to concatenate
        output_path: Output file path
        reencode: If True, re-encode video (slower, better compatibility).
                  If False (default), use -c copy (fast, no quality loss).
        preset: FFmpeg encoding preset (only used if reencode=True)

    Returns:
        True if successful, False otherwise
    """
    logger = get_logger()

    if not input_files:
        logger.error("No input files provided")
        return False

    with traced_operation(
        "concat_clip",
        {"file_count": len(input_files), "reencode": reencode},
    ):
        # Create concat file list
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            concat_file = Path(f.name)
            for file_path in input_files:
                escaped = str(file_path).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
            ]

            if reencode:
                # Re-encode for better compatibility
                cmd.extend(["-preset", preset])
            else:
                # Stream copy (fast, no re-encoding)
                cmd.extend(["-c", "copy"])

            cmd.append(str(output_path))

            logger.info(
                "Starting clip concatenation",
                file_count=len(input_files),
                reencode=reencode,
            )

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            _, stderr = process.communicate()

            if process.returncode != 0:
                logger.error("Clip concatenation failed", stderr=stderr)
                return False

            logger.info("Clip created", output=str(output_path))
            return True

        finally:
            concat_file.unlink(missing_ok=True)


def create_clip(
    instance_path: Path,
    camera: str,
    start: datetime,
    end: datetime,
    output_path: Path,
    reencode: bool = False,
    preset: str = "fast",
    progress_callback: Callable[[ClipProgress], None] | None = None,
) -> bool:
    """Create a clip from Frigate recordings.

    Finds all recording segments overlapping the time range and
    concatenates them into a single file.

    Args:
        instance_path: Path to Frigate instance
        camera: Camera name
        start: Clip start time
        end: Clip end time
        output_path: Output file path
        reencode: Re-encode video (slower) vs stream copy (fast, default)
        preset: FFmpeg preset (only used if reencode=True)
        progress_callback: Optional callback for progress updates

    Returns:
        True if successful, False otherwise
    """
    logger = get_logger()

    with traced_operation(
        "create_clip",
        {"camera": camera, "start": start.isoformat(), "end": end.isoformat()},
    ):
        # Report finding stage
        if progress_callback:
            progress_callback(ClipProgress(
                stage="finding",
                message=f"Finding segments for {camera}...",
            ))

        # Find overlapping segments
        files = find_overlapping_segments(instance_path, camera, start, end)

        if not files:
            logger.error("No recording files found for clip")
            return False

        logger.info(
            "Found segments for clip",
            camera=camera,
            file_count=len(files),
        )

        # Report concatenation stage
        if progress_callback:
            progress_callback(ClipProgress(
                stage="concatenating",
                message=f"Concatenating {len(files)} segments...",
            ))

        # Concatenate
        success = concat_clip(
            input_files=files,
            output_path=output_path,
            reencode=reencode,
            preset=preset,
        )

        if success and progress_callback:
            progress_callback(ClipProgress(
                stage="complete",
                percent=100.0,
                message="Clip created successfully",
            ))

        return success


def create_multi_camera_clip(
    instance_path: Path,
    cameras: list[str],
    start: datetime,
    end: datetime,
    output_dir: Path,
    separate: bool = True,
    reencode: bool = False,
    preset: str = "fast",
    progress_callback: Callable[[ClipProgress], None] | None = None,
) -> dict[str, Path] | Path | None:
    """Create clips from multiple cameras.

    Args:
        instance_path: Path to Frigate instance
        cameras: List of camera names
        start: Clip start time
        end: Clip end time
        output_dir: Output directory (for separate mode) or base name
        separate: If True, create individual files per camera.
                  If False, create grid layout (requires grid module).
        reencode: Re-encode video
        preset: FFmpeg preset
        progress_callback: Optional callback for progress updates

    Returns:
        If separate=True: dict mapping camera name to output path
        If separate=False: single Path to grid output
        Returns None on failure
    """
    logger = get_logger()

    with traced_operation(
        "create_multi_camera_clip",
        {"cameras": ",".join(cameras), "separate": separate},
    ):
        if separate:
            # Create individual clips for each camera
            results = {}
            for i, camera in enumerate(cameras):
                if progress_callback:
                    progress_callback(ClipProgress(
                        stage="processing",
                        percent=(i / len(cameras)) * 100,
                        message=f"Processing {camera}...",
                    ))

                output_path = output_dir / f"{camera}_{start.strftime('%Y%m%d_%H%M')}.mp4"

                success = create_clip(
                    instance_path=instance_path,
                    camera=camera,
                    start=start,
                    end=end,
                    output_path=output_path,
                    reencode=reencode,
                    preset=preset,
                )

                if not success:
                    logger.error(f"Failed to create clip for {camera}")
                    return None

                results[camera] = output_path

            return results

        else:
            # Grid layout - collect files for each camera then use grid module
            from frigate_tools.grid import create_grid_video

            camera_files = {}
            for camera in cameras:
                files = find_overlapping_segments(instance_path, camera, start, end)
                if not files:
                    logger.error(f"No files found for camera {camera}")
                    return None
                camera_files[camera] = files

            # Output to single file
            output_path = output_dir / f"grid_{start.strftime('%Y%m%d_%H%M')}.mp4"

            success = create_grid_video(
                camera_files=camera_files,
                output_path=output_path,
                preset=preset,
            )

            if not success:
                return None

            return output_path
