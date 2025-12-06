"""Multi-camera grid layout for video tiling.

Automatically calculates optimal grid dimensions and generates ffmpeg
filter_complex for xstack-based video tiling.
"""

import math
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from frigate_tools.observability import get_logger, traced_operation


@dataclass
class GridProgress:
    """Grid encoding progress information."""

    percent: float | None = None
    fps: float = 0.0
    speed: float = 0.0
    time_seconds: float = 0.0


def parse_ffmpeg_progress(line: str, total_duration: float | None = None) -> GridProgress | None:
    """Parse FFmpeg progress line for grid encoding."""
    time_match = re.search(r"time=(\d+):(\d+):([\d.]+)", line)
    fps_match = re.search(r"fps=\s*([\d.]+)", line)
    speed_match = re.search(r"speed=\s*([\d.]+)x", line)

    if not time_match:
        return None

    hours, minutes, seconds = time_match.groups()
    time_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    fps = float(fps_match.group(1)) if fps_match else 0.0
    speed = float(speed_match.group(1)) if speed_match else 0.0

    percent = None
    if total_duration and total_duration > 0:
        percent = min(100.0, (time_seconds / total_duration) * 100)

    return GridProgress(
        percent=percent,
        fps=fps,
        speed=speed,
        time_seconds=time_seconds,
    )


@dataclass
class GridLayout:
    """Grid layout dimensions."""

    rows: int
    cols: int

    @property
    def total_cells(self) -> int:
        """Total number of cells in the grid."""
        return self.rows * self.cols


def calculate_grid_layout(camera_count: int) -> GridLayout:
    """Calculate optimal grid layout for given camera count.

    Minimizes empty cells while keeping aspect ratio reasonable.

    Args:
        camera_count: Number of cameras to tile

    Returns:
        GridLayout with rows and columns
    """
    if camera_count <= 0:
        return GridLayout(rows=0, cols=0)

    if camera_count == 1:
        return GridLayout(rows=1, cols=1)

    if camera_count == 2:
        return GridLayout(rows=1, cols=2)

    # For 3+ cameras, calculate optimal grid
    # Try to make it roughly square, favoring more columns than rows
    cols = math.ceil(math.sqrt(camera_count))
    rows = math.ceil(camera_count / cols)

    return GridLayout(rows=rows, cols=cols)


def generate_xstack_filter(
    input_count: int,
    layout: GridLayout,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Generate ffmpeg xstack filter for grid layout.

    Args:
        input_count: Number of input videos
        layout: Grid layout (rows x cols)
        width: Optional output width per cell
        height: Optional output height per cell

    Returns:
        ffmpeg filter_complex string
    """
    if input_count <= 0 or layout.total_cells <= 0:
        return ""

    # Build layout string: pipe-separated row positions
    # e.g., "0_0|w0_0|0_h0|w0_h0" for 2x2 grid
    positions = []
    for i in range(input_count):
        row = i // layout.cols
        col = i % layout.cols

        # Position using relative widths/heights
        if col == 0:
            x = "0"
        else:
            # Sum of previous column widths
            x = "+".join(f"w{j}" for j in range(col))

        if row == 0:
            y = "0"
        else:
            # Sum of previous row heights
            y = "+".join(f"h{j * layout.cols}" for j in range(row))

        positions.append(f"{x}_{y}")

    layout_str = "|".join(positions)

    # Build filter chain
    filters = []

    # Scale each input if dimensions specified
    if width and height:
        for i in range(input_count):
            filters.append(f"[{i}:v]scale={width}:{height}[v{i}]")
        input_refs = "".join(f"[v{i}]" for i in range(input_count))
    else:
        input_refs = "".join(f"[{i}:v]" for i in range(input_count))

    # Add xstack filter
    xstack = f"{input_refs}xstack=inputs={input_count}:layout={layout_str}[out]"
    filters.append(xstack)

    return ";".join(filters)


@dataclass
class SyncedFileSet:
    """Synchronized file sets across cameras with gap handling."""

    camera_files: dict[str, list[Path]]
    gap_indices: set[int]  # Indices where at least one camera has a gap


def sync_file_lists(
    camera_files: dict[str, list[Path]],
) -> SyncedFileSet:
    """Synchronize file lists across cameras, identifying gaps.

    When cameras have different numbers of files, identifies positions
    where gaps exist. This allows proper handling during encoding.

    Args:
        camera_files: Dict mapping camera name to list of file paths

    Returns:
        SyncedFileSet with gap information
    """
    if not camera_files:
        return SyncedFileSet(camera_files={}, gap_indices=set())

    # Find max file count
    max_files = max(len(files) for files in camera_files.values())

    # Identify gap indices (where any camera is missing a file)
    gap_indices = set()
    for idx in range(max_files):
        for files in camera_files.values():
            if idx >= len(files):
                gap_indices.add(idx)
                break

    return SyncedFileSet(camera_files=camera_files, gap_indices=gap_indices)


def create_grid_video(
    camera_files: dict[str, list[Path]],
    output_path: Path,
    cell_width: int | None = None,
    cell_height: int | None = None,
    preset: str = "fast",
    progress_callback: Callable[[GridProgress], None] | None = None,
    estimated_duration: float | None = None,
) -> bool:
    """Create a grid video from multiple camera inputs.

    Args:
        camera_files: Dict mapping camera name to list of video files
        output_path: Output video path
        cell_width: Width of each cell in grid (auto-detected if None)
        cell_height: Height of each cell in grid (auto-detected if None)
        preset: FFmpeg encoding preset
        progress_callback: Optional callback for progress updates
        estimated_duration: Estimated output duration (for progress %)

    Returns:
        True if successful, False otherwise
    """
    logger = get_logger()

    if not camera_files:
        logger.error("No camera files provided")
        return False

    camera_names = list(camera_files.keys())
    camera_count = len(camera_names)

    with traced_operation(
        "create_grid_video",
        {"camera_count": camera_count, "output": str(output_path)},
    ):
        layout = calculate_grid_layout(camera_count)
        logger.info(
            "Grid layout calculated",
            cameras=camera_count,
            rows=layout.rows,
            cols=layout.cols,
        )

        # Sync file lists
        synced = sync_file_lists(camera_files)
        if synced.gap_indices:
            logger.warning(
                "Gaps detected in camera files",
                gap_count=len(synced.gap_indices),
            )

        # For each camera, create a concat file
        concat_files = []

        for camera in camera_names:
            files = camera_files[camera]
            if not files:
                logger.error(f"No files for camera {camera}")
                return False

            # Create concat file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=f"_{camera}.txt", delete=False
            ) as f:
                concat_file = Path(f.name)
                for file_path in files:
                    escaped = str(file_path).replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
                concat_files.append(concat_file)

        try:
            # Build ffmpeg command
            cmd = ["ffmpeg", "-y"]

            # Add input files (using concat demuxer for each camera)
            for concat_file in concat_files:
                cmd.extend(["-f", "concat", "-safe", "0", "-i", str(concat_file)])

            # Generate and add filter complex
            filter_complex = generate_xstack_filter(
                camera_count, layout, cell_width, cell_height
            )
            cmd.extend(["-filter_complex", filter_complex])

            # Output options
            cmd.extend([
                "-map", "[out]",
                "-preset", preset,
                "-an",  # No audio for grid
            ])

            # Add progress output if callback provided
            if progress_callback:
                cmd.extend(["-progress", "pipe:1"])

            cmd.append(str(output_path))

            logger.info("Starting grid encoding", cameras=camera_names)

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if progress_callback and process.stdout:
                # Parse progress from stdout
                for line in process.stdout:
                    progress = parse_ffmpeg_progress(line.strip(), estimated_duration)
                    if progress:
                        progress_callback(progress)
                # Read any remaining stderr and wait
                stderr = process.stderr.read() if process.stderr else ""
                process.wait()
            else:
                _, stderr = process.communicate()

            if process.returncode != 0:
                logger.error("Grid encoding failed", stderr=stderr)
                return False

            logger.info("Grid video created", output=str(output_path))
            return True

        finally:
            # Cleanup concat files
            for concat_file in concat_files:
                concat_file.unlink(missing_ok=True)
