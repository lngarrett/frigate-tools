"""Main CLI entry point for frigate-tools."""

import atexit
import re
import time as time_module
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from frigate_tools.clip import create_clip, create_multi_camera_clip, ClipProgress
from frigate_tools.file_list import generate_file_lists
from frigate_tools.grid import calculate_grid_layout, create_grid_video, GridProgress
from frigate_tools.observability import init_observability, shutdown_observability, get_logger, traced_operation
from frigate_tools.timelapse import (
    create_timelapse,
    encode_timelapse,
    concat_files,
    ProgressInfo,
    ConcatProgress,
    HWAccel,
    get_hwaccel,
)

app = typer.Typer(
    name="frigate-tools",
    help="CLI utilities for working with Frigate NVR recordings.",
    no_args_is_help=True,
)
console = Console()

timelapse_app = typer.Typer(help="Generate timelapses from Frigate recordings.")
clip_app = typer.Typer(help="Export clips from Frigate recordings.")

app.add_typer(timelapse_app, name="timelapse")
app.add_typer(clip_app, name="clip")


@app.callback()
def main_callback() -> None:
    """Frigate Tools - work with Frigate NVR recordings."""
    # Use sync export for CLI to ensure spans are sent before process exits
    init_observability(sync_export=True)
    atexit.register(shutdown_observability)


@timelapse_app.callback(invoke_without_command=True)
def timelapse_callback(ctx: typer.Context) -> None:
    """Generate timelapses from Frigate recordings."""
    if ctx.invoked_subcommand is None:
        console.print("[yellow]Use --help to see available options[/yellow]")


@clip_app.callback(invoke_without_command=True)
def clip_callback(ctx: typer.Context) -> None:
    """Export clips from Frigate recordings."""
    if ctx.invoked_subcommand is None:
        console.print("[yellow]Use --help to see available options[/yellow]")


# Default Frigate instance paths to search
DEFAULT_FRIGATE_PATHS = [
    Path("/data/nvr/frigate"),
    Path("/media/frigate"),
    Path("/var/lib/frigate"),
    Path.home() / "frigate",
]


def local_to_utc(dt: datetime) -> datetime:
    """Convert naive local datetime to naive UTC datetime.

    Frigate stores recordings using UTC timestamps in directory structure.
    User inputs are assumed to be local time and need conversion for file matching.

    Args:
        dt: Naive datetime in local time

    Returns:
        Naive datetime in UTC
    """
    # Get local timezone offset (accounting for DST)
    if time_module.daylight and time_module.localtime(dt.timestamp()).tm_isdst > 0:
        utc_offset = -time_module.altzone
    else:
        utc_offset = -time_module.timezone

    # Create timezone-aware local datetime, convert to UTC, return naive
    local_tz = timezone(timedelta(seconds=utc_offset))
    local_aware = dt.replace(tzinfo=local_tz)
    utc_aware = local_aware.astimezone(timezone.utc)
    return utc_aware.replace(tzinfo=None)


def find_frigate_instance() -> Path | None:
    """Auto-detect Frigate instance path.

    Returns:
        Path to Frigate instance containing recordings/, or None if not found
    """
    for path in DEFAULT_FRIGATE_PATHS:
        # Check for subdirectories that look like instances
        if path.exists():
            # Check if this path directly has recordings/
            if (path / "recordings").exists():
                return path
            # Check subdirectories (named instances like 'cherokee')
            for subdir in path.iterdir():
                if subdir.is_dir() and (subdir / "recordings").exists():
                    return subdir
    return None


def estimate_source_size(files: list[Path]) -> int:
    """Estimate total size of source files in bytes."""
    total = 0
    for f in files:
        if f.exists():
            total += f.stat().st_size
    return total


def estimate_output_size(source_size: int, target_duration: float, source_duration_estimate: float) -> int:
    """Estimate output file size based on source size and compression.

    Args:
        source_size: Total source file size in bytes
        target_duration: Target output duration in seconds
        source_duration_estimate: Estimated source duration in seconds

    Returns:
        Estimated output size in bytes
    """
    if source_duration_estimate <= 0:
        return 0

    # Ratio of output to source duration
    duration_ratio = target_duration / source_duration_estimate

    # Re-encoding typically produces 0.3-0.5x the size of -c copy concat
    # With fast preset, estimate ~0.4x of proportional size
    compression_factor = 0.4

    return int(source_size * duration_ratio * compression_factor)


def get_available_disk_space(path: Path) -> int:
    """Get available disk space in bytes for the filesystem containing path."""
    import shutil
    # Get the parent directory if path doesn't exist yet
    check_path = path if path.exists() else path.parent
    while not check_path.exists() and check_path != check_path.parent:
        check_path = check_path.parent
    usage = shutil.disk_usage(check_path)
    return usage.free


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def parse_duration(duration_str: str) -> float:
    """Parse duration string like '5m', '1h30m', '90s' to seconds.

    Args:
        duration_str: Duration string (e.g., '5m', '1h', '90s', '1h30m')

    Returns:
        Duration in seconds

    Raises:
        ValueError: If duration string is invalid
    """
    # Match patterns like: 5m, 1h, 90s, 1h30m, 2h30m15s
    pattern = r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?"
    match = re.fullmatch(pattern, duration_str.lower().strip())

    if not match or not any(match.groups()):
        raise ValueError(f"Invalid duration format: {duration_str}. Use format like 5m, 1h, 90s, 1h30m")

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    total_seconds = hours * 3600 + minutes * 60 + seconds

    if total_seconds <= 0:
        raise ValueError("Duration must be greater than 0")

    return float(total_seconds)


@timelapse_app.command("create")
def timelapse_create(
    cameras: Annotated[
        str,
        typer.Option(
            "--cameras", "-c",
            help="Comma-separated camera names (e.g., bporchcam,frontcam)"
        ),
    ],
    start: Annotated[
        datetime,
        typer.Option(
            "--start", "-s",
            help="Start time in local timezone (ISO format: 2025-12-01T08:00)",
            formats=["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"],
        ),
    ],
    end: Annotated[
        datetime,
        typer.Option(
            "--end", "-e",
            help="End time in local timezone (ISO format: 2025-12-05T16:00)",
            formats=["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"],
        ),
    ],
    duration: Annotated[
        str,
        typer.Option(
            "--duration", "-d",
            help="Target output duration (e.g., 5m, 1h, 90s)",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o",
            help="Output file path",
        ),
    ],
    instance: Annotated[
        Optional[Path],
        typer.Option(
            "--instance", "-i",
            help="Frigate instance path (auto-detected if not specified)",
        ),
    ] = None,
    skip_days: Annotated[
        Optional[str],
        typer.Option(
            "--skip-days",
            help="Days to skip (e.g., sat,sun)",
        ),
    ] = None,
    skip_hours: Annotated[
        Optional[str],
        typer.Option(
            "--skip-hours",
            help="Hour ranges to skip (e.g., 16-8 for 4pm to 8am)",
        ),
    ] = None,
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            help="FFmpeg encoding preset",
        ),
    ] = "fast",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be done without creating files",
        ),
    ] = False,
) -> None:
    """Create a timelapse from Frigate recordings.

    Example:
        frigate-tools timelapse create \\
            --cameras bporchcam,frontcam \\
            --start 2025-12-01T08:00 \\
            --end 2025-12-05T16:00 \\
            --duration 5m \\
            --skip-days sat,sun \\
            --skip-hours 16-8 \\
            -o timelapse.mp4
    """
    logger = get_logger()

    with traced_operation(
        "timelapse_create",
        {
            "cameras": cameras,
            "start": str(start),
            "end": str(end),
            "duration": duration,
            "output": str(output),
        },
    ):
        _timelapse_create_impl(
            cameras, start, end, duration, output, instance,
            skip_days, skip_hours, preset, dry_run, logger
        )


def _timelapse_create_impl(
    cameras: str,
    start: datetime,
    end: datetime,
    duration: str,
    output: Path,
    instance: Path | None,
    skip_days: str | None,
    skip_hours: str | None,
    preset: str,
    dry_run: bool,
    logger,
) -> None:
    """Implementation of timelapse_create command."""
    # Parse duration
    try:
        target_duration = parse_duration(duration)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Parse cameras
    camera_list = [c.strip() for c in cameras.split(",") if c.strip()]
    if not camera_list:
        console.print("[red]Error:[/red] No cameras specified")
        raise typer.Exit(1)

    # Auto-detect or validate instance path
    if instance is None:
        instance = find_frigate_instance()
        if instance is None:
            console.print(
                "[red]Error:[/red] Could not auto-detect Frigate instance. "
                "Use --instance to specify the path."
            )
            raise typer.Exit(1)
        console.print(f"[dim]Using Frigate instance: {instance}[/dim]")

    if not instance.exists():
        console.print(f"[red]Error:[/red] Instance path does not exist: {instance}")
        raise typer.Exit(1)

    # Parse skip options
    skip_days_list = [d.strip() for d in (skip_days or "").split(",") if d.strip()]
    skip_hours_list = [h.strip() for h in (skip_hours or "").split(",") if h.strip()]

    # Convert local time inputs to UTC for file matching (Frigate stores in UTC)
    start_utc = local_to_utc(start)
    end_utc = local_to_utc(end)

    console.print(f"[bold]Generating timelapse[/bold]")
    console.print(f"  Cameras: {', '.join(camera_list)}")
    console.print(f"  Time range: {start} to {end} (local)")
    console.print(f"  Target duration: {duration} ({target_duration}s)")
    if skip_days_list:
        console.print(f"  Skipping days: {', '.join(skip_days_list)}")
    if skip_hours_list:
        console.print(f"  Skipping hours: {', '.join(skip_hours_list)}")
    console.print()

    # Determine hardware acceleration once
    hwaccel = get_hwaccel()
    if hwaccel == HWAccel.NONE:
        console.print("[dim]Using software encoding.[/dim]")
    else:
        console.print(f"[dim]Using hardware acceleration: {hwaccel.value}[/dim]")
    console.print()

    # Generate file lists
    with console.status("[bold blue]Finding recording files..."):
        file_lists = generate_file_lists(
            cameras=camera_list,
            start=start_utc,
            end=end_utc,
            instance_path=instance,
            skip_days=skip_days_list if skip_days_list else None,
            skip_hours=skip_hours_list if skip_hours_list else None,
        )

    # Report file counts and calculate sizes
    total_files = 0
    all_files = []
    for camera, files in file_lists.items():
        console.print(f"  {camera}: {len(files)} files")
        total_files += len(files)
        all_files.extend(files)

    if total_files == 0:
        console.print("[red]Error:[/red] No recording files found")
        raise typer.Exit(1)

    console.print(f"  [bold]Total: {total_files} files[/bold]")

    # Calculate sizes for dry-run or disk space check
    source_size = estimate_source_size(all_files)
    # Estimate ~10 seconds per file (Frigate default segment length)
    source_duration_estimate = total_files * 10
    estimated_output_size = estimate_output_size(source_size, target_duration, source_duration_estimate)
    available_space = get_available_disk_space(output)

    console.print(f"  Source size: {format_size(source_size)}")
    console.print(f"  Estimated output: {format_size(estimated_output_size)}")
    console.print(f"  Available space: {format_size(available_space)}")
    console.print()

    # Dry-run mode - show what would happen and exit
    if dry_run:
        console.print("[bold yellow]Dry run - no files created[/bold yellow]")
        console.print()
        console.print("Would create:")
        console.print(f"  Output file: {output}")
        console.print(f"  Estimated size: {format_size(estimated_output_size)}")
        console.print()

        if estimated_output_size > available_space:
            console.print(
                f"[red]Warning:[/red] Estimated output ({format_size(estimated_output_size)}) "
                f"exceeds available space ({format_size(available_space)})"
            )
        else:
            console.print("[green]Disk space check: OK[/green]")

        console.print()
        console.print(f"Speed factor: {source_duration_estimate / target_duration:.1f}x")
        return

    # Check disk space before proceeding
    if estimated_output_size > available_space * 0.9:  # Leave 10% buffer
        console.print(
            f"[red]Error:[/red] Insufficient disk space. "
            f"Estimated output ({format_size(estimated_output_size)}) "
            f"may exceed available space ({format_size(available_space)})"
        )
        console.print("Use --dry-run to see details without creating files.")
        raise typer.Exit(1)

    # Create timelapse based on camera count
    if len(camera_list) == 1:
        # Single camera - use adaptive timelapse approach
        # (keyframe extraction for high speedups, concat+encode for low speedups)
        camera = camera_list[0]
        files = file_lists[camera]

        speedup = source_duration_estimate / target_duration
        if speedup >= 30.0:
            description = "Creating timelapse (keyframe extraction)..."
        else:
            description = "Creating timelapse..."

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(description, total=100)
            total_files = len(files)

            def update_progress(info: ProgressInfo) -> None:
                if info.percent is not None:
                    # Update description based on phase
                    # Frame-based: 0-60% extraction, 60-100% encoding
                    # Concat-based: 0-88% concat, 88-100% encoding
                    if info.percent < 60:
                        # Frame extraction phase - use actual total from callback if available
                        actual_total = info.total if info.total else total_files
                        files_done = info.frame if info.frame else int(info.percent / 60 * actual_total)
                        progress.update(
                            task,
                            completed=info.percent,
                            description=f"Extracting frames... ({files_done}/{actual_total} files)",
                        )
                    elif info.percent < 95:
                        # Encoding phase
                        progress.update(
                            task,
                            completed=info.percent,
                            description="Encoding video...",
                        )
                    else:
                        progress.update(
                            task,
                            completed=info.percent,
                            description="Finalizing...",
                        )

            success = create_timelapse(
                input_files=files,
                output_path=output,
                target_duration=target_duration,
                preset=preset,
                progress_callback=update_progress,
                hwaccel=hwaccel,
            )

        if not success:
            console.print("[red]Error:[/red] Timelapse creation failed")
            raise typer.Exit(1)

    else:
        # Multi-camera - use grid layout with two-step approach
        layout = calculate_grid_layout(len(camera_list))
        console.print(f"[dim]Grid layout: {layout.rows}x{layout.cols}[/dim]")

        # Step 1: Create grid video at full speed to temp file
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="frigate_grid_"))
        grid_temp = temp_dir / "grid_full_speed.mp4"

        try:
            # Estimate source duration for progress calculation
            grid_duration_estimate = total_files * 10  # ~10 seconds per segment

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Step 1/2: Creating grid layout...", total=100)

                def update_grid_progress(info: GridProgress) -> None:
                    if info.percent is not None:
                        progress.update(task, completed=info.percent)

                success = create_grid_video(
                    camera_files=file_lists,
                    output_path=grid_temp,
                    preset=preset,
                    progress_callback=update_grid_progress,
                    estimated_duration=grid_duration_estimate,
                    hwaccel=hwaccel,
                )

            if not success:
                console.print("[red]Error:[/red] Grid creation failed")
                raise typer.Exit(1)

            # Step 2: Apply timelapse encoding to the grid video
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Step 2/2: Encoding timelapse...", total=100)

                def update_progress(info: ProgressInfo) -> None:
                    if info.percent is not None:
                        progress.update(task, completed=info.percent)

                success = encode_timelapse(
                    input_path=grid_temp,
                    output_path=output,
                    target_duration=target_duration,
                    preset=preset,
                    progress_callback=update_progress,
                    hwaccel=hwaccel,
                )

            if not success:
                console.print("[red]Error:[/red] Timelapse encoding failed")
                raise typer.Exit(1)

        finally:
            # Clean up temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    # Report success
    file_size = output.stat().st_size
    size_mb = file_size / (1024 * 1024)
    console.print()
    console.print(f"[green]Success![/green] Created {output}")
    console.print(f"  Size: {size_mb:.1f} MB")


@clip_app.command("create")
def clip_create(
    cameras: Annotated[
        str,
        typer.Option(
            "--cameras", "-c",
            help="Comma-separated camera names (e.g., bporchcam,frontcam)"
        ),
    ],
    start: Annotated[
        datetime,
        typer.Option(
            "--start", "-s",
            help="Start time in local timezone (ISO format: 2025-12-01T12:00)",
            formats=["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"],
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output", "-o",
            help="Output file path (for single camera) or directory (for --separate)",
        ),
    ],
    end: Annotated[
        Optional[datetime],
        typer.Option(
            "--end", "-e",
            help="End time in local timezone (ISO format: 2025-12-01T12:05)",
            formats=["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"],
        ),
    ] = None,
    duration: Annotated[
        Optional[str],
        typer.Option(
            "--duration", "-d",
            help="Clip duration as alternative to --end (e.g., 5m, 1h)",
        ),
    ] = None,
    instance: Annotated[
        Optional[Path],
        typer.Option(
            "--instance", "-i",
            help="Frigate instance path (auto-detected if not specified)",
        ),
    ] = None,
    separate: Annotated[
        bool,
        typer.Option(
            "--separate",
            help="Create separate files for each camera instead of grid",
        ),
    ] = False,
    reencode: Annotated[
        bool,
        typer.Option(
            "--reencode",
            help="Re-encode video (slower but better compatibility)",
        ),
    ] = False,
    preset: Annotated[
        str,
        typer.Option(
            "--preset",
            help="FFmpeg encoding preset (only used with --reencode)",
        ),
    ] = "fast",
) -> None:
    """Create a clip from Frigate recordings.

    Example (single camera):
        frigate-tools clip create \\
            --cameras frontcam \\
            --start 2025-12-01T12:00 \\
            --end 2025-12-01T12:05 \\
            -o clip.mp4

    Example (multiple cameras with grid):
        frigate-tools clip create \\
            --cameras bporchcam,frontcam \\
            --start 2025-12-01T12:00 \\
            --duration 5m \\
            -o clip.mp4

    Example (separate files per camera):
        frigate-tools clip create \\
            --cameras bporchcam,frontcam \\
            --start 2025-12-01T12:00 \\
            --duration 5m \\
            --separate \\
            -o output_dir/
    """
    logger = get_logger()

    with traced_operation(
        "clip_create",
        {
            "cameras": cameras,
            "start": str(start),
            "end": str(end) if end else None,
            "duration": duration,
            "output": str(output),
            "separate": separate,
            "reencode": reencode,
        },
    ):
        _clip_create_impl(
            cameras, start, end, duration, output, instance,
            separate, reencode, preset, logger
        )


def _clip_create_impl(
    cameras: str,
    start: datetime,
    end: datetime | None,
    duration: str | None,
    output: Path,
    instance: Path | None,
    separate: bool,
    reencode: bool,
    preset: str,
    logger,
) -> None:
    """Implementation of clip_create command."""
    # Validate end time or duration
    if end is None and duration is None:
        console.print("[red]Error:[/red] Must specify either --end or --duration")
        raise typer.Exit(1)

    if end is not None and duration is not None:
        console.print("[red]Error:[/red] Cannot specify both --end and --duration")
        raise typer.Exit(1)

    # Calculate end time from duration if needed
    if duration is not None:
        try:
            duration_seconds = parse_duration(duration)
            from datetime import timedelta
            end = start + timedelta(seconds=duration_seconds)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Parse cameras
    camera_list = [c.strip() for c in cameras.split(",") if c.strip()]
    if not camera_list:
        console.print("[red]Error:[/red] No cameras specified")
        raise typer.Exit(1)

    # Auto-detect or validate instance path
    if instance is None:
        instance = find_frigate_instance()
        if instance is None:
            console.print(
                "[red]Error:[/red] Could not auto-detect Frigate instance. "
                "Use --instance to specify the path."
            )
            raise typer.Exit(1)
        console.print(f"[dim]Using Frigate instance: {instance}[/dim]")

    if not instance.exists():
        console.print(f"[red]Error:[/red] Instance path does not exist: {instance}")
        raise typer.Exit(1)

    # Convert local time inputs to UTC for file matching (Frigate stores in UTC)
    start_utc = local_to_utc(start)
    end_utc = local_to_utc(end)

    # Display info
    console.print(f"[bold]Creating clip[/bold]")
    console.print(f"  Cameras: {', '.join(camera_list)}")
    console.print(f"  Time range: {start} to {end} (local)")
    if len(camera_list) > 1:
        console.print(f"  Mode: {'separate files' if separate else 'grid layout'}")
    if reencode:
        console.print(f"  Re-encoding: yes (preset: {preset})")
    console.print()

    # Create clip(s)
    if len(camera_list) == 1:
        # Single camera
        camera = camera_list[0]

        if reencode:
            # Use progress bar for re-encoding operations
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Creating clip for {camera}...", total=100)

                def update_clip_progress(info: ClipProgress) -> None:
                    if info.percent is not None:
                        progress.update(task, completed=info.percent)
                    elif info.message:
                        progress.update(task, description=info.message)

                success = create_clip(
                    instance_path=instance,
                    camera=camera,
                    start=start_utc,
                    end=end_utc,
                    output_path=output,
                    reencode=reencode,
                    preset=preset,
                    progress_callback=update_clip_progress,
                )
        else:
            # Use spinner for fast stream copy
            with console.status(f"[bold blue]Creating clip for {camera}..."):
                success = create_clip(
                    instance_path=instance,
                    camera=camera,
                    start=start_utc,
                    end=end_utc,
                    output_path=output,
                    reencode=reencode,
                    preset=preset,
                )

        if not success:
            console.print("[red]Error:[/red] Clip creation failed")
            raise typer.Exit(1)

        file_size = output.stat().st_size
        size_mb = file_size / (1024 * 1024)
        console.print(f"[green]Success![/green] Created {output}")
        console.print(f"  Size: {size_mb:.1f} MB")

    else:
        # Multi-camera
        if separate:
            # Create output directory if needed
            output.mkdir(parents=True, exist_ok=True)

            with console.status("[bold blue]Creating clips..."):
                result = create_multi_camera_clip(
                    instance_path=instance,
                    cameras=camera_list,
                    start=start_utc,
                    end=end_utc,
                    output_dir=output,
                    separate=True,
                    reencode=reencode,
                    preset=preset,
                )

            if result is None:
                console.print("[red]Error:[/red] Clip creation failed")
                raise typer.Exit(1)

            console.print(f"[green]Success![/green] Created clips:")
            for camera, path in result.items():
                size_mb = path.stat().st_size / (1024 * 1024)
                console.print(f"  {camera}: {path} ({size_mb:.1f} MB)")

        else:
            # Grid layout
            with console.status("[bold blue]Creating grid clip..."):
                result = create_multi_camera_clip(
                    instance_path=instance,
                    cameras=camera_list,
                    start=start_utc,
                    end=end_utc,
                    output_dir=output.parent,
                    separate=False,
                    reencode=reencode,
                    preset=preset,
                )

            if result is None:
                console.print("[red]Error:[/red] Grid clip creation failed")
                raise typer.Exit(1)

            # Move to requested output path if different
            if isinstance(result, Path) and result != output:
                result.rename(output)

            file_size = output.stat().st_size
            size_mb = file_size / (1024 * 1024)
            console.print(f"[green]Success![/green] Created {output}")
            console.print(f"  Size: {size_mb:.1f} MB")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
