"""Main CLI entry point for frigate-tools."""

import atexit
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from frigate_tools.file_list import generate_file_lists
from frigate_tools.grid import calculate_grid_layout, create_grid_video
from frigate_tools.observability import init_observability, shutdown_observability, get_logger
from frigate_tools.timelapse import create_timelapse, ProgressInfo

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
    init_observability()
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
            help="Start time (ISO format: 2025-12-01T08:00)",
            formats=["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"],
        ),
    ],
    end: Annotated[
        datetime,
        typer.Option(
            "--end", "-e",
            help="End time (ISO format: 2025-12-05T16:00)",
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

    console.print(f"[bold]Generating timelapse[/bold]")
    console.print(f"  Cameras: {', '.join(camera_list)}")
    console.print(f"  Time range: {start} to {end}")
    console.print(f"  Target duration: {duration} ({target_duration}s)")
    if skip_days_list:
        console.print(f"  Skipping days: {', '.join(skip_days_list)}")
    if skip_hours_list:
        console.print(f"  Skipping hours: {', '.join(skip_hours_list)}")
    console.print()

    # Generate file lists
    with console.status("[bold blue]Finding recording files..."):
        file_lists = generate_file_lists(
            cameras=camera_list,
            start=start,
            end=end,
            instance_path=instance,
            skip_days=skip_days_list if skip_days_list else None,
            skip_hours=skip_hours_list if skip_hours_list else None,
        )

    # Report file counts
    total_files = 0
    for camera, files in file_lists.items():
        console.print(f"  {camera}: {len(files)} files")
        total_files += len(files)

    if total_files == 0:
        console.print("[red]Error:[/red] No recording files found")
        raise typer.Exit(1)

    console.print(f"  [bold]Total: {total_files} files[/bold]")
    console.print()

    # Create timelapse based on camera count
    if len(camera_list) == 1:
        # Single camera - use timelapse module directly
        camera = camera_list[0]
        files = file_lists[camera]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Encoding timelapse...", total=100)

            def update_progress(info: ProgressInfo) -> None:
                if info.percent is not None:
                    progress.update(task, completed=info.percent)

            success = create_timelapse(
                input_files=files,
                output_path=output,
                target_duration=target_duration,
                preset=preset,
                progress_callback=update_progress,
            )

        if not success:
            console.print("[red]Error:[/red] Timelapse creation failed")
            raise typer.Exit(1)

    else:
        # Multi-camera - use grid layout
        layout = calculate_grid_layout(len(camera_list))
        console.print(f"[dim]Grid layout: {layout.rows}x{layout.cols}[/dim]")

        with console.status("[bold blue]Creating grid timelapse..."):
            success = create_grid_video(
                camera_files=file_lists,
                output_path=output,
                preset=preset,
            )

        if not success:
            console.print("[red]Error:[/red] Grid timelapse creation failed")
            raise typer.Exit(1)

    # Report success
    file_size = output.stat().st_size
    size_mb = file_size / (1024 * 1024)
    console.print()
    console.print(f"[green]Success![/green] Created {output}")
    console.print(f"  Size: {size_mb:.1f} MB")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
