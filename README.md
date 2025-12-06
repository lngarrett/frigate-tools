# frigate-tools

CLI utilities for working with Frigate NVR recordings.

## Features

- **Timelapse generation** - Create timelapses from camera recordings with configurable speed
- **Clip export** - Extract clips from recordings with optional re-encoding
- **Multi-camera support** - Grid layouts for combining multiple camera feeds
- **Calendar filtering** - Skip specific days or hours (e.g., skip weekends, night hours)
- **Auto-detection** - Automatically finds Frigate instance paths

## Installation

```bash
# From source
git clone https://github.com/lngarrett/frigate-tools.git
cd frigate-tools
pip install -e .

# Or with uv
uv pip install -e .
```

## Usage

### Timelapse

Create a timelapse from camera recordings:

```bash
# Single camera timelapse
frigate-tools timelapse create \
    --cameras bporchcam \
    --start 2025-12-01T08:00 \
    --end 2025-12-05T16:00 \
    --duration 5m \
    -o timelapse.mp4

# Multi-camera grid timelapse
frigate-tools timelapse create \
    --cameras bporchcam,frontcam,byardcam,eastcam \
    --start 2025-12-01T08:00 \
    --end 2025-12-05T16:00 \
    --duration 5m \
    -o timelapse.mp4

# With calendar filtering (skip weekends and night hours)
frigate-tools timelapse create \
    --cameras bporchcam \
    --start 2025-12-01T08:00 \
    --end 2025-12-05T16:00 \
    --duration 5m \
    --skip-days sat,sun \
    --skip-hours 16-8 \
    -o timelapse.mp4
```

### Clips

Export video clips from recordings:

```bash
# Single camera clip
frigate-tools clip create \
    --cameras frontcam \
    --start 2025-12-01T12:00 \
    --end 2025-12-01T12:05 \
    -o clip.mp4

# Using duration instead of end time
frigate-tools clip create \
    --cameras frontcam \
    --start 2025-12-01T12:00 \
    --duration 5m \
    -o clip.mp4

# Multi-camera grid
frigate-tools clip create \
    --cameras bporchcam,frontcam \
    --start 2025-12-01T12:00 \
    --duration 5m \
    -o clip.mp4

# Separate files per camera
frigate-tools clip create \
    --cameras bporchcam,frontcam \
    --start 2025-12-01T12:00 \
    --duration 5m \
    --separate \
    -o output_dir/
```

### Options

Common options for both commands:

| Option | Description |
|--------|-------------|
| `--cameras, -c` | Comma-separated camera names |
| `--start, -s` | Start time (ISO format) |
| `--end, -e` | End time (ISO format) |
| `--duration, -d` | Duration (e.g., 5m, 1h, 90s) |
| `--output, -o` | Output file or directory |
| `--instance, -i` | Frigate instance path (auto-detected if not specified) |

Timelapse-specific options:

| Option | Description |
|--------|-------------|
| `--skip-days` | Days to skip (e.g., sat,sun) |
| `--skip-hours` | Hour ranges to skip (e.g., 16-8 for 4pm to 8am) |
| `--preset` | FFmpeg encoding preset (default: fast) |

Clip-specific options:

| Option | Description |
|--------|-------------|
| `--separate` | Create separate files for each camera |
| `--reencode` | Re-encode video (slower but better compatibility) |
| `--preset` | FFmpeg encoding preset (only with --reencode) |

## Frigate Directory Structure

frigate-tools expects recordings in the standard Frigate structure:

```
{instance}/recordings/{YYYY-MM-DD}/{HH}/{camera}/{MM}.{SS}.mp4
```

The tool will auto-detect Frigate instances from common paths:
- `/data/nvr/frigate`
- `/media/frigate`
- `/var/lib/frigate`
- `~/frigate`

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=frigate_tools
```

## Requirements

- Python 3.11+
- FFmpeg (must be in PATH)
- Frigate NVR with recordings

## License

MIT
