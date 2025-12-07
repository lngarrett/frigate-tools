#!/usr/bin/env python3
"""Prototype: Two-pass concat + bitstream filter timelapse.

Approach:
1. Pass 1: Fast concat all files with -c copy (no decode)
2. Pass 2: Bitstream filter to drop packets and retime (no decode)

This works around the issue where -discard nokey is ignored with concat demuxer.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def get_video_duration(file_path: Path) -> float:
    """Get video duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def count_keyframes(file_path: Path) -> int:
    """Count keyframes in a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-skip_frame", "nokey",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def pass1_concat(file_list: Path, output: Path) -> tuple[bool, float]:
    """Pass 1: Concatenate all files with stream copy (no decode)."""
    print(f"\n=== Pass 1: Concatenating files ===")
    print(f"Input list: {file_list}")
    print(f"Output: {output}")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(file_list),
        "-c", "copy",
        "-an",  # No audio
        str(output),
    ]

    print(f"Command: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False, elapsed

    size_gb = output.stat().st_size / (1024**3)
    print(f"Pass 1 complete: {elapsed:.1f}s, output size: {size_gb:.2f} GB")
    return True, elapsed


def pass2_bsf(input_file: Path, output: Path, packet_interval: int, output_fps: float) -> tuple[bool, float]:
    """Pass 2: Apply bitstream filter to drop packets and retime."""
    print(f"\n=== Pass 2: Bitstream filter ===")
    print(f"Input: {input_file}")
    print(f"Output: {output}")
    print(f"Keeping every {packet_interval}th packet")
    print(f"Output framerate: {output_fps} fps")

    # Build BSF expression
    # noise=drop keeps packets where expression is 0 (false)
    # We want to DROP packets where mod(n, interval) != 0
    # So we keep packets where mod(n, interval) == 0
    drop_expr = f"mod(n\\,{packet_interval})"

    # setts retimes packets: N is packet number, TB_OUT is output timebase
    setts_expr = f"N/{output_fps}/TB_OUT"

    cmd = [
        "ffmpeg", "-y",
        "-discard", "nokey",  # Only read keyframes
        "-i", str(input_file),
        "-c", "copy",
        "-an",
        "-bsf:v", f"noise=drop='{drop_expr}',setts=ts='{setts_expr}'",
        str(output),
    ]

    print(f"Command: {' '.join(cmd)}")

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False, elapsed

    size_mb = output.stat().st_size / (1024**2)
    print(f"Pass 2 complete: {elapsed:.1f}s, output size: {size_mb:.2f} MB")
    return True, elapsed


def generate_file_list(files: list[Path], output: Path) -> None:
    """Generate ffmpeg concat file list."""
    with open(output, "w") as f:
        for file_path in files:
            escaped = str(file_path.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    print(f"Generated file list with {len(files)} files")


def main():
    parser = argparse.ArgumentParser(description="Two-pass BSF timelapse prototype")
    parser.add_argument("--files", type=Path, help="File containing list of input files (one per line)")
    parser.add_argument("--file-list", type=Path, help="Existing ffmpeg concat file list")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output video path")
    parser.add_argument("--duration", "-d", type=float, default=300.0, help="Target duration in seconds")
    parser.add_argument("--fps", type=float, default=30.0, help="Output framerate")
    parser.add_argument("--keep-temp", action="store_true", help="Keep intermediate files")

    args = parser.parse_args()

    # Determine file list
    if args.file_list:
        file_list = args.file_list
        # Count files in list
        with open(file_list) as f:
            file_count = sum(1 for line in f if line.strip().startswith("file "))
    elif args.files:
        # Read file paths and generate list
        with open(args.files) as f:
            files = [Path(line.strip()) for line in f if line.strip()]
        file_count = len(files)
        file_list = args.output.parent / f".{args.output.stem}_list.txt"
        generate_file_list(files, file_list)
    else:
        print("Error: Must provide --files or --file-list")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Two-Pass BSF Timelapse Prototype")
    print(f"{'='*60}")
    print(f"Input files: {file_count}")
    print(f"Target duration: {args.duration}s")
    print(f"Output fps: {args.fps}")
    print(f"Output: {args.output}")

    # Calculate parameters
    # Assume ~10 seconds per file, ~1 keyframe per second
    estimated_duration = file_count * 10.0
    estimated_keyframes = int(estimated_duration)  # ~1 per second
    frames_needed = int(args.duration * args.fps)
    packet_interval = max(1, estimated_keyframes // frames_needed)

    print(f"\nEstimated source duration: {estimated_duration:.0f}s")
    print(f"Estimated keyframes: {estimated_keyframes}")
    print(f"Frames needed: {frames_needed}")
    print(f"Packet interval: {packet_interval} (keep every {packet_interval}th)")

    # Temp file for concat output
    concat_output = args.output.parent / f".{args.output.stem}_concat.mp4"

    total_start = time.time()

    # Pass 1: Concat
    success, pass1_time = pass1_concat(file_list, concat_output)
    if not success:
        print("Pass 1 failed!")
        sys.exit(1)

    # Get actual keyframe count from concat file
    print("\nCounting keyframes in concatenated file...")
    actual_keyframes = count_keyframes(concat_output)
    if actual_keyframes > 0:
        packet_interval = max(1, actual_keyframes // frames_needed)
        print(f"Actual keyframes: {actual_keyframes}")
        print(f"Adjusted packet interval: {packet_interval}")

    # Pass 2: BSF
    success, pass2_time = pass2_bsf(concat_output, args.output, packet_interval, args.fps)
    if not success:
        print("Pass 2 failed!")
        sys.exit(1)

    total_time = time.time() - total_start

    # Cleanup
    if not args.keep_temp:
        concat_output.unlink(missing_ok=True)
        if args.files:
            file_list.unlink(missing_ok=True)

    # Get output duration
    output_duration = get_video_duration(args.output)

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Pass 1 (concat):     {pass1_time:>8.1f}s")
    print(f"Pass 2 (BSF):        {pass2_time:>8.1f}s")
    print(f"Total time:          {total_time:>8.1f}s")
    print(f"Output duration:     {output_duration:>8.1f}s (target: {args.duration}s)")
    print(f"Output size:         {args.output.stat().st_size / (1024**2):>8.1f} MB")

    speedup = estimated_duration / output_duration if output_duration > 0 else 0
    print(f"Effective speedup:   {speedup:>8.1f}x")


if __name__ == "__main__":
    main()
