#!/usr/bin/env python3
"""Prototype: Image2pipe sparse extraction for timelapse generation.

Pipes sparse frames between two ffmpeg instances - first extracts at low rate, second encodes.

Three methods tested:
1. Direct frame rate reduction via -r flag
2. Select filter with rawvideo pipe
3. Keyframe-only with select and pipe

Usage:
    python image2pipe.py <file_list.txt> <output.mp4> <target_duration_seconds> [--method 1|2|3|all]

Example:
    python image2pipe.py files.txt output.mp4 300 --method all
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def get_video_resolution(file_path: Path) -> tuple[int, int]:
    """Get video resolution using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    return 1920, 1080  # Default fallback


def get_file_duration(file_path: Path) -> float:
    """Get video duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass
    return 5.0  # Default Frigate segment duration


def load_file_list(file_list_path: Path) -> list[Path]:
    """Load file list from ffmpeg concat format or plain text."""
    files = []
    with open(file_list_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Handle ffmpeg concat format: file 'path'
            if line.startswith("file "):
                # Extract path from: file 'path' or file "path"
                path_part = line[5:].strip()
                if path_part.startswith("'") and path_part.endswith("'"):
                    path_part = path_part[1:-1]
                elif path_part.startswith('"') and path_part.endswith('"'):
                    path_part = path_part[1:-1]
                files.append(Path(path_part))
            else:
                # Plain path
                files.append(Path(line))
    return files


def create_concat_file(files: list[Path], output_path: Path) -> None:
    """Create ffmpeg concat demuxer file."""
    with open(output_path, "w") as f:
        for file_path in files:
            escaped = str(file_path.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")


def method1_framerate_reduction(
    concat_file: Path,
    output_path: Path,
    target_fps: float,
    output_fps: float = 30.0,
) -> tuple[bool, float, str]:
    """Method 1: Direct frame rate reduction via -r flag.

    Extracts frames at target_fps and pipes to encoder.
    """
    print(f"\n=== Method 1: Framerate reduction (extract at {target_fps:.4f} fps) ===")

    # First process: extract at low framerate, output as image stream
    extract_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-r", str(target_fps),
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", "2",
        "-"
    ]

    # Second process: encode from image pipe
    encode_cmd = [
        "ffmpeg", "-y",
        "-f", "image2pipe",
        "-framerate", str(output_fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    print(f"Extract: {' '.join(extract_cmd[:10])}...")
    print(f"Encode: {' '.join(encode_cmd[:10])}...")

    start_time = time.time()

    try:
        # Create pipe between processes
        extract_proc = subprocess.Popen(
            extract_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        encode_proc = subprocess.Popen(
            encode_cmd,
            stdin=extract_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Allow extract_proc to receive SIGPIPE if encode_proc exits
        extract_proc.stdout.close()

        # Wait for encode to finish
        _, encode_stderr = encode_proc.communicate()
        extract_proc.wait()

        elapsed = time.time() - start_time

        if encode_proc.returncode != 0:
            return False, elapsed, encode_stderr.decode(errors="replace")

        return True, elapsed, ""

    except Exception as e:
        elapsed = time.time() - start_time
        return False, elapsed, str(e)


def method2_select_rawvideo(
    concat_file: Path,
    output_path: Path,
    frame_interval: int,
    width: int,
    height: int,
    output_fps: float = 30.0,
) -> tuple[bool, float, str]:
    """Method 2: Select filter with rawvideo pipe.

    Uses select filter to pick every Nth frame, pipes as raw video.
    """
    print(f"\n=== Method 2: Select filter + rawvideo pipe (every {frame_interval} frames) ===")

    select_expr = f"not(mod(n\\,{frame_interval}))"

    # First process: select frames and output as raw video
    extract_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-"
    ]

    # Second process: encode from raw video
    encode_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}",
        "-framerate", str(output_fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        str(output_path),
    ]

    print(f"Extract: {' '.join(extract_cmd[:10])}...")
    print(f"Encode: {' '.join(encode_cmd[:10])}...")

    start_time = time.time()

    try:
        extract_proc = subprocess.Popen(
            extract_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        encode_proc = subprocess.Popen(
            encode_cmd,
            stdin=extract_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        extract_proc.stdout.close()

        _, encode_stderr = encode_proc.communicate()
        extract_proc.wait()

        elapsed = time.time() - start_time

        if encode_proc.returncode != 0:
            return False, elapsed, encode_stderr.decode(errors="replace")

        return True, elapsed, ""

    except Exception as e:
        elapsed = time.time() - start_time
        return False, elapsed, str(e)


def method3_keyframe_select_pipe(
    concat_file: Path,
    output_path: Path,
    keyframe_interval: int,
    output_fps: float = 30.0,
) -> tuple[bool, float, str]:
    """Method 3: Keyframe-only with select and pipe.

    Decodes only keyframes (-skip_frame nokey), selects every Nth, pipes to encoder.
    This should be much faster since we skip decoding non-keyframes.
    """
    print(f"\n=== Method 3: Keyframe-only + select + pipe (every {keyframe_interval} keyframes) ===")

    select_expr = f"not(mod(n\\,{keyframe_interval}))"

    # First process: keyframe-only decode, select, output as PPM (lossless, fast)
    extract_cmd = [
        "ffmpeg", "-y",
        "-skip_frame", "nokey",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
        "-f", "image2pipe",
        "-vcodec", "ppm",
        "-"
    ]

    # Second process: encode from PPM pipe
    encode_cmd = [
        "ffmpeg", "-y",
        "-f", "image2pipe",
        "-framerate", str(output_fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    print(f"Extract: {' '.join(extract_cmd[:10])}...")
    print(f"Encode: {' '.join(encode_cmd[:10])}...")

    start_time = time.time()

    try:
        extract_proc = subprocess.Popen(
            extract_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        encode_proc = subprocess.Popen(
            encode_cmd,
            stdin=extract_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        extract_proc.stdout.close()

        _, encode_stderr = encode_proc.communicate()
        extract_proc.wait()

        elapsed = time.time() - start_time

        if encode_proc.returncode != 0:
            return False, elapsed, encode_stderr.decode(errors="replace")

        return True, elapsed, ""

    except Exception as e:
        elapsed = time.time() - start_time
        return False, elapsed, str(e)


def method3b_keyframe_mjpeg_pipe(
    concat_file: Path,
    output_path: Path,
    keyframe_interval: int,
    output_fps: float = 30.0,
) -> tuple[bool, float, str]:
    """Method 3b: Keyframe-only with MJPEG pipe (smaller than PPM).

    Like Method 3 but uses MJPEG which is compressed but still fast.
    """
    print(f"\n=== Method 3b: Keyframe-only + MJPEG pipe (every {keyframe_interval} keyframes) ===")

    select_expr = f"not(mod(n\\,{keyframe_interval}))"

    extract_cmd = [
        "ffmpeg", "-y",
        "-skip_frame", "nokey",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", "2",
        "-"
    ]

    encode_cmd = [
        "ffmpeg", "-y",
        "-f", "image2pipe",
        "-framerate", str(output_fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    print(f"Extract: {' '.join(extract_cmd[:10])}...")
    print(f"Encode: {' '.join(encode_cmd[:10])}...")

    start_time = time.time()

    try:
        extract_proc = subprocess.Popen(
            extract_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        encode_proc = subprocess.Popen(
            encode_cmd,
            stdin=extract_proc.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        extract_proc.stdout.close()

        _, encode_stderr = encode_proc.communicate()
        extract_proc.wait()

        elapsed = time.time() - start_time

        if encode_proc.returncode != 0:
            return False, elapsed, encode_stderr.decode(errors="replace")

        return True, elapsed, ""

    except Exception as e:
        elapsed = time.time() - start_time
        return False, elapsed, str(e)


def run_tests(
    file_list_path: Path,
    output_base: Path,
    target_duration: float,
    methods: list[str],
    output_fps: float = 30.0,
) -> dict:
    """Run selected test methods and collect results."""

    # Load and analyze input files
    files = load_file_list(file_list_path)
    if not files:
        print(f"Error: No files found in {file_list_path}")
        sys.exit(1)

    print(f"Loaded {len(files)} files from {file_list_path}")

    # Sample first file for resolution
    first_file = files[0]
    if not first_file.exists():
        print(f"Error: First file not found: {first_file}")
        sys.exit(1)

    width, height = get_video_resolution(first_file)
    print(f"Video resolution: {width}x{height}")

    # Calculate source duration (assuming ~5s per segment)
    sample_duration = get_file_duration(first_file)
    source_duration = len(files) * sample_duration
    print(f"Estimated source duration: {source_duration:.0f}s ({source_duration/3600:.1f}h)")
    print(f"Target duration: {target_duration:.0f}s")

    # Calculate parameters
    speedup = source_duration / target_duration
    frames_needed = int(target_duration * output_fps)

    # For method 1: target extraction fps
    target_fps = frames_needed / source_duration

    # For method 2: frame interval (assumes 30fps source)
    source_fps = 30.0
    total_source_frames = source_duration * source_fps
    frame_interval = max(1, int(total_source_frames / frames_needed))

    # For method 3: keyframe interval (assumes ~1 keyframe/second)
    keyframes_available = int(source_duration)  # ~1 keyframe per second
    keyframe_interval = max(1, keyframes_available // frames_needed)

    print(f"\nSpeedup: {speedup:.0f}x")
    print(f"Frames needed: {frames_needed}")
    print(f"Target extraction fps: {target_fps:.6f}")
    print(f"Frame interval (for select): {frame_interval}")
    print(f"Keyframe interval: {keyframe_interval}")

    # Create concat file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file = Path(f.name)
        create_concat_file(files, concat_file)

    results = {}

    try:
        # Run selected methods
        if "1" in methods or "all" in methods:
            output = output_base.parent / f"{output_base.stem}_method1.mp4"
            success, elapsed, error = method1_framerate_reduction(
                concat_file, output, target_fps, output_fps
            )
            results["method1"] = {
                "success": success,
                "elapsed": elapsed,
                "error": error,
                "output": output if success else None,
            }
            print(f"Method 1: {'SUCCESS' if success else 'FAILED'} in {elapsed:.1f}s")
            if error:
                print(f"  Error: {error[:200]}...")

        if "2" in methods or "all" in methods:
            output = output_base.parent / f"{output_base.stem}_method2.mp4"
            success, elapsed, error = method2_select_rawvideo(
                concat_file, output, frame_interval, width, height, output_fps
            )
            results["method2"] = {
                "success": success,
                "elapsed": elapsed,
                "error": error,
                "output": output if success else None,
            }
            print(f"Method 2: {'SUCCESS' if success else 'FAILED'} in {elapsed:.1f}s")
            if error:
                print(f"  Error: {error[:200]}...")

        if "3" in methods or "all" in methods:
            output = output_base.parent / f"{output_base.stem}_method3.mp4"
            success, elapsed, error = method3_keyframe_select_pipe(
                concat_file, output, keyframe_interval, output_fps
            )
            results["method3"] = {
                "success": success,
                "elapsed": elapsed,
                "error": error,
                "output": output if success else None,
            }
            print(f"Method 3: {'SUCCESS' if success else 'FAILED'} in {elapsed:.1f}s")
            if error:
                print(f"  Error: {error[:200]}...")

        if "3b" in methods or "all" in methods:
            output = output_base.parent / f"{output_base.stem}_method3b.mp4"
            success, elapsed, error = method3b_keyframe_mjpeg_pipe(
                concat_file, output, keyframe_interval, output_fps
            )
            results["method3b"] = {
                "success": success,
                "elapsed": elapsed,
                "error": error,
                "output": output if success else None,
            }
            print(f"Method 3b: {'SUCCESS' if success else 'FAILED'} in {elapsed:.1f}s")
            if error:
                print(f"  Error: {error[:200]}...")

    finally:
        concat_file.unlink(missing_ok=True)

    return results


def print_summary(results: dict, source_duration: float) -> None:
    """Print summary of test results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [(name, r) for name, r in results.items() if r["success"]]
    failed = [(name, r) for name, r in results.items() if not r["success"]]

    if successful:
        print("\nSuccessful methods (sorted by speed):")
        successful.sort(key=lambda x: x[1]["elapsed"])
        for name, r in successful:
            speedup = source_duration / r["elapsed"]
            print(f"  {name}: {r['elapsed']:.1f}s ({speedup:.1f}x realtime)")
            if r["output"]:
                size_mb = r["output"].stat().st_size / (1024 * 1024)
                print(f"    Output: {r['output']} ({size_mb:.1f} MB)")

    if failed:
        print("\nFailed methods:")
        for name, r in failed:
            print(f"  {name}: {r['error'][:100]}...")

    if successful:
        best_name, best_result = successful[0]
        print(f"\n🏆 Best method: {best_name} ({best_result['elapsed']:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Test image2pipe approaches for timelapse generation"
    )
    parser.add_argument("file_list", type=Path, help="Path to file list (concat format or plain)")
    parser.add_argument("output", type=Path, help="Output base path (methods add suffix)")
    parser.add_argument("target_duration", type=float, help="Target duration in seconds")
    parser.add_argument(
        "--method", "-m",
        choices=["1", "2", "3", "3b", "all"],
        default="all",
        help="Which method(s) to test (default: all)"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=30.0,
        help="Output frame rate (default: 30)"
    )

    args = parser.parse_args()

    if not args.file_list.exists():
        print(f"Error: File list not found: {args.file_list}")
        sys.exit(1)

    methods = [args.method] if args.method != "all" else ["all"]

    # Load files to calculate source duration for summary
    files = load_file_list(args.file_list)
    sample_duration = get_file_duration(files[0]) if files else 5.0
    source_duration = len(files) * sample_duration

    results = run_tests(
        args.file_list,
        args.output,
        args.target_duration,
        methods,
        args.fps,
    )

    print_summary(results, source_duration)


if __name__ == "__main__":
    main()
