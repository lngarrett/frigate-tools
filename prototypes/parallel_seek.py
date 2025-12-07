#!/usr/bin/env python3
"""Prototype: Parallel seek-based frame extraction for timelapse generation.

This approach:
1. Calculates which timestamps need frames based on target duration
2. Uses N parallel ffmpeg processes to seek and extract single frames
3. Encodes extracted frames to video

Usage:
    python parallel_seek.py --files file_list.txt --output timelapse.mp4 --duration 15 --workers 16
"""

import argparse
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import shutil
import re


@dataclass
class FileInfo:
    """Information about a single recording file."""
    path: Path
    start_time: float  # Cumulative start time in seconds
    duration: float = 10.0  # Frigate default segment length


@dataclass
class FrameExtraction:
    """A frame to extract."""
    frame_index: int
    file_path: Path
    seek_offset: float  # Offset within the file


@dataclass
class TimingResults:
    """Timing results for each phase."""
    file_analysis: float = 0.0
    frame_calculation: float = 0.0
    frame_extraction: float = 0.0
    video_encoding: float = 0.0
    total: float = 0.0


def parse_file_list(file_list_path: Path) -> list[Path]:
    """Parse a file containing list of video paths."""
    files = []
    with open(file_list_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle ffmpeg concat format: file 'path'
                if line.startswith("file '"):
                    path = line[6:-1]  # Remove "file '" and trailing "'"
                else:
                    path = line
                files.append(Path(path))
    return files


def analyze_files(files: list[Path], segment_duration: float = 10.0) -> list[FileInfo]:
    """Build timeline from file list.

    Assumes files are ordered chronologically and each is ~segment_duration seconds.
    """
    file_infos = []
    current_time = 0.0

    for path in files:
        file_infos.append(FileInfo(
            path=path,
            start_time=current_time,
            duration=segment_duration,
        ))
        current_time += segment_duration

    return file_infos


def calculate_frame_extractions(
    file_infos: list[FileInfo],
    target_duration: float,
    output_fps: float = 30.0,
) -> list[FrameExtraction]:
    """Calculate which frames to extract and from which files.

    Args:
        file_infos: List of files with timing info
        target_duration: Target output duration in seconds
        output_fps: Output framerate

    Returns:
        List of FrameExtraction objects describing each frame to extract
    """
    if not file_infos:
        return []

    total_source_duration = file_infos[-1].start_time + file_infos[-1].duration
    total_frames_needed = int(target_duration * output_fps)

    # Calculate time interval between frames in source
    source_interval = total_source_duration / total_frames_needed

    extractions = []
    file_idx = 0

    for frame_num in range(total_frames_needed):
        source_time = frame_num * source_interval

        # Find the file containing this timestamp
        while file_idx < len(file_infos) - 1:
            file_end = file_infos[file_idx].start_time + file_infos[file_idx].duration
            if source_time < file_end:
                break
            file_idx += 1

        file_info = file_infos[file_idx]
        seek_offset = source_time - file_info.start_time

        extractions.append(FrameExtraction(
            frame_index=frame_num,
            file_path=file_info.path,
            seek_offset=max(0, seek_offset),
        ))

    return extractions


def extract_single_frame(args: tuple) -> tuple[int, bool, str]:
    """Extract a single frame using ffmpeg.

    Args:
        args: Tuple of (frame_index, file_path, seek_offset, output_path)

    Returns:
        Tuple of (frame_index, success, error_message)
    """
    frame_index, file_path, seek_offset, output_path = args

    cmd = [
        "ffmpeg",
        "-ss", str(seek_offset),
        "-i", str(file_path),
        "-frames:v", "1",
        "-q:v", "2",  # High quality JPEG
        "-y",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and Path(output_path).exists():
            return (frame_index, True, "")
        else:
            return (frame_index, False, result.stderr.decode()[:200])
    except subprocess.TimeoutExpired:
        return (frame_index, False, "Timeout")
    except Exception as e:
        return (frame_index, False, str(e))


def extract_frames_parallel(
    extractions: list[FrameExtraction],
    temp_dir: Path,
    num_workers: int,
) -> tuple[int, int]:
    """Extract frames in parallel.

    Args:
        extractions: List of frames to extract
        temp_dir: Directory to store extracted frames
        num_workers: Number of parallel workers

    Returns:
        Tuple of (successful_count, failed_count)
    """
    # Prepare arguments for parallel extraction
    args_list = [
        (
            ext.frame_index,
            ext.file_path,
            ext.seek_offset,
            temp_dir / f"frame_{ext.frame_index:06d}.jpg",
        )
        for ext in extractions
    ]

    successful = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(extract_single_frame, args): args[0] for args in args_list}

        for future in as_completed(futures):
            frame_idx, success, error = future.result()
            if success:
                successful += 1
            else:
                failed += 1
                if failed <= 5:  # Only print first few errors
                    print(f"  Frame {frame_idx} failed: {error[:100]}")

    return successful, failed


def encode_frames_to_video(
    temp_dir: Path,
    output_path: Path,
    fps: float = 30.0,
    use_hw: bool = True,
) -> bool:
    """Encode extracted frames to video.

    Args:
        temp_dir: Directory containing frame_*.jpg files
        output_path: Output video path
        fps: Output framerate
        use_hw: Whether to try hardware encoding

    Returns:
        True if successful
    """
    input_pattern = str(temp_dir / "frame_%06d.jpg")

    if use_hw:
        # Try VAAPI hardware encoding first
        cmd = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-vaapi_device", "/dev/dri/renderD128",
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi",
            "-qp", "23",
            "-y",
            str(output_path),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-y",
            str(output_path),
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            if use_hw:
                print("  Hardware encoding failed, falling back to software...")
                return encode_frames_to_video(temp_dir, output_path, fps, use_hw=False)
            print(f"  Encoding failed: {result.stderr.decode()[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  Encoding timed out")
        return False


def run_prototype(
    files: list[Path],
    output_path: Path,
    target_duration: float,
    num_workers: int,
    output_fps: float = 30.0,
) -> TimingResults:
    """Run the parallel seek prototype.

    Args:
        files: List of input video files
        output_path: Output video path
        target_duration: Target duration in seconds
        num_workers: Number of parallel workers
        output_fps: Output framerate

    Returns:
        TimingResults with timing for each phase
    """
    timing = TimingResults()
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"Parallel Seek Prototype")
    print(f"{'='*60}")
    print(f"Input files: {len(files)}")
    print(f"Target duration: {target_duration}s @ {output_fps}fps")
    print(f"Workers: {num_workers}")
    print()

    # Phase 1: Analyze files
    print("Phase 1: Analyzing files...")
    phase_start = time.time()
    file_infos = analyze_files(files)
    timing.file_analysis = time.time() - phase_start

    total_source = file_infos[-1].start_time + file_infos[-1].duration if file_infos else 0
    print(f"  Total source duration: {total_source:.1f}s ({total_source/3600:.2f}h)")
    print(f"  Time: {timing.file_analysis:.2f}s")
    print()

    # Phase 2: Calculate frame extractions
    print("Phase 2: Calculating frame extractions...")
    phase_start = time.time()
    extractions = calculate_frame_extractions(file_infos, target_duration, output_fps)
    timing.frame_calculation = time.time() - phase_start

    speedup = total_source / target_duration if target_duration > 0 else 0
    unique_files = len(set(ext.file_path for ext in extractions))
    print(f"  Frames to extract: {len(extractions)}")
    print(f"  Unique files needed: {unique_files} of {len(files)}")
    print(f"  Speedup factor: {speedup:.1f}x")
    print(f"  Time: {timing.frame_calculation:.2f}s")
    print()

    # Phase 3: Extract frames in parallel
    print(f"Phase 3: Extracting {len(extractions)} frames with {num_workers} workers...")
    phase_start = time.time()

    with tempfile.TemporaryDirectory(prefix="parallel_seek_") as temp_dir:
        temp_path = Path(temp_dir)
        successful, failed = extract_frames_parallel(extractions, temp_path, num_workers)
        timing.frame_extraction = time.time() - phase_start

        fps_extraction = len(extractions) / timing.frame_extraction if timing.frame_extraction > 0 else 0
        print(f"  Successful: {successful}, Failed: {failed}")
        print(f"  Extraction rate: {fps_extraction:.1f} frames/sec")
        print(f"  Time: {timing.frame_extraction:.2f}s")
        print()

        if failed > len(extractions) * 0.1:  # More than 10% failed
            print("ERROR: Too many extraction failures, aborting")
            return timing

        # Phase 4: Encode to video
        print("Phase 4: Encoding to video...")
        phase_start = time.time()
        success = encode_frames_to_video(temp_path, output_path, output_fps)
        timing.video_encoding = time.time() - phase_start

        if success:
            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"  Output: {output_path}")
            print(f"  Size: {file_size:.1f} MB")
        print(f"  Time: {timing.video_encoding:.2f}s")

    timing.total = time.time() - total_start

    print()
    print(f"{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"  File analysis:     {timing.file_analysis:6.2f}s")
    print(f"  Frame calculation: {timing.frame_calculation:6.2f}s")
    print(f"  Frame extraction:  {timing.frame_extraction:6.2f}s")
    print(f"  Video encoding:    {timing.video_encoding:6.2f}s")
    print(f"  {'─'*30}")
    print(f"  TOTAL:             {timing.total:6.2f}s")
    print()

    return timing


def main():
    parser = argparse.ArgumentParser(description="Parallel seek-based frame extraction prototype")
    parser.add_argument("--files", type=Path, required=True, help="Path to file list (one file per line)")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output video path")
    parser.add_argument("--duration", "-d", type=float, required=True, help="Target duration in seconds")
    parser.add_argument("--workers", "-w", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--fps", type=float, default=30.0, help="Output framerate")

    args = parser.parse_args()

    # Parse file list
    files = parse_file_list(args.files)
    if not files:
        print(f"Error: No files found in {args.files}")
        return 1

    # Run prototype
    timing = run_prototype(
        files=files,
        output_path=args.output,
        target_duration=args.duration,
        num_workers=args.workers,
        output_fps=args.fps,
    )

    return 0


if __name__ == "__main__":
    exit(main())
