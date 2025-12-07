# Parallel Seek Prototype Results

## Test Configuration
- **Source**: 24 hours of bporchcam recordings (8,640 files)
- **Target**: 15 second timelapse @ 30fps (450 frames)
- **Speedup**: 5760x

## Results by Worker Count

| Workers | Extraction Rate | Extraction Time | Encode Time | Total Time |
|---------|-----------------|-----------------|-------------|------------|
| 8       | 16.5 fps        | 27.32s          | 5.67s       | **33.02s** |
| 16      | 20.8 fps        | 21.59s          | 6.26s       | **27.89s** |
| 32      | 24.1 fps        | 18.70s          | 5.79s       | **24.53s** |

## Key Findings

1. **Parallel extraction works**: Successfully extracted 450 frames from 8,640 files
2. **Scaling**: ~26% speedup going from 8 to 32 workers
3. **Bottleneck**: Frame extraction is the main bottleneck (~75% of time)
4. **Output quality**: Full resolution (2688x1520), hardware encoded

## Comparison with Current Implementation

Current implementation for 24h footage:
- Concatenates ALL 8,640 files first (~35GB temp file)
- Then processes every frame with setpts filter
- Estimated time: **10-15+ minutes** (killed due to timeout)

Parallel seek prototype:
- Only opens 450 files (ones with needed frames)
- Direct frame extraction with fast seek
- Total time: **~25 seconds** (32 workers)

## Recommendation

The parallel seek approach is **~20-30x faster** for high speedup timelapses.
Optimal worker count appears to be 16-32 for this hardware.
