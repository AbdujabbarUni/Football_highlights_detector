"""
Configuration parameters for Football Highlights Detector
"""

# ============= AUDIO PROCESSING =============
# FFT parameters
N_FFT = 2048              # FFT window size
HOP_LENGTH = 512          # Samples between frames
SR = 22050                # Sample rate (Hz)

# ============= PEAK DETECTION =============
# Sound intensity threshold (0.0-1.0)
# Lower = more sensitive (detects more events)
# Higher = more specific (detects only loud events)
# OPTIMIZED FOR GOALS: Use 0.90-0.95 for best results
SOUND_THRESHOLD = 0.85

# Minimum distance between peaks (seconds)
MIN_PEAK_DISTANCE = 8.0   # At least 8 seconds apart

# Height of peaks (relative to max)
PEAK_HEIGHT = 0.6

# Select only top clips by prominence
SELECT_TOP_CLIPS = True

# Maximum number of clips to keep
MAX_CLIPS = 30  # 30 clips × 20 seconds = 10 minutes

# Minimum prominence for clip selection
MIN_CLIP_PROMINENCE = 0.2

# ============= CLIP EXTRACTION =============
# Duration of clip around each detected peak (seconds)
CLIP_DURATION = 20        # Extract 20-second clips

# Pre-peak duration (how much before peak)
CLIP_PRE_DURATION = 5     # 5 seconds before

# Post-peak duration (how much after peak)
CLIP_POST_DURATION = 15   # 15 seconds after

# ============= VIDEO COMPILATION =============
# Maximum total highlight length (seconds)
MAX_HIGHLIGHT_LENGTH = 600  # 10 minutes

# Target frame rate for output video
OUTPUT_FPS = 30

# Video codec
VIDEO_CODEC = 'libx264'    # H.264 codec

# Video quality (crf: 0-51, lower = better, 23 = default)
VIDEO_CRF = 18

# ============= PROCESSING =============
# Chunk size for processing large files (seconds)
CHUNK_SIZE = 300          # Process 5-minute chunks

# Whether to use GPU for processing (if available)
USE_GPU = False

# Number of worker threads
NUM_WORKERS = 4

# ============= OUTPUT =============
# Output directory
OUTPUT_DIR = './output'

# Generate visualization plots
GENERATE_PLOTS = True

# Save timestamps of detected events
SAVE_TIMESTAMPS = True

# Verbose logging
VERBOSE = True

# ============= ADVANCED PARAMETERS =============
# Smoothing window for energy curve (seconds)
SMOOTHING_WINDOW = 2.0

# Energy normalization method ('minmax', 'zscore', 'none')
NORMALIZATION_METHOD = 'minmax'

# Additional peak detection sensitivity (0.0-1.0)
SENSITIVITY = 0.7
