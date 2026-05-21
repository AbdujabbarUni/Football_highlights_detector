"""
Utility functions for Football Highlights Detector
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(verbose=True):
    """
    Setup logging configuration
    
    Args:
        verbose: If True, use DEBUG level, else INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def ensure_output_dir(output_dir):
    """
    Ensure output directory exists
    
    Args:
        output_dir: Path to output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ready: {output_dir}")


def seconds_to_time(seconds):
    """
    Convert seconds to HH:MM:SS format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    return str(timedelta(seconds=int(seconds)))


def save_timestamps(timestamps, output_file):
    """
    Save detected peak timestamps to JSON file
    
    Args:
        timestamps: List of peak timestamps (seconds)
        output_file: Path to output JSON file
    """
    data = {
        'peaks_detected': len(timestamps),
        'timestamps': [{
            'second': float(ts),
            'time': seconds_to_time(ts)
        } for ts in timestamps]
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Timestamps saved: {output_file}")


def load_timestamps(input_file):
    """
    Load timestamps from JSON file
    
    Args:
        input_file: Path to JSON file
        
    Returns:
        List of timestamps (seconds)
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    return [item['second'] for item in data['timestamps']]


def normalize_audio(audio, method='minmax'):
    """
    Normalize audio signal
    
    Args:
        audio: Audio signal (numpy array)
        method: Normalization method ('minmax', 'zscore', 'none')
        
    Returns:
        Normalized audio signal
    """
    if method == 'minmax':
        min_val = np.min(audio)
        max_val = np.max(audio)
        if max_val - min_val == 0:
            return audio
        return (audio - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        mean = np.mean(audio)
        std = np.std(audio)
        if std == 0:
            return audio
        return (audio - mean) / std
    
    else:  # 'none'
        return audio


def smooth_signal(signal, window_size=5):
    """
    Apply smoothing to signal using moving average
    
    Args:
        signal: Input signal (numpy array)
        window_size: Size of moving average window
        
    Returns:
        Smoothed signal
    """
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(signal, kernel, mode='same')
    return smoothed


def get_video_duration(video_path):
    """
    Get video duration in seconds
    
    Args:
        video_path: Path to video file
        
    Returns:
        Duration in seconds
    """
    import subprocess
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1:noprint_wrappers=1',
             video_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error getting video duration: {e}")
        return None


def get_video_fps(video_path):
    """
    Get video frames per second
    
    Args:
        video_path: Path to video file
        
    Returns:
        FPS value
    """
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    except Exception as e:
        logger.error(f"Error getting FPS: {e}")
        return 30


def get_video_resolution(video_path):
    """
    Get video resolution (width, height)
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (width, height)
    """
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height
    except Exception as e:
        logger.error(f"Error getting resolution: {e}")
        return (1920, 1080)


def validate_input_file(file_path):
    """
    Validate that input video file exists and is readable
    
    Args:
        file_path: Path to video file
        
    Returns:
        True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    if not os.path.isfile(file_path):
        logger.error(f"Not a file: {file_path}")
        return False
    
    return True


def format_file_size(bytes_size):
    """
    Format bytes to human-readable size
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"
