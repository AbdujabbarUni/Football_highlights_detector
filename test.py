#!/usr/bin/env python3
"""
Testing and demonstration script for Football Highlights Detector
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.audio_processor import AudioProcessor
from src.peak_detector import PeakDetector
from src.utils import setup_logging
from config import *


def test_audio_processor():
    """
    Test audio processor with synthetic audio
    """
    print("\n=== Testing Audio Processor ===")
    
    processor = AudioProcessor(sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Create synthetic audio signal
    duration = 10  # 10 seconds
    t = np.arange(0, duration, 1/SR)
    
    # Base signal (quiet)
    audio = np.sin(2 * np.pi * 440 * t) * 0.1
    
    # Add peaks (loud cheering)
    peaks_at = [2.0, 4.5, 7.3]  # seconds
    for peak_time in peaks_at:
        peak_idx = int(peak_time * SR)
        window_size = int(0.5 * SR)  # 0.5 second peaks
        for i in range(window_size):
            if peak_idx + i < len(audio):
                audio[peak_idx + i] += np.sin(2 * np.pi * 1000 * t[peak_idx + i]) * 0.8
    
    # Compute energy
    energy = processor.compute_energy(audio, use_stft=True)
    print(f"Energy computed: {len(energy)} frames")
    print(f"Energy range: {np.min(energy):.4f} - {np.max(energy):.4f} dB")
    
    return True


def test_peak_detector():
    """
    Test peak detector with synthetic energy signal
    """
    print("\n=== Testing Peak Detector ===")
    
    detector = PeakDetector(
        threshold=SOUND_THRESHOLD,
        min_distance=MIN_PEAK_DISTANCE,
        hop_length=HOP_LENGTH,
        sr=SR
    )
    
    # Create synthetic energy signal
    n_frames = 1000
    timestamps = np.arange(n_frames) * HOP_LENGTH / SR
    
    # Base signal
    energy = np.random.normal(-20, 5, n_frames)
    
    # Add peaks
    peaks_at = [100, 300, 650, 900]  # frame indices
    for peak_idx in peaks_at:
        energy[peak_idx] = 20  # High value
        # Create peak envelope
        for i in range(-20, 21):
            if 0 <= peak_idx + i < len(energy):
                energy[peak_idx + i] += 10 * np.exp(-(i**2) / 50)
    
    # Detect peaks
    peak_timestamps, peaks, _ = detector.detect_peaks(energy, timestamps)
    
    print(f"Detected {len(peak_timestamps)} peaks:")
    for i, ts in enumerate(peak_timestamps):
        print(f"  Peak {i+1}: {ts:.2f}s")
    
    return len(peak_timestamps) > 0


def test_configuration():
    """
    Test and display configuration
    """
    print("\n=== Configuration Settings ===")
    
    config_items = [
        ('Audio Sample Rate', SR, 'Hz'),
        ('FFT Window Size', N_FFT, 'samples'),
        ('Hop Length', HOP_LENGTH, 'samples'),
        ('Sound Threshold', SOUND_THRESHOLD, '(0.0-1.0)'),
        ('Min Peak Distance', MIN_PEAK_DISTANCE, 'seconds'),
        ('Clip Duration', CLIP_DURATION, 'seconds'),
        ('Pre-peak Duration', CLIP_PRE_DURATION, 'seconds'),
        ('Post-peak Duration', CLIP_POST_DURATION, 'seconds'),
        ('Max Highlight Length', MAX_HIGHLIGHT_LENGTH, 'seconds'),
        ('Output FPS', OUTPUT_FPS, 'fps'),
        ('Video Codec', VIDEO_CODEC, ''),
    ]
    
    for name, value, unit in config_items:
        print(f"  {name:.<30} {value} {unit}")
    
    return True


def test_dependencies():
    """
    Test if all required dependencies are installed
    """
    print("\n=== Checking Dependencies ===")
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('cv2', 'OpenCV'),
        ('librosa', 'Librosa'),
    ]
    
    all_ok = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {name} installed")
        except ImportError:
            print(f"  ✗ {name} NOT installed")
            all_ok = False
    
    # Check for FFmpeg
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        print(f"  ✓ FFmpeg installed")
    except:
        print(f"  ✗ FFmpeg NOT installed")
        all_ok = False
    
    return all_ok


def main():
    """
    Run all tests
    """
    print("\n" + "="*50)
    print("Football Highlights Detector - Test Suite")
    print("="*50)
    
    setup_logging(verbose=True)
    
    tests = [
        ('Dependencies', test_dependencies),
        ('Configuration', test_configuration),
        ('Audio Processor', test_audio_processor),
        ('Peak Detector', test_peak_detector),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, 'PASS' if result else 'FAIL'))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((test_name, 'ERROR'))
    
    print("\n" + "="*50)
    print("Test Results")
    print("="*50)
    
    for test_name, result in results:
        status_symbol = '✓' if result == 'PASS' else '✗'
        print(f"  {status_symbol} {test_name}: {result}")
    
    print("\n" + "="*50)
    print("To run on your video:")
    print("  python main.py --input match_video.mp4 --output highlights.mp4")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()
