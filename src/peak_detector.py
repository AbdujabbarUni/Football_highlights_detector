"""
Peak detection module - Identify sound intensity peaks in audio
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import logging
from config.config import (
    SOUND_THRESHOLD, MIN_PEAK_DISTANCE, HOP_LENGTH, SR, SMOOTHING_WINDOW,
    SELECT_TOP_CLIPS, MAX_CLIPS, MIN_CLIP_PROMINENCE
)
from .utils import normalize_audio, smooth_signal

logger = logging.getLogger(__name__)


class PeakDetector:
    """
    Detects peaks in audio energy signal representing highlight moments
    """
    
    def __init__(self, threshold=SOUND_THRESHOLD, min_distance=MIN_PEAK_DISTANCE,
                 hop_length=HOP_LENGTH, sr=SR):
        """
        Initialize PeakDetector
        
        Args:
            threshold: Relative height threshold for peak detection (0.0-1.0)
            min_distance: Minimum distance between peaks in seconds
            hop_length: Samples between frames
            sr: Sample rate
        """
        self.threshold = threshold
        self.min_distance = min_distance
        self.hop_length = hop_length
        self.sr = sr
        # Convert min_distance from seconds to frames
        self.min_distance_frames = int(min_distance * sr / hop_length)
        logger.info(f"PeakDetector initialized: threshold={threshold}, "
                   f"min_distance={min_distance}s ({self.min_distance_frames} frames)")
    
    def detect_peaks(self, energy, timestamps):
        """
        Detect peaks in energy signal
        
        Args:
            energy: Energy signal (numpy array)
            timestamps: Time values for each frame (numpy array)
            
        Returns:
            List of peak timestamps (seconds)
        """
        try:
            # Normalize energy to 0-1 range
            energy_normalized = normalize_audio(energy, method='minmax')
            
            # Apply smoothing to reduce noise
            smoothing_frames = int(SMOOTHING_WINDOW * self.sr / self.hop_length)
            energy_smoothed = smooth_signal(energy_normalized, window_size=smoothing_frames)
            
            # Calculate threshold value
            energy_max = np.max(energy_smoothed)
            height_threshold = energy_max * self.threshold
            
            logger.info(f"Energy range: {np.min(energy_smoothed):.4f} - {energy_max:.4f}")
            logger.info(f"Peak height threshold: {height_threshold:.4f}")
            
            # Find peaks
            peaks, properties = find_peaks(
                energy_smoothed,
                height=height_threshold,
                distance=self.min_distance_frames,
                prominence=MIN_CLIP_PROMINENCE
            )
            
            logger.info(f"Detected {len(peaks)} initial peaks")
            
            # Filter to top peaks if too many
            if SELECT_TOP_CLIPS and len(peaks) > MAX_CLIPS:
                peaks = self._select_top_peaks(energy_smoothed, peaks, MAX_CLIPS)
                logger.info(f"Filtered to top {len(peaks)} peaks by prominence")
            
            # Convert peak indices to timestamps
            peak_timestamps = timestamps[peaks].tolist()
            
            # Sort by timestamp
            peak_timestamps.sort()
            
            return peak_timestamps, peaks, energy_smoothed
            
        except Exception as e:
            logger.error(f"Error detecting peaks: {e}")
            return [], np.array([]), energy
    
    def _select_top_peaks(self, energy, peaks, num_peaks):
        """
        Select only the top N peaks by prominence
        
        Args:
            energy: Energy signal
            peaks: Peak indices
            num_peaks: Number of peaks to select
            
        Returns:
            Filtered peak indices
        """
        try:
            # Calculate prominence for each peak
            prominences = signal.peak_prominences(energy, peaks)[0]
            
            # Sort by prominence and select top N
            top_indices = np.argsort(prominences)[-num_peaks:]
            top_peaks = peaks[np.sort(top_indices)]
            
            logger.info(f"Selected top {num_peaks} peaks by prominence")
            return top_peaks
            
        except Exception as e:
            logger.error(f"Error selecting top peaks: {e}")
            return peaks[:num_peaks]
    
    def adaptive_threshold_detection(self, energy, timestamps):
        """
        Advanced peak detection using adaptive thresholding
        
        Args:
            energy: Energy signal (numpy array)
            timestamps: Time values for each frame (numpy array)
            
        Returns:
            List of peak timestamps (seconds)
        """
        try:
            # Normalize energy
            energy_normalized = normalize_audio(energy, method='zscore')
            
            # Calculate mean and std for adaptive threshold
            mean_energy = np.mean(energy_normalized)
            std_energy = np.std(energy_normalized)
            
            # Adaptive threshold = mean + k*std (k determines sensitivity)
            k = 2.0  # Very high threshold for only major events
            adaptive_threshold = mean_energy + k * std_energy
            
            logger.info(f"Adaptive threshold: mean={mean_energy:.4f}, std={std_energy:.4f}, threshold={adaptive_threshold:.4f}")
            
            # Find peaks using adaptive threshold
            peaks, _ = find_peaks(
                energy_normalized,
                height=adaptive_threshold,
                distance=self.min_distance_frames
            )
            
            # Filter to top peaks
            if SELECT_TOP_CLIPS and len(peaks) > MAX_CLIPS:
                peaks = self._select_top_peaks(energy_normalized, peaks, MAX_CLIPS)
            
            logger.info(f"Adaptive detection found {len(peaks)} peaks")
            
            peak_timestamps = timestamps[peaks].tolist()
            peak_timestamps.sort()
            
            return peak_timestamps, peaks, energy_normalized
            
        except Exception as e:
            logger.error(f"Error in adaptive detection: {e}")
            return [], np.array([]), energy
    
    def filter_peaks_by_prominence(self, energy, peaks, min_prominence=0.15):
        """
        Filter peaks by prominence to keep only significant ones
        
        Args:
            energy: Energy signal
            peaks: Peak indices
            min_prominence: Minimum prominence threshold
            
        Returns:
            Filtered peak indices
        """
        try:
            # Calculate prominence for each peak
            prominences = signal.peak_prominences(energy, peaks)[0]
            
            # Filter peaks
            filtered_peaks = peaks[prominences > min_prominence]
            
            logger.info(f"Filtered peaks by prominence: {len(peaks)} -> {len(filtered_peaks)}")
            
            return filtered_peaks
            
        except Exception as e:
            logger.error(f"Error filtering peaks: {e}")
            return peaks
    
    def get_peak_info(self, energy, peaks, timestamps):
        """
        Get detailed information about detected peaks
        
        Args:
            energy: Energy signal
            peaks: Peak indices
            timestamps: Time values
            
        Returns:
            List of peak information dictionaries
        """
        peak_info = []
        
        for peak in peaks:
            info = {
                'index': int(peak),
                'timestamp': float(timestamps[peak]),
                'energy': float(energy[peak]),
                'time_formatted': self._format_time(timestamps[peak])
            }
            peak_info.append(info)
        
        return peak_info
    
    @staticmethod
    def _format_time(seconds):
        """
        Format seconds to MM:SS format
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes:02d}:{secs:02d}"
