"""
Audio processing module - Extract and analyze audio from video
"""

import os
import subprocess
import numpy as np
import librosa
import logging
from config import *

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handles audio extraction and analysis from video files
    """
    
    def __init__(self, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH):
        """
        Initialize AudioProcessor
        
        Args:
            sr: Sample rate (Hz)
            n_fft: FFT window size
            hop_length: Samples between frames
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        logger.info(f"AudioProcessor initialized: SR={sr}, N_FFT={n_fft}")
    
    def extract_audio(self, video_path, audio_output_path, audio_format='wav'):
        """
        Extract audio track from video file using FFmpeg
        
        Args:
            video_path: Path to input video
            audio_output_path: Path for output audio file
            audio_format: Audio format ('wav', 'mp3', 'aac')
            
        Returns:
            Path to extracted audio file or None if failed
        """
        try:
            logger.info(f"Extracting audio from: {video_path}")
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-q:a', '9',  # Audio quality
                '-n',  # Don't overwrite
                audio_output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info(f"Audio extracted successfully: {audio_output_path}")
                return audio_output_path
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out")
            return None
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return None
    
    def load_audio(self, audio_path):
        """
        Load audio file using librosa
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            logger.info(f"Loading audio: {audio_path}")
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            logger.info(f"Audio loaded: {len(y)} samples at {sr} Hz")
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            return None, None
    
    def compute_energy(self, audio, use_stft=True):
        """
        Compute energy/intensity of audio signal
        
        Args:
            audio: Audio signal (numpy array)
            use_stft: If True, use STFT; else use simple RMS
            
        Returns:
            Energy values (numpy array)
        """
        try:
            if use_stft:
                logger.info("Computing STFT energy...")
                # Compute STFT
                S = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
                # Compute magnitude
                magnitude = np.abs(S)
                # Compute energy (sum of squared magnitude)
                energy = np.sum(magnitude ** 2, axis=0)
                # Convert to dB scale
                energy_db = librosa.power_to_db(energy, ref=np.max)
            else:
                logger.info("Computing RMS energy...")
                # Compute RMS energy
                energy = librosa.feature.rms(
                    y=audio,
                    frame_length=self.n_fft,
                    hop_length=self.hop_length
                )[0]
                # Convert to dB scale
                energy_db = librosa.power_to_db(energy ** 2, ref=np.max)
            
            logger.info(f"Energy computed: {len(energy_db)} frames")
            return energy_db
            
        except Exception as e:
            logger.error(f"Error computing energy: {e}")
            return None
    
    def get_timestamps(self, energy, sr=None):
        """
        Convert frame indices to time timestamps (seconds)
        
        Args:
            energy: Energy array (frames)
            sr: Sample rate
            
        Returns:
            Array of timestamps (seconds)
        """
        if sr is None:
            sr = self.sr
        
        # Calculate time for each frame
        times = librosa.frames_to_time(
            np.arange(len(energy)),
            sr=sr,
            hop_length=self.hop_length
        )
        return times
    
    def process_audio_file(self, video_path, temp_audio_path='temp_audio.wav'):
        """
        Complete audio processing pipeline
        
        Args:
            video_path: Path to input video
            temp_audio_path: Path for temporary audio file
            
        Returns:
            Tuple of (energy, timestamps) or (None, None) if failed
        """
        try:
            # Extract audio
            audio_path = self.extract_audio(video_path, temp_audio_path)
            if audio_path is None:
                return None, None
            
            # Load audio
            audio, sr = self.load_audio(audio_path)
            if audio is None:
                return None, None
            
            # Compute energy
            energy = self.compute_energy(audio, use_stft=True)
            if energy is None:
                return None, None
            
            # Get timestamps
            timestamps = self.get_timestamps(energy, sr=sr)
            
            logger.info(f"Audio processing complete: {len(timestamps)} frames")
            return energy, timestamps
            
        except Exception as e:
            logger.error(f"Error in audio processing pipeline: {e}")
            return None, None
    
    def cleanup_temp_audio(self, temp_audio_path):
        """
        Clean up temporary audio file
        
        Args:
            temp_audio_path: Path to temporary audio file
        """
        try:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                logger.info(f"Temporary audio cleaned up: {temp_audio_path}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp audio: {e}")
