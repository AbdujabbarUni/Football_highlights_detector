"""
Clip extraction module - Extract video clips around detected peaks
"""

import cv2
import numpy as np
import subprocess
import os
import logging
from pathlib import Path
from config.config import CLIP_DURATION, CLIP_PRE_DURATION, CLIP_POST_DURATION
from .utils import seconds_to_time

logger = logging.getLogger(__name__)


class ClipExtractor:
    """
    Extracts video clips around detected peak timestamps
    """
    
    def __init__(self, clip_duration=CLIP_DURATION, pre_duration=CLIP_PRE_DURATION,
                 post_duration=CLIP_POST_DURATION):
        """
        Initialize ClipExtractor
        
        Args:
            clip_duration: Total duration of clip (seconds)
            pre_duration: Duration before peak (seconds)
            post_duration: Duration after peak (seconds)
        """
        self.clip_duration = clip_duration
        self.pre_duration = pre_duration
        self.post_duration = post_duration
        logger.info(f"ClipExtractor initialized: {clip_duration}s clips "
                   f"({pre_duration}s before, {post_duration}s after peak)")
    
    def get_clip_bounds(self, peak_timestamp, video_duration):
        """
        Calculate start and end times for clip around peak
        
        Args:
            peak_timestamp: Peak timestamp in seconds
            video_duration: Total video duration in seconds
            
        Returns:
            Tuple of (start_time, end_time) in seconds
        """
        start_time = peak_timestamp - self.pre_duration
        end_time = peak_timestamp + self.post_duration
        
        # Clamp to video boundaries
        start_time = max(0, start_time)
        end_time = min(video_duration, end_time)
        
        return start_time, end_time
    
    def extract_clip_ffmpeg(self, video_path, output_path, start_time, end_time):
        """
        Extract clip using FFmpeg (fast, with AUDIO included)
        
        Args:
            video_path: Path to input video
            output_path: Path for output clip
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            duration = end_time - start_time
            
            # Extract with original audio
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c:v', 'libx264',      # Video codec
                '-preset', 'veryfast',  # Fast encoding
                '-c:a', 'aac',          # AUDIO: AAC codec
                '-b:a', '192k',         # AUDIO: 192kbps bitrate (high quality)
                '-n',                   # Don't overwrite
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info(f"Clip extracted WITH AUDIO: {output_path} ({start_time:.2f}s-{end_time:.2f}s)")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Clip extraction timed out: {output_path}")
            return False
        except Exception as e:
            logger.error(f"Error extracting clip: {e}")
            return False
    
    def extract_clip_opencv(self, video_path, output_path, start_time, end_time):
        """
        Extract clip using OpenCV (with audio using ffmpeg)
        
        Args:
            video_path: Path to input video
            output_path: Path for output clip
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use FFmpeg for audio+video extraction
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(start_time),
                '-to', str(end_time),
                '-c:v', 'copy',         # Copy video stream (fast)
                '-c:a', 'aac',          # AUDIO: AAC codec
                '-b:a', '192k',         # AUDIO: 192kbps bitrate
                '-n',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info(f"Clip extracted WITH AUDIO (FFmpeg): {output_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
            
        except Exception as e:
            logger.error(f"Error extracting clip: {e}")
            return False
    
    def extract_multiple_clips(self, video_path, peak_timestamps, output_dir):
        """
        Extract multiple clips from detected peaks with AUDIO
        
        Args:
            video_path: Path to input video
            peak_timestamps: List of peak timestamps (seconds)
            output_dir: Directory for output clips
            
        Returns:
            List of successfully extracted clip paths
        """
        try:
            # Get video duration
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = total_frames / fps
            cap.release()
            
            logger.info(f"Video duration: {video_duration:.2f}s, FPS: {fps}")
            logger.info(f"Extracting clips WITH ORIGINAL AUDIO")
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            extracted_clips = []
            
            for i, timestamp in enumerate(peak_timestamps):
                start_time, end_time = self.get_clip_bounds(timestamp, video_duration)
                
                output_path = os.path.join(output_dir, f"clip_{i:03d}.mp4")
                
                # Use FFmpeg to preserve audio
                success = self.extract_clip_ffmpeg(video_path, output_path, start_time, end_time)
                
                if success and os.path.exists(output_path):
                    extracted_clips.append(output_path)
                else:
                    logger.warning(f"Failed to extract clip {i}")
            
            logger.info(f"Successfully extracted {len(extracted_clips)}/{len(peak_timestamps)} clips WITH AUDIO")
            return extracted_clips
            
        except Exception as e:
            logger.error(f"Error extracting multiple clips: {e}")
            return []
    
    def get_clip_info(self, clip_path):
        """
        Get information about a video clip
        
        Args:
            clip_path: Path to clip file
            
        Returns:
            Dictionary with clip information
        """
        try:
            cap = cv2.VideoCapture(clip_path)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            file_size = os.path.getsize(clip_path)
            
            return {
                'path': clip_path,
                'fps': fps,
                'duration': duration,
                'width': width,
                'height': height,
                'frames': frame_count,
                'file_size': file_size
            }
        except Exception as e:
            logger.error(f"Error getting clip info: {e}")
            return None
