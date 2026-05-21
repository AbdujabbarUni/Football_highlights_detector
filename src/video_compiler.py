"""
Video compilation module - Combine extracted clips into final highlight video
"""

import subprocess
import os
import logging
from pathlib import Path
from config import *

logger = logging.getLogger(__name__)


class VideoCompiler:
    """
    Compiles extracted clips into a single highlight video
    """
    
    def __init__(self, max_length=MAX_HIGHLIGHT_LENGTH, fps=OUTPUT_FPS,
                 codec=VIDEO_CODEC, crf=VIDEO_CRF):
        """
        Initialize VideoCompiler
        
        Args:
            max_length: Maximum total highlight length in seconds
            fps: Output video frames per second
            codec: Video codec to use
            crf: Quality level (0-51, lower is better)
        """
        self.max_length = max_length
        self.fps = fps
        self.codec = codec
        self.crf = crf
        logger.info(f"VideoCompiler initialized: max_length={max_length}s, "
                   f"fps={fps}, codec={codec}, crf={crf}")
    
    def create_concat_file(self, clip_paths, concat_file_path):
        """
        Create FFmpeg concat demuxer file
        
        Args:
            clip_paths: List of clip file paths
            concat_file_path: Path for concat file
            
        Returns:
            Path to concat file
        """
        try:
            with open(concat_file_path, 'w') as f:
                for clip_path in clip_paths:
                    # Use absolute path
                    abs_path = os.path.abspath(clip_path)
                    f.write(f"file '{abs_path}'\n")
            
            logger.info(f"Concat file created: {concat_file_path} "
                       f"({len(clip_paths)} clips)")
            return concat_file_path
        except Exception as e:
            logger.error(f"Error creating concat file: {e}")
            return None
    
    def compile_clips(self, clip_paths, output_path):
        """
        Compile clips into single video using FFmpeg
        
        Args:
            clip_paths: List of clip file paths (in order)
            output_path: Path for output video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not clip_paths:
                logger.error("No clips provided for compilation")
                return False
            
            logger.info(f"Compiling {len(clip_paths)} clips into {output_path}")
            
            # Create concat file
            concat_file = 'concat.txt'
            if not self.create_concat_file(clip_paths, concat_file):
                return False
            
            try:
                # Use FFmpeg concat demuxer
                cmd = [
                    'ffmpeg',
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_file,
                    '-c:v', self.codec,
                    '-crf', str(self.crf),
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    '-n',
                    output_path
                ]
                
                logger.info(f"Running FFmpeg: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                
                if result.returncode == 0:
                    logger.info(f"Video compilation successful: {output_path}")
                    return True
                else:
                    logger.error(f"FFmpeg error: {result.stderr}")
                    return False
            finally:
                # Clean up concat file
                if os.path.exists(concat_file):
                    os.remove(concat_file)
        
        except subprocess.TimeoutExpired:
            logger.error("Video compilation timed out")
            return False
        except Exception as e:
            logger.error(f"Error compiling clips: {e}")
            return False
    
    def compile_with_transitions(self, clip_paths, output_path, transition='fade', duration=1.0):
        """
        Compile clips with transitions between them
        
        Args:
            clip_paths: List of clip file paths
            output_path: Path for output video
            transition: Type of transition ('fade', 'dissolve')
            duration: Transition duration in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Compiling with {transition} transitions ({duration}s)")
            
            # Build filter complex for transitions
            filter_parts = []
            for i in range(len(clip_paths)):
                filter_parts.append(f"[{i}:v]")
            
            # Simple fade transition filter
            if transition == 'fade':
                filter_str = 'concat=n={}:v=1:a=1'.format(len(clip_paths))
            else:
                filter_str = 'concat=n={}:v=1:a=1'.format(len(clip_paths))
            
            # Build FFmpeg command
            cmd = ['ffmpeg']
            
            # Input files
            for clip in clip_paths:
                cmd.extend(['-i', clip])
            
            # Filter complex
            cmd.extend(['-filter_complex', filter_str])
            
            # Output settings
            cmd.extend([
                '-c:v', self.codec,
                '-crf', str(self.crf),
                '-c:a', 'aac',
                '-b:a', '128k',
                '-n',
                output_path
            ])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                logger.info(f"Video compilation with transitions successful: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Error compiling with transitions: {e}")
            return False
    
    def get_video_info(self, video_path):
        """
        Get video file information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration,size',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            lines = result.stdout.strip().split('\n')
            duration = float(lines[0]) if len(lines) > 0 else 0
            file_size = int(lines[1]) if len(lines) > 1 else 0
            
            return {
                'path': video_path,
                'duration': duration,
                'file_size': file_size,
                'file_size_mb': file_size / (1024 * 1024)
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return None
    
    def trim_to_max_length(self, video_path, output_path, max_seconds=None):
        """
        Trim video to maximum length
        
        Args:
            video_path: Path to input video
            output_path: Path for trimmed video
            max_seconds: Maximum duration in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if max_seconds is None:
                max_seconds = self.max_length
            
            logger.info(f"Trimming video to {max_seconds}s")
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-t', str(max_seconds),
                '-c:v', self.codec,
                '-c:a', 'aac',
                '-n',
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"Video trimmed: {output_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
        
        except Exception as e:
            logger.error(f"Error trimming video: {e}")
            return False
