#!/usr/bin/env python3
"""
Football Highlights Detector - Main execution script

Usage:
    python main.py --input video.mp4 --output highlights.mp4
    python main.py --input video.mp4 --output highlights.mp4 --threshold 0.6
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.audio_processor import AudioProcessor
from src.peak_detector import PeakDetector
from src.clip_extractor import ClipExtractor
from src.video_compiler import VideoCompiler
from src.utils import (
    setup_logging, ensure_output_dir, save_timestamps,
    validate_input_file, get_video_duration, format_file_size
)
from config import *


class HighlightsDetector:
    """
    Main class for football highlights detection
    """
    
    def __init__(self, threshold=SOUND_THRESHOLD, verbose=VERBOSE):
        """
        Initialize detector
        
        Args:
            threshold: Sound sensitivity threshold
            verbose: Enable verbose logging
        """
        setup_logging(verbose)
        self.logger = logging.getLogger(__name__)
        
        self.audio_processor = AudioProcessor(sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
        self.peak_detector = PeakDetector(
            threshold=threshold,
            min_distance=MIN_PEAK_DISTANCE,
            hop_length=HOP_LENGTH,
            sr=SR
        )
        self.clip_extractor = ClipExtractor(
            clip_duration=CLIP_DURATION,
            pre_duration=CLIP_PRE_DURATION,
            post_duration=CLIP_POST_DURATION
        )
        self.video_compiler = VideoCompiler(
            max_length=MAX_HIGHLIGHT_LENGTH,
            fps=OUTPUT_FPS,
            codec=VIDEO_CODEC,
            crf=VIDEO_CRF
        )
        
        self.logger.info("HighlightsDetector initialized")
    
    def process(self, input_video, output_video, temp_dir='./temp'):
        """
        Complete pipeline: detect highlights and compile video
        
        Args:
            input_video: Path to input match video
            output_video: Path for output highlight video
            temp_dir: Directory for temporary files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate input
            if not validate_input_file(input_video):
                self.logger.error("Invalid input file")
                return False
            
            # Ensure directories exist
            ensure_output_dir(OUTPUT_DIR)
            ensure_output_dir(temp_dir)
            
            # Get video info
            video_duration = get_video_duration(input_video)
            if video_duration is None:
                self.logger.error("Could not get video duration")
                return False
            self.logger.info(f"Video duration: {video_duration/60:.2f} minutes")
            
            # Step 1: Extract and analyze audio
            self.logger.info("\n=== Step 1: Audio Processing ===")
            temp_audio = os.path.join(temp_dir, 'temp_audio.wav')
            energy, timestamps = self.audio_processor.process_audio_file(
                input_video, temp_audio
            )
            
            if energy is None or timestamps is None:
                self.logger.error("Audio processing failed")
                return False
            
            # Step 2: Detect peaks (highlight moments)
            self.logger.info("\n=== Step 2: Peak Detection ===")
            peak_timestamps, peaks, energy_smoothed = self.peak_detector.detect_peaks(
                energy, timestamps
            )
            
            if not peak_timestamps:
                self.logger.warning("No peaks detected. Try lowering threshold.")
                return False
            
            self.logger.info(f"Found {len(peak_timestamps)} highlight moments")
            
            # Save timestamps
            timestamp_file = os.path.join(OUTPUT_DIR, 'timestamps.json')
            save_timestamps(peak_timestamps, timestamp_file)
            
            # Generate visualization (optional)
            if GENERATE_PLOTS:
                self._generate_plot(energy_smoothed, timestamps, peaks,
                                   os.path.join(OUTPUT_DIR, 'audio_analysis.png'))
            
            # Step 3: Extract clips around each peak
            self.logger.info("\n=== Step 3: Clip Extraction ===")
            clips_dir = os.path.join(temp_dir, 'clips')
            extracted_clips = self.clip_extractor.extract_multiple_clips(
                input_video, peak_timestamps, clips_dir
            )
            
            if not extracted_clips:
                self.logger.error("No clips extracted")
                return False
            
            self.logger.info(f"Extracted {len(extracted_clips)} clips")
            
            # Step 4: Compile clips into final video
            self.logger.info("\n=== Step 4: Video Compilation ===")
            temp_output = os.path.join(temp_dir, 'compiled_full.mp4')
            
            if not self.video_compiler.compile_clips(extracted_clips, temp_output):
                self.logger.error("Video compilation failed")
                return False
            
            # Step 5: Trim to max length
            self.logger.info("\n=== Step 5: Final Processing ===")
            if not self.video_compiler.trim_to_max_length(
                temp_output, output_video, MAX_HIGHLIGHT_LENGTH
            ):
                self.logger.error("Video trimming failed")
                return False
            
            # Get final video info
            final_info = self.video_compiler.get_video_info(output_video)
            if final_info:
                self.logger.info(f"Final video duration: {final_info['duration']:.2f}s")
                self.logger.info(f"Final video size: {format_file_size(final_info['file_size'])}")
            
            # Cleanup temp files
            self._cleanup_temp(temp_dir)
            
            self.logger.info("\n=== PROCESSING COMPLETE ===")
            self.logger.info(f"Highlight video saved: {output_video}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            return False
    
    def _generate_plot(self, energy, timestamps, peaks, output_file):
        """
        Generate visualization of audio analysis
        
        Args:
            energy: Energy signal
            timestamps: Time values
            peaks: Peak indices
            output_file: Path for output image
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(16, 4))
            plt.plot(timestamps, energy, label='Audio Energy')
            plt.plot(timestamps[peaks], energy[peaks], 'r*', markersize=15, label='Detected Peaks')
            plt.xlabel('Time (s)')
            plt.ylabel('Energy (dB)')
            plt.title('Audio Analysis - Highlight Moment Detection')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_file, dpi=100)
            plt.close()
            
            self.logger.info(f"Plot saved: {output_file}")
        except Exception as e:
            self.logger.warning(f"Could not generate plot: {e}")
    
    def _cleanup_temp(self, temp_dir):
        """
        Clean up temporary directory
        
        Args:
            temp_dir: Temporary directory path
        """
        try:
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                self.logger.info(f"Temporary files cleaned up: {temp_dir}")
        except Exception as e:
            self.logger.warning(f"Could not clean up temp: {e}")


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(
        description='Football Highlights Detector - Extract highlights from match videos'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input match video (2-3 hours)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path for output highlight video'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=SOUND_THRESHOLD,
        help=f'Sound sensitivity threshold (0.0-1.0, default: {SOUND_THRESHOLD})'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=MAX_HIGHLIGHT_LENGTH,
        help=f'Maximum highlight video length in seconds (default: {MAX_HIGHLIGHT_LENGTH})'
    )
    
    parser.add_argument(
        '--clip-duration',
        type=int,
        default=CLIP_DURATION,
        help=f'Duration of each clip in seconds (default: {CLIP_DURATION})'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    # Create detector
    detector = HighlightsDetector(
        threshold=args.threshold,
        verbose=args.verbose
    )
    
    # Process video
    success = detector.process(args.input, args.output)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
