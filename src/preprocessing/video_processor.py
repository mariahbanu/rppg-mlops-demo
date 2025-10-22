"""
Video processing for rPPG signal extraction
Processes actual video files to extract RGB signals
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Extract RGB signals from facial videos for rPPG analysis
    
    This class handles:
    - Face detection using MediaPipe
    - ROI (Region of Interest) extraction
    - Frame-by-frame RGB signal capture
    - Signal quality validation
    """
    
    def __init__(self, target_fps: int = 30, min_face_confidence: float = 0.5):
        """
        Args:
            target_fps: Target frame rate for processing
            min_face_confidence: Minimum confidence for face detection
        """
        self.target_fps = target_fps
        self.min_face_confidence = min_face_confidence
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for short range (better for selfie/portrait videos)
            min_detection_confidence=min_face_confidence
        )
        
        logger.info(f"VideoProcessor initialized (target_fps={target_fps})")
    
    def process_video(self, video_path: str, max_duration: int = 30) -> Optional[dict]:
        """
        Process video file and extract RGB signal
        
        Args:
            video_path: Path to video file
            max_duration: Maximum duration to process (seconds)
        
        Returns:
            Dictionary with:
                - rgb_signal: (T, 3) array of mean RGB values
                - fps: Actual video FPS
                - frames_processed: Number of frames
                - face_detection_rate: Percentage of frames with face detected
                - duration: Actual duration processed
        """
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height}, {original_fps} FPS, {total_frames} frames")
        
        # Calculate max frames to process
        max_frames = int(min(max_duration * original_fps, total_frames))
        
        rgb_signals = []
        faces_detected = 0
        frames_processed = 0
        
        logger.info(f"Processing up to {max_frames} frames...")
        
        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames_processed += 1
            
            # Convert BGR to RGB (OpenCV uses BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            results = self.face_detection.process(frame_rgb)
            
            if results.detections:
                faces_detected += 1
                
                # Get first detection (assumes single person)
                detection = results.detections[0]
                
                # Extract bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                
                # Extract face region
                face_roi = frame_rgb[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    # Calculate mean RGB values
                    mean_rgb = face_roi.mean(axis=(0, 1))
                    rgb_signals.append(mean_rgb)
            
            # Progress update every 100 frames
            if frames_processed % 100 == 0:
                logger.info(f"Processed {frames_processed}/{max_frames} frames "
                          f"({faces_detected/frames_processed*100:.1f}% face detection)")
        
        cap.release()
        
        if len(rgb_signals) == 0:
            logger.error("No valid frames extracted (no faces detected)")
            return None
        
        # Convert to numpy array
        rgb_signals = np.array(rgb_signals)
        
        # Calculate statistics
        face_detection_rate = faces_detected / frames_processed
        duration = frames_processed / original_fps
        
        logger.info(f"✅ Processing complete!")
        logger.info(f"   Frames processed: {frames_processed}")
        logger.info(f"   Face detection rate: {face_detection_rate:.1%}")
        logger.info(f"   Signal length: {len(rgb_signals)} frames")
        logger.info(f"   Duration: {duration:.1f} seconds")
        
        # Quality check
        if face_detection_rate < 0.6:
            logger.warning(f"⚠️ Low face detection rate: {face_detection_rate:.1%}")
            logger.warning("   Video quality may be poor or face not clearly visible")
        
        return {
            'rgb_signal': rgb_signals,
            'fps': original_fps,
            'frames_processed': frames_processed,
            'face_detection_rate': face_detection_rate,
            'duration': duration,
            'original_fps': original_fps,
            'video_path': video_path
        }
    
    def process_video_to_model_input(self, video_path: str) -> Optional[np.ndarray]:
        """
        Process video and prepare input for model (900 frames)
        
        Args:
            video_path: Path to video file
        
        Returns:
            (900, 3) numpy array ready for model input, or None if processing fails
        """
        result = self.process_video(video_path, max_duration=30)
        
        if result is None:
            return None
        
        signal = result['rgb_signal']
        
        # Resample to exactly 900 frames (30 seconds @ 30 FPS)
        if len(signal) != 900:
            from scipy.interpolate import interp1d
            
            original_length = len(signal)
            original_indices = np.arange(original_length)
            target_indices = np.linspace(0, original_length - 1, 900)
            
            # Interpolate each channel
            resampled = np.zeros((900, 3))
            for channel in range(3):
                interp_func = interp1d(original_indices, signal[:, channel], kind='cubic')
                resampled[:, channel] = interp_func(target_indices)
            
            signal = resampled
            logger.info(f"Resampled signal from {original_length} to 900 frames")
        
        # Normalize to [0, 1] range
        signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
        
        return signal
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


def process_video_file(video_path: str, output_path: Optional[str] = None) -> dict:
    """
    Convenience function to process a video file
    
    Args:
        video_path: Path to video file
        output_path: Optional path to save processed signal
    
    Returns:
        Processing results dictionary
    """
    processor = VideoProcessor()
    result = processor.process_video(video_path)
    
    if result and output_path:
        np.save(output_path, result['rgb_signal'])
        logger.info(f"Saved processed signal to: {output_path}")
    
    return result


# Test script
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_processor.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    result = process_video_file(video_path)
    
    if result:
        print("\n" + "="*60)
        print("VIDEO PROCESSING RESULTS")
        print("="*60)
        print(f"Video: {video_path}")
        print(f"Duration: {result['duration']:.1f} seconds")
        print(f"FPS: {result['fps']:.1f}")
        print(f"Frames processed: {result['frames_processed']}")
        print(f"Face detection rate: {result['face_detection_rate']:.1%}")
        print(f"RGB signal shape: {result['rgb_signal'].shape}")
        print("="*60)
