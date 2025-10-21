"""
Prepare synthetic dataset for demonstration
Creates realistic rPPG signal data without massive downloads
"""
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_ppg_signal(heart_rate, duration=30, fps=30, noise_level=0.1):
    """
    Generate synthetic photoplethysmography (PPG) signal
    
    Args:
        heart_rate: Heart rate in BPM (e.g., 70)
        duration: Duration in seconds (e.g., 30)
        fps: Frames per second (e.g., 30)
        noise_level: Amount of noise to add (0.1 = 10%)
    
    Returns:
        RGB signal array of shape (num_frames, 3)
    """
    num_frames = duration * fps
    time = np.linspace(0, duration, num_frames)
    
    # Convert BPM to Hz (cycles per second)
    hr_hz = heart_rate / 60.0
    
    # Create base PPG signal (green channel strongest)
    # Real PPG signal has multiple harmonics
    ppg_base = (
        np.sin(2 * np.pi * hr_hz * time) +  # Fundamental frequency
        0.3 * np.sin(4 * np.pi * hr_hz * time) +  # Second harmonic
        0.1 * np.sin(6 * np.pi * hr_hz * time)    # Third harmonic
    )
    
    # Normalize to 0-1 range
    ppg_base = (ppg_base - ppg_base.min()) / (ppg_base.max() - ppg_base.min())
    
    # Create RGB channels (Green has strongest PPG signal in reality)
    rgb_signal = np.zeros((num_frames, 3))
    
    # Red channel: weaker PPG signal
    rgb_signal[:, 0] = 0.5 + 0.15 * ppg_base + noise_level * np.random.randn(num_frames)
    
    # Green channel: strongest PPG signal
    rgb_signal[:, 1] = 0.5 + 0.25 * ppg_base + noise_level * np.random.randn(num_frames)
    
    # Blue channel: weakest PPG signal
    rgb_signal[:, 2] = 0.5 + 0.10 * ppg_base + noise_level * np.random.randn(num_frames)
    
    # Add slow baseline wander (breathing, motion)
    baseline_wander = 0.05 * np.sin(2 * np.pi * 0.2 * time)  # 0.2 Hz breathing
    rgb_signal += baseline_wander[:, np.newaxis]
    
    # Clip to valid range
    rgb_signal = np.clip(rgb_signal, 0, 1)
    
    return rgb_signal

def create_dataset(num_samples=200):
    """
    Create complete synthetic dataset
    
    Args:
        num_samples: Number of samples to generate
    
    Returns:
        DataFrame with all samples
    """
    logger.info(f"Generating {num_samples} synthetic samples...")
    
    samples = []
    
    for i in range(num_samples):
        # Random heart rate in physiological range
        heart_rate = np.random.uniform(50, 120)  # 50-120 BPM
        
        # Random duration (but we'll use 30s for consistency)
        duration = 30
        fps = 30
        
        # Generate signal
        rgb_signal = generate_ppg_signal(
            heart_rate=heart_rate,
            duration=duration,
            fps=fps,
            noise_level=np.random.uniform(0.05, 0.15)  # Random noise level
        )
        
        # Create sample
        sample = {
            'sample_id': f'synthetic_{i:04d}',
            'rgb_signal': rgb_signal,
            'ground_truth_hr': heart_rate,
            'duration': duration,
            'fps': fps,
            'num_frames': len(rgb_signal)
        }
        
        samples.append(sample)
        
        if (i + 1) % 50 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} samples...")
    
    logger.info(f"✅ Generated {num_samples} samples")
    
    return pd.DataFrame(samples)

def split_dataset(df, train_ratio=0.7, val_ratio=0.15):
    """
    Split dataset into train/val/test
    
    Args:
        df: Full dataset
        train_ratio: Proportion for training (0.7 = 70%)
        val_ratio: Proportion for validation (0.15 = 15%)
    
    Returns:
        train_df, val_df, test_df
    """
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]
    
    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df

def main():
    """Main function"""
    # Create data directories
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    logger.info("Creating synthetic rPPG dataset...")
    df = create_dataset(num_samples=200)
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(df)
    
    # Save datasets (use pickle for numpy arrays)
    logger.info("Saving datasets...")
    train_df.to_pickle('data/processed/train.pkl')
    val_df.to_pickle('data/processed/val.pkl')
    test_df.to_pickle('data/processed/test.pkl')
    
    # Save metadata
    metadata = {
        'total_samples': len(df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'mean_hr': df['ground_truth_hr'].mean(),
        'std_hr': df['ground_truth_hr'].std(),
        'min_hr': df['ground_truth_hr'].min(),
        'max_hr': df['ground_truth_hr'].max()
    }
    
    pd.DataFrame([metadata]).to_csv('data/processed/metadata.csv', index=False)
    
    logger.info("✅ Dataset preparation complete!")
    logger.info(f"   Train: {len(train_df)} samples")
    logger.info(f"   Val:   {len(val_df)} samples")
    logger.info(f"   Test:  {len(test_df)} samples")
    logger.info(f"   Mean HR: {metadata['mean_hr']:.1f} ± {metadata['std_hr']:.1f} BPM")

if __name__ == '__main__':
    main()
