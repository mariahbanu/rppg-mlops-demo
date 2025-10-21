"""
Training script with MLflow integration
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.rppg_net import create_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class rPPGDataset(Dataset):
    """PyTorch Dataset for rPPG signals"""
    
    def __init__(self, dataframe):
        """
        Args:
            dataframe: Pandas DataFrame with 'rgb_signal' and 'ground_truth_hr'
        """
        self.signals = []
        self.heart_rates = []
        
        for idx, row in dataframe.iterrows():
            signal = row['rgb_signal']
            hr = row['ground_truth_hr']
            
            self.signals.append(torch.FloatTensor(signal))
            self.heart_rates.append(torch.FloatTensor([hr]))
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        return self.signals[idx], self.heart_rates[idx]


def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    predictions = predictions.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()

    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # Pearson Correlation
    correlation = np.corrcoef(predictions, targets)[0, 1]
    
    # Percentage within 5 BPM
    within_5bpm = np.mean(np.abs(predictions - targets) <= 5) * 100
    
    # Percentage within 10 BPM
    within_10bpm = np.mean(np.abs(predictions - targets) <= 10) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'within_5bpm_percent': within_5bpm,
        'within_10bpm_percent': within_10bpm
    }


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for signals, heart_rates in progress_bar:
        signals = signals.to(device)
        heart_rates = heart_rates.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(signals)
        loss = criterion(predictions, heart_rates)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for signals, heart_rates in tqdm(val_loader, desc='Validation'):
            signals = signals.to(device)
            heart_rates = heart_rates.to(device)
            
            predictions = model(signals)
            loss = criterion(predictions, heart_rates)
            
            total_loss += loss.item()
            all_predictions.append(predictions)
            all_targets.append(heart_rates)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics


def main():
    """Main training function"""
    # Configuration
    config = {
        'batch_size': 16,
        'learning_rate': 0.001,
        'num_epochs': 30,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'sequence_length': 900,
        'num_workers': 4,
        'early_stopping_patience': 10
    }
    
    logger.info("=" * 60)
    logger.info("🚀 Starting Training")
    logger.info("=" * 60)
    logger.info(f"Device: {config['device']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Epochs: {config['num_epochs']}")
    logger.info("=" * 60)
    
    # Load datasets
    logger.info("📂 Loading datasets...")
    train_df = pd.read_pickle('data/processed/train.pkl')
    val_df = pd.read_pickle('data/processed/val.pkl')
    
    logger.info(f"   Train samples: {len(train_df)}")
    logger.info(f"   Val samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = rPPGDataset(train_df)
    val_dataset = rPPGDataset(val_df)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Create model
    logger.info("🏗️ Creating model...")
    model = create_model(sequence_length=config['sequence_length'])
    model = model.to(config['device'])
    
    # Loss and optimizer
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # MLflow setup
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("rppg_heart_rate_estimation")
    
    # Start MLflow run
    with mlflow.start_run(run_name="training_run"):
        # Log parameters
        mlflow.log_params(config)
        mlflow.log_param("model_architecture", "CNN-LSTM-Lite")
        mlflow.log_param("loss_function", "L1Loss (MAE)")
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("val_samples", len(val_df))
        
        # Model parameters
        params = model.count_parameters()
        mlflow.log_param("total_parameters", params['total'])
        mlflow.log_param("trainable_parameters", params['trainable'])
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_correlation': []
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("🎓 Training Started")
        logger.info("=" * 60)
        
        for epoch in range(config['num_epochs']):
            logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            logger.info("-" * 60)
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, config['device'])
            history['train_loss'].append(train_loss)
            
            # Validate
            val_metrics = validate(model, val_loader, criterion, config['device'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_mae'].append(val_metrics['mae'])
            history['val_correlation'].append(val_metrics['correlation'])
            
            # Log to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_metrics['loss'], step=epoch)
            mlflow.log_metric("val_mae_bpm", val_metrics['mae'], step=epoch)
            mlflow.log_metric("val_rmse_bpm", val_metrics['rmse'], step=epoch)
            mlflow.log_metric("val_correlation", val_metrics['correlation'], step=epoch)
            mlflow.log_metric("val_within_5bpm", val_metrics['within_5bpm_percent'], step=epoch)
            mlflow.log_metric("val_within_10bpm", val_metrics['within_10bpm_percent'], step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)
            
            # Learning rate scheduling
            scheduler.step(val_metrics['loss'])
            
            # Print epoch summary
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss:   {val_metrics['loss']:.4f}")
            logger.info(f"Val MAE:    {val_metrics['mae']:.2f} BPM")
            logger.info(f"Val RMSE:   {val_metrics['rmse']:.2f} BPM")
            logger.info(f"Val Corr:   {val_metrics['correlation']:.3f}")
            logger.info(f"Within 5 BPM:  {val_metrics['within_5bpm_percent']:.1f}%")
            logger.info(f"Within 10 BPM: {val_metrics['within_10bpm_percent']:.1f}%")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save model checkpoint
                model_path = 'data/models/best_model.pth'
                Path('data/models').mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_path)
                
                logger.info(f"✅ Best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f"⏳ Patience: {patience_counter}/{config['early_stopping_patience']}")
            
            # Early stopping
            if patience_counter >= config['early_stopping_patience']:
                logger.info(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Load best model
        model.load_state_dict(torch.load('data/models/best_model.pth'))
        
        # Final validation
        logger.info("\n" + "=" * 60)
        logger.info("📊 Final Validation")
        logger.info("=" * 60)
        final_metrics = validate(model, val_loader, criterion, config['device'])
        
        logger.info(f"Final MAE:    {final_metrics['mae']:.2f} BPM")
        logger.info(f"Final RMSE:   {final_metrics['rmse']:.2f} BPM")
        logger.info(f"Final Corr:   {final_metrics['correlation']:.3f}")
        logger.info(f"Within 5 BPM:  {final_metrics['within_5bpm_percent']:.1f}%")
        logger.info(f"Within 10 BPM: {final_metrics['within_10bpm_percent']:.1f}%")
        
        # Log final metrics
        mlflow.log_metric("final_val_mae", final_metrics['mae'])
        mlflow.log_metric("final_val_correlation", final_metrics['correlation'])
        
        # Save model to MLflow
        logger.info("\n📦 Saving model to MLflow...")
        try:
            # Create model signature
            input_example = torch.randn(1, 900, 3)
            signature = mlflow.models.infer_signature(
                input_example.numpy(),
                model(input_example).detach().numpy()
            )

            # Log model with signature
            mlflow.pytorch.log_model(
                model,
                "model",
                signature=signature,
                input_example=input_example.numpy()
            )

            # Save model checkpoint as artifact
            mlflow.log_artifact('data/models/best_model.pth')

            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, "rPPGHeartRateEstimator")

            logger.info("✅ Model saved to MLflow successfully")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save model to MLflow: {e}")
            logger.info("Model is still saved locally at data/models/best_model.pth")
        
        logger.info("=" * 60)
        logger.info("✅ Training Complete!")
        logger.info("=" * 60)
        logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        logger.info(f"View results: http://localhost:5001")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
