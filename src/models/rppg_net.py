"""
Lightweight rPPG model for CPU training
Optimized for quick demonstration without GPU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class rPPGNetLite(nn.Module):
    """
    Efficient CNN-LSTM hybrid for heart rate estimation
    
    Architecture:
    - 1D Convolutional layers for temporal feature extraction
    - LSTM for sequence modeling
    - Fully connected layers for regression
    
    Input: (batch_size, sequence_length, 3) - RGB signals over time
    Output: (batch_size, 1) - Heart rate in BPM
    """
    
    def __init__(self, sequence_length=900, hidden_size=64, num_layers=1, dropout=0.3):
        """
        Args:
            sequence_length: Number of frames (30 fps * 30 sec = 900)
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(rPPGNetLite, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        
        # 1D Convolutional layers for temporal features
        # Input: (batch, 3, 900) where 3 is RGB channels
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_conv = nn.Dropout(dropout)
        
        # After 3 pooling layers: 900 -> 450 -> 225 -> 112
        reduced_length = sequence_length // (2 ** 3)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=64,  # From conv3 output channels
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fully connected layers for regression
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout_fc = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, 3)
               Example: (16, 900, 3) for batch of 16 videos
        
        Returns:
            Heart rate predictions of shape (batch_size, 1)
        """
        # Reshape for 1D convolution: (batch, channels, length)
        x = x.permute(0, 2, 1)  # (batch, 3, 900)
        
        # Convolutional feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # (batch, 16, 450)
        x = self.dropout_conv(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # (batch, 32, 225)
        x = self.dropout_conv(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # (batch, 64, 112)
        x = self.dropout_conv(x)
        
        # Reshape for LSTM: (batch, length, features)
        x = x.permute(0, 2, 1)  # (batch, 112, 64)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch, 112, hidden_size)
        
        # Use last time step
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Regression
        x = F.relu(self.fc1(last_output))
        x = self.dropout_fc(x)
        heart_rate = self.fc2(x)  # (batch, 1)
        
        return heart_rate
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


def create_model(sequence_length=900):
    """
    Factory function to create model
    
    Args:
        sequence_length: Input sequence length
    
    Returns:
        Initialized model
    """
    model = rPPGNetLite(sequence_length=sequence_length)
    
    # Print model info
    params = model.count_parameters()
    print(f"Model created:")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    
    return model


# Test the model
if __name__ == '__main__':
    # Create dummy input
    batch_size = 4
    sequence_length = 900
    num_channels = 3
    
    x = torch.randn(batch_size, sequence_length, num_channels)
    
    # Create model
    model = create_model(sequence_length)
    
    # Forward pass
    output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output (heart rates): {output.squeeze().detach().numpy()}")
