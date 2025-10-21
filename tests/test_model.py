"""
Test suite for model
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.rppg_net import create_model


def test_model_creation():
    """Test model can be created"""
    model = create_model(sequence_length=900)
    assert model is not None
    
    params = model.count_parameters()
    assert params['total'] > 0
    assert params['trainable'] > 0


def test_model_forward_pass():
    """Test forward pass works"""
    model = create_model(sequence_length=900)
    model.eval()
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 900, 3)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, 1)
    assert not torch.isnan(output).any()


def test_model_output_range():
    """Test outputs are in valid heart rate range"""
    model = create_model(sequence_length=900)
    model.eval()
    
    x = torch.randn(8, 900, 3)
    
    with torch.no_grad():
        output = model(x)
    
    # Predictions should be somewhat reasonable (even untrained)
    assert output.min() > 0
    assert output.max() < 300  # Physiological maximum


def test_model_batch_sizes():
    """Test different batch sizes"""
    model = create_model(sequence_length=900)
    model.eval()
    
    for batch_size in [1, 4, 8, 16]:
        x = torch.randn(batch_size, 900, 3)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1)


def test_model_determinism():
    """Test model produces same output for same input"""
    model = create_model(sequence_length=900)
    model.eval()
    
    x = torch.randn(2, 900, 3)
    
    outputs = []
    for _ in range(3):
        with torch.no_grad():
            output = model(x)
        outputs.append(output)
    
    # All outputs should be identical
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
