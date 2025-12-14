"""
Model utility functions for saving, loading, and managing models.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
import json
import pickle


def save_model(model: nn.Module, 
               optimizer: torch.optim.Optimizer,
               epoch: int,
               loss: float,
               metrics: Dict[str, float],
               save_dir: Path,
               filename: str = 'best_model.pth') -> Path:
    """
    Save model checkpoint with metadata.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        metrics: Dictionary of metrics
        save_dir: Directory to save model
        filename: Model filename
        
    Returns:
        Path to saved model file
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'model_config': {
            'model_class': model.__class__.__name__,
            'model_params': getattr(model, 'config', {})
        }
    }
    
    torch.save(checkpoint, save_path)
    return save_path


def load_model(model: nn.Module,
               model_path: Path,
               optimizer: Optional[torch.optim.Optimizer] = None,
               device: str = 'cpu') -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model (structure should match saved model)
        model_path: Path to saved model
        optimizer: Optimizer to load state into
        device: Device to load model to
        
    Returns:
        Dictionary with loaded metadata
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
        'metrics': checkpoint.get('metrics', {}),
        'model_config': checkpoint.get('model_config', {})
    }
    
    return metadata


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def save_training_config(config: Dict[str, Any], save_dir: Path) -> Path:
    """
    Save training configuration to JSON file.
    
    Args:
        config: Training configuration dictionary
        save_dir: Directory to save config
        
    Returns:
        Path to saved config file
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / 'training_config.json'
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path


def save_scaler(scaler: Any, save_dir: Path, filename: str = 'scaler.pkl') -> Path:
    """
    Save fitted scaler for later use.
    
    Args:
        scaler: Fitted scaler object
        save_dir: Directory to save scaler
        filename: Scaler filename
        
    Returns:
        Path to saved scaler file
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = save_dir / filename
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    return scaler_path


def load_scaler(scaler_path: Path) -> Any:
    """
    Load fitted scaler.
    
    Args:
        scaler_path: Path to saved scaler
        
    Returns:
        Loaded scaler object
    """
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return scaler


def get_device() -> str:
    """
    Get the best available device.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def model_summary(model: nn.Module, input_size: tuple) -> str:
    """
    Generate a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (without batch dimension)
        
    Returns:
        Model summary string
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx+1}"
            summary[m_key] = {
                'input_shape': list(input[0].shape),
                'output_shape': list(output.shape),
                'nb_params': sum([p.numel() for p in module.parameters()])
            }
        
        if not isinstance(module, nn.Sequential) and \
           not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))
    
    # Create a dummy input
    device = next(model.parameters()).device
    x = torch.zeros(1, *input_size).to(device)
    
    # Register hooks
    summary = {}
    hooks = []
    model.apply(register_hook)
    
    # Forward pass
    model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Format summary
    summary_str = "Model Summary\n"
    summary_str += "=" * 50 + "\n"
    for layer_name, layer_info in summary.items():
        summary_str += f"{layer_name:<20} {str(layer_info['output_shape']):<20} {layer_info['nb_params']:<10}\n"
    
    total_params = sum([layer['nb_params'] for layer in summary.values()])
    summary_str += "=" * 50 + "\n"
    summary_str += f"Total parameters: {total_params:,}\n"
    
    return summary_str
