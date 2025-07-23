"""
Utility Functions and Configuration Management

This module provides utility functions, configuration classes, and helper
functions for the MultiCoCo system.

Key Features:
- MultiCoCoConfig dataclass with comprehensive configuration options
- Logging setup and management
- Configuration loading and validation
- Helper functions for model and data management
"""

import os
import yaml
import json
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from pathlib import Path


@dataclass
class MultiCoCoConfig:
    """
    Comprehensive configuration class for MultiCoCo system.
    
    This dataclass contains all configuration parameters needed for training,
    evaluation, and inference with multimodal Coconut models.
    """
    
    # Model configuration
    model_id: str = "OpenGVLab/InternVL3-1B-Pretrained"
    torch_dtype: str = "bfloat16"
    low_cpu_mem_usage: bool = True
    trust_remote_code: bool = True
    use_flash_attn: bool = True
    
    # Coconut-specific parameters
    c_thought: int = 2                    # Continuous thoughts per reasoning step
    epochs_per_stage: int = 3             # Training epochs per stage
    max_latent_stage: int = 4             # Maximum latent stages
    pad_latent_to_max: bool = True        # Consistent padding strategy
    reset_optimizer: bool = True          # Reset optimizer between stages
    
    # Training configuration
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 20
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    bf16: bool = True
    use_gradient_checkpointing: bool = True
    
    # Data configuration
    max_sequence_length: int = 2048
    image_size: int = 448
    max_num_tiles: int = 12
    num_image_token: int = 256
    
    # Dataset configuration
    dataset_name: str = "aokvqa"
    hf_dataset_id: str = "HuggingFaceM4/A-OKVQA"
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    
    # Hardware configuration
    device: str = "cuda"
    world_size: int = 1
    distributed: bool = False
    num_workers: int = 4
    
    # Generation parameters
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    do_sample: bool = False
    
    # InternVL3 specific parameters
    downsample_ratio: float = 0.5
    ps_version: str = "v2"
    select_layer: int = -1
    template: str = "internvl2_5"
    max_dynamic_patch: int = 12
    min_dynamic_patch: int = 1
    use_thumbnail: bool = True
    pad2square: bool = False
    dynamic_image_size: bool = True
    force_image_size: int = 448
    
    # Logging and output
    output_dir: str = "./outputs"
    logging_dir: str = "./logs"
    log_level: str = "INFO"
    save_steps: int = 500
    eval_steps: int = 500
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("Warning: CUDA not available, using CPU")
        
        # Validate paths
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        # Validate numeric parameters
        if self.c_thought <= 0:
            raise ValueError("c_thought must be positive")
        if self.max_latent_stage < 0:
            raise ValueError("max_latent_stage must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MultiCoCoConfig':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            MultiCoCoConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert string values to appropriate types
        config_dict = cls._convert_types(config_dict)
        
        return cls(**config_dict)
    
    @classmethod
    def _convert_types(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert string values from YAML to appropriate types.
        
        Args:
            config_dict: Raw configuration dictionary from YAML
            
        Returns:
            Configuration dictionary with proper types
        """
        # Define type mappings for configuration fields
        type_mappings = {
            # Numeric fields that should be int
            'c_thought': int,
            'epochs_per_stage': int,
            'max_latent_stage': int,
            'batch_size': int,
            'gradient_accumulation_steps': int,
            'num_epochs': int,
            'warmup_steps': int,
            'max_sequence_length': int,
            'image_size': int,
            'max_num_tiles': int,
            'num_image_token': int,
            'world_size': int,
            'num_workers': int,
            'max_new_tokens': int,
            'select_layer': int,
            'max_dynamic_patch': int,
            'min_dynamic_patch': int,
            'force_image_size': int,
            'save_steps': int,
            'eval_steps': int,
            'seed': int,
            
            # Numeric fields that should be float
            'learning_rate': float,
            'weight_decay': float,
            'max_grad_norm': float,
            'temperature': float,
            'top_p': float,
            'downsample_ratio': float,
            
            # Boolean fields
            'low_cpu_mem_usage': bool,
            'trust_remote_code': bool,
            'use_flash_attn': bool,
            'pad_latent_to_max': bool,
            'reset_optimizer': bool,
            'bf16': bool,
            'use_gradient_checkpointing': bool,
            'distributed': bool,
            'do_sample': bool,
            'use_thumbnail': bool,
            'pad2square': bool,
            'dynamic_image_size': bool,
        }
        
        # Convert types
        converted_dict = {}
        for key, value in config_dict.items():
            if key in type_mappings and value is not None:
                target_type = type_mappings[key]
                try:
                    if target_type == bool:
                        # Handle boolean conversion from various formats
                        if isinstance(value, str):
                            converted_dict[key] = value.lower() in ('true', 'yes', '1', 'on')
                        else:
                            converted_dict[key] = bool(value)
                    else:
                        converted_dict[key] = target_type(value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert {key}={value} to {target_type.__name__}, using default")
                    converted_dict[key] = value
            else:
                converted_dict[key] = value
        
        return converted_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MultiCoCoConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            MultiCoCoConfig instance
        """
        # Convert types for dictionary input as well
        config_dict = cls._convert_types(config_dict)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration file
        """
        config_dict = self.__dict__.copy()
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return self.__dict__.copy()


def setup_logging(config: MultiCoCoConfig) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration object with logging parameters
        
    Returns:
        Configured logger instance
    """
    # Create logging directory
    os.makedirs(config.logging_dir, exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.logging_dir, 'multicoco.log')),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('multicoco')
    logger.info(f"Logging initialized with level: {config.log_level}")
    
    return logger


def load_config(config_path: str) -> MultiCoCoConfig:
    """
    Load configuration from file.
    
    Supports both YAML and JSON configuration files.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        MultiCoCoConfig instance
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        return MultiCoCoConfig.from_yaml(str(config_path))
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return MultiCoCoConfig.from_dict(config_dict)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Dictionary containing device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
    }
    
    return device_info


def format_model_size(num_parameters: int) -> str:
    """
    Format model size in human-readable format.
    
    Args:
        num_parameters: Number of model parameters
        
    Returns:
        Formatted string (e.g., "1.2B", "345M")
    """
    if num_parameters >= 1e9:
        return f"{num_parameters / 1e9:.1f}B"
    elif num_parameters >= 1e6:
        return f"{num_parameters / 1e6:.1f}M"
    elif num_parameters >= 1e3:
        return f"{num_parameters / 1e3:.1f}K"
    else:
        return str(num_parameters)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
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


def create_dataset_configs() -> Dict[str, Dict[str, Any]]:
    """
    Create default dataset configurations.
    
    Returns:
        Dictionary of dataset configurations
    """
    return {
        'aokvqa': {
            'hf_dataset_id': 'HuggingFaceM4/A-OKVQA',
            'train_split': 'train',
            'val_split': 'validation',
            'test_split': 'test',
            'task_type': 'multiple_choice',
            'num_choices': 4,
            'metric': 'choice_accuracy'
        },
        'vqav2': {
            'hf_dataset_id': 'HuggingFaceM4/VQAv2',
            'train_split': 'train',
            'val_split': 'validation',
            'test_split': 'test',
            'task_type': 'open_ended',
            'metric': 'vqa_accuracy'
        },
        'gqa': {
            'hf_dataset_id': 'HuggingFaceM4/GQA',
            'train_split': 'train',
            'val_split': 'validation',
            'test_split': 'test',
            'task_type': 'open_ended',
            'metric': 'exact_match'
        },
        'textvqa': {
            'hf_dataset_id': 'HuggingFaceM4/TextVQA',
            'train_split': 'train',
            'val_split': 'validation',
            'test_split': 'test',
            'task_type': 'open_ended',
            'metric': 'exact_match'
        }
    }


def validate_config(config: MultiCoCoConfig) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation messages
    """
    messages = []
    
    # Check hardware compatibility
    if config.device == "cuda" and not torch.cuda.is_available():
        messages.append("Warning: CUDA requested but not available, will use CPU")
    
    # Check memory requirements
    if config.batch_size > 16 and config.use_gradient_checkpointing is False:
        messages.append("Warning: Large batch size without gradient checkpointing may cause OOM")
    
    # Check dataset configuration
    dataset_configs = create_dataset_configs()
    if config.dataset_name not in dataset_configs:
        messages.append(f"Warning: Unknown dataset '{config.dataset_name}', using default configuration")
    
    # Check training parameters
    if config.learning_rate > 1e-3:
        messages.append("Warning: Learning rate seems high, consider reducing")
    
    if config.max_latent_stage > 6:
        messages.append("Warning: Very high max_latent_stage may lead to training instability")
    
    return messages