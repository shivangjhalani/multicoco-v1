"""
MultiCoCo: Multimodal Coconut - Visual Question Answering with Latent Space Reasoning

This package extends Meta's Coconut latent-space reasoning framework to support 
multimodal visual question answering through integration with InternVL3-1B-Pretrained.

Core Components:
- MultimodalCoconut: Main model class combining Coconut reasoning with InternVL3 vision
- VQAMultimodalProcessor: Data processing pipeline for VQA datasets
- MultimodalCoconutCollator: Batch collation with variable tile support
- MultimodalCoconutTrainer: Training infrastructure with stage-based curriculum
- MultimodalCoconutEvaluator: Evaluation and metrics system
"""

from .model import MultimodalCoconut, setup_special_tokens
from .dataset import VQAMultimodalProcessor
from .collator import MultimodalCoconutCollator
from .trainer import MultimodalCoconutTrainer
from .evaluator import MultimodalCoconutEvaluator
from .utils import MultiCoCoConfig, setup_logging, load_config

__version__ = "0.1.0"
__author__ = "MultiCoCo Team"

__all__ = [
    "MultimodalCoconut",
    "setup_special_tokens",
    "VQAMultimodalProcessor", 
    "MultimodalCoconutCollator",
    "MultimodalCoconutTrainer",
    "MultimodalCoconutEvaluator",
    "MultiCoCoConfig",
    "setup_logging",
    "load_config"
]