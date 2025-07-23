# Design Document

## Overview

MultiCoCo (Multimodal Coconut) is a novel multimodal reasoning system that extends Meta's Coconut latent-space reasoning framework to support visual question answering through integration with InternVL3-1B-Pretrained. The system combines continuous thought reasoning with state-of-the-art multimodal capabilities to perform visual reasoning through latent space rather than traditional token-based processing.

The key innovation lies in extending Coconut's continuous thought mechanism from purely textual reasoning to multimodal contexts, enabling the model to reason about visual content through hidden state representations while maintaining the breadth-first search capabilities that make Coconut superior for complex reasoning tasks.

## Architecture

### High-Level System Architecture

```
Input Processing:
[Image] → InternVL3 Vision Encoder → Visual Features (256 tokens/tile)
[Text with <IMG_CONTEXT> tokens] → Tokenizer → Token Sequence

Token-Level Multimodal Fusion (InternVL3 Pattern):
[Token Sequence] → Replace <IMG_CONTEXT> with Visual Features → Multimodal Sequence

Continuous Reasoning:
[Multimodal Sequence] → Coconut Multi-Pass Reasoning → Latent Thoughts
                                                    ↓
[Final Sequence] → Language Model → Generated Answer
```

### Core Components

#### 1. MultimodalCoconut Model
- **Base Architecture**: Extends Coconut's multi-pass reasoning to handle visual inputs
- **Model Delegation**: Uses complete InternVL3 model for visual processing delegation
- **Token Management**: Integrates both Coconut and InternVL3 special tokens
- **Continuous Reasoning**: Implements iterative multi-pass processing for latent token replacement

#### 2. Visual Processing Pipeline
- **Dynamic Preprocessing**: Uses InternVL3's dynamic tiling for variable image sizes (1-12 tiles)
- **Feature Extraction**: Delegates to InternVL3's extract_feature method for consistency
- **Token Replacement**: Follows InternVL3's exact pattern for IMG_CONTEXT token replacement
- **Batch Processing**: Handles variable tile counts using concatenated pixel_values and image_flags

#### 3. Data Processing System
- **VQA Integration**: Supports A-OKVQA, VQAv2, GQA, and TextVQA through HuggingFace datasets
- **Stage-Based Processing**: Applies Coconut's curriculum learning with visual-aware adaptations
- **Prompt Engineering**: Creates proper multimodal prompts with correct token ordering
- **Label Management**: Supervises only reasoning and answer portions

## Components and Interfaces

### MultimodalCoconut Class Interface

```python
class MultimodalCoconut(nn.Module):
    def __init__(self,
                 base_internvl_model: InternVLChatModel,
                 tokenizer: AutoTokenizer,
                 config: Config,
                 latent_token_id: Optional[int] = None,
                 start_latent_id: Optional[int] = None,
                 end_latent_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None,
                 img_context_token_id: Optional[int] = None,
                 **kwargs) -> None
    
    def forward(self,
                input_ids: torch.Tensor,           # [batch_size, seq_len]
                attention_mask: torch.Tensor,      # [batch_size, seq_len]
                labels: torch.Tensor,              # [batch_size, seq_len]
                position_ids: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
                pixel_values: Optional[torch.Tensor] = None,  # [total_tiles, 3, 448, 448]
                image_flags: Optional[torch.Tensor] = None,   # [total_tiles]
                **kwargs) -> Dict[str, torch.Tensor]
    
    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 pixel_values: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 50,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 synced_gpus: bool = False,
                 **kwargs) -> Dict[str, torch.Tensor]
    
    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor
    
    def _coconut_continuous_reasoning(self,
                                    inputs_embeds: torch.Tensor,
                                    input_ids: torch.Tensor,
                                    attention_mask: torch.Tensor,
                                    labels: torch.Tensor,
                                    position_ids: torch.Tensor) -> Dict[str, torch.Tensor]
```

### VQAMultimodalProcessor Interface

```python
class VQAMultimodalProcessor:
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 image_processor: AutoImageProcessor,
                 config: Config) -> None
    
    def process_sample(self,
                      sample: Dict[str, Any],
                      stage: int = 0) -> Dict[str, torch.Tensor]
    
    def load_image_with_dynamic_tiling(self,
                                     image: PIL.Image,
                                     input_size: int = 448,
                                     max_num: int = 12) -> torch.Tensor
    
    def _apply_coconut_stage_training(self,
                                    reasoning_steps: List[str],
                                    stage: int,
                                    has_visual_context: bool = False) -> str
```

### MultimodalCoconutCollator Interface

```python
class MultimodalCoconutCollator:
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 latent_id: int,
                 label_pad_token_id: int = -100) -> None
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]
```

### MultimodalCoconutTrainer Interface

```python
class MultimodalCoconutTrainer:
    def __init__(self, config: Config) -> None
    
    def setup_model_and_data(self) -> None
    
    def train_all_stages(self) -> None
    
    def train_single_stage(self,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          optimizer: torch.optim.Optimizer,
                          stage: int) -> float
    
    def evaluate(self, val_loader: DataLoader) -> float
    
    def load_vqa_dataset(self, split: str = 'train') -> Dataset
    
    def save_checkpoint(self, stage: int, accuracy: float) -> None
```

## Data Models

### Configuration Schema

```python
@dataclass
class MultiCoCoConfig:
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
```

### Data Sample Schema

```python
@dataclass
class VQASample:
    image: PIL.Image                      # Visual content
    question: str                         # Natural language question
    choices: List[str]                    # Multiple choice options
    correct_choice_idx: int               # Index of correct answer
    rationales: List[str]                 # Human-provided reasoning steps
    question_id: str                      # Unique identifier
    image_id: str                         # Image identifier
    
    # Processed fields
    pixel_values: torch.Tensor            # [num_tiles, 3, 448, 448]
    input_ids: torch.Tensor               # [seq_len]
    attention_mask: torch.Tensor          # [seq_len]
    labels: torch.Tensor                  # [seq_len]
    image_flags: torch.Tensor             # [num_tiles]
    position_ids: torch.Tensor            # [seq_len]
```

### Training Batch Schema

```python
@dataclass
class TrainingBatch:
    input_ids: torch.Tensor               # [batch_size, seq_len]
    attention_mask: torch.Tensor          # [batch_size, seq_len]
    labels: torch.Tensor                  # [batch_size, seq_len]
    pixel_values: torch.Tensor            # [total_tiles_in_batch, 3, 448, 448]
    image_flags: torch.Tensor             # [total_tiles_in_batch]
    # position_ids generated dynamically in forward()
```

## Error Handling

### Model Initialization Errors
- **Missing Components**: Validate that base_internvl_model has required attributes (vision_model, language_model, mlp1)
- **Token ID Errors**: Handle cases where special tokens are not found in tokenizer vocabulary
- **Configuration Errors**: Validate critical configuration parameters with proper defaults

### Runtime Errors
- **Shape Mismatches**: Handle tensor shape mismatches in visual processing with fallback logic
- **Memory Errors**: Implement gradient checkpointing and proper memory management
- **Generation Errors**: Handle EOS token detection and max_new_tokens limits properly

### Data Processing Errors
- **Image Loading Errors**: Handle corrupted or missing images with appropriate fallbacks
- **Tokenization Errors**: Handle sequence length limits and special token conflicts
- **Batch Processing Errors**: Handle variable-sized inputs with proper padding and masking

### Error Recovery Strategies
- **Graceful Degradation**: Fall back to text-only processing when visual processing fails
- **Checkpoint Recovery**: Resume training from last valid checkpoint on failures
- **Distributed Training**: Handle GPU failures and synchronization issues

## Testing Strategy

### Unit Tests

#### Model Component Tests
```python
def test_multimodal_coconut_initialization():
    """Test proper model initialization with all components"""
    
def test_token_management():
    """Test special token addition and embedding initialization"""
    
def test_visual_processing():
    """Test image processing and feature extraction"""
    
def test_continuous_reasoning():
    """Test multi-pass latent token processing"""
    
def test_generation():
    """Test autoregressive generation with latent handling"""
```

#### Data Processing Tests
```python
def test_vqa_sample_processing():
    """Test VQA sample conversion to training format"""
    
def test_stage_based_processing():
    """Test curriculum learning stage progression"""
    
def test_collator_functionality():
    """Test batch collation with variable inputs"""
    
def test_dynamic_tiling():
    """Test image preprocessing with dynamic tiling"""
```

### Integration Tests

#### End-to-End Pipeline Tests
```python
def test_training_pipeline():
    """Test complete training pipeline from data loading to model saving"""
    
def test_evaluation_pipeline():
    """Test evaluation on validation set with metric computation"""
    
def test_generation_pipeline():
    """Test inference pipeline with sample generation"""
    
def test_distributed_training():
    """Test multi-GPU training with proper synchronization"""
```

#### Dataset Integration Tests
```python
def test_aokvqa_integration():
    """Test A-OKVQA dataset loading and processing"""
    
def test_vqav2_integration():
    """Test VQAv2 dataset loading and processing"""
    
def test_batch_processing():
    """Test batch processing with variable image counts"""
```

### Performance Tests

#### Memory and Speed Tests
```python
def test_memory_usage():
    """Test memory consumption during training and inference"""
    
def test_training_speed():
    """Test training throughput and convergence speed"""
    
def test_inference_speed():
    """Test generation speed and latency"""
    
def test_scalability():
    """Test performance scaling with batch size and sequence length"""
```

### Validation Tests

#### Model Behavior Tests
```python
def test_stage_progression():
    """Validate that stage-based training follows curriculum correctly"""
    
def test_latent_token_replacement():
    """Validate that latent tokens are replaced with correct hidden states"""
    
def test_multimodal_fusion():
    """Validate that visual and text information are properly integrated"""
    
def test_generation_quality():
    """Validate that generated answers are coherent and relevant"""
```

## Implementation Notes

### Critical Implementation Requirements

1. **Token Replacement Logic**: Must follow exact Coconut pattern for latent token replacement using hidden states from preceding positions
2. **KV Cache Management**: Essential for multi-pass efficiency - must extract and reuse past_key_values correctly
3. **Visual Token Integration**: Must use InternVL3's exact mathematical operations to preserve gradient flow
4. **Memory Optimization**: Use .detach() to break computational graph and prevent O(N²) memory growth
5. **Distributed Training**: Handle FSDP synchronization with proper forward pass counting

### Performance Considerations

1. **Gradient Checkpointing**: Enable for both vision and language models to reduce memory usage
2. **Mixed Precision**: Use bfloat16 for training efficiency while maintaining numerical stability
3. **Dynamic Batching**: Handle variable tile counts efficiently using concatenation instead of padding
4. **Cache Optimization**: Reuse KV cache across multi-pass reasoning for computational efficiency

### Extensibility Design

1. **Dataset Adapters**: Modular design allows easy addition of new VQA datasets
2. **Configuration System**: YAML-based configs enable easy experimentation
3. **Model Components**: Clean interfaces allow swapping of vision or language components
4. **Evaluation Metrics**: Pluggable metric system supports various evaluation protocols