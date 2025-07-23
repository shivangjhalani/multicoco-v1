# Implementation Plan

## IMPORTANT NOTE
ALWAYS REFER TO comprehensive-design.md PRESENT IN PROJECT ROOT
ALSO REFER TO THESE REFERENCE CODEBASES WHEN NEEDED
**Reference Codebases:** These are located in the `reference/` folder.

    *   `coconut`: The original Meta implementation.

    *   `InternVL`: A state-of-the-art multimodal architecture.


> All design references in tasks refer to comprehensive-design.md

- [ ] 1. Set up project structure and core interfaces
  - Create directory structure following comprehensive-design.md Architecture section: multicoco/ with model.py, dataset.py, collator.py, trainer.py, evaluator.py, utils.py
  - Define base configuration classes implementing MultiCoCoConfig dataclass from comprehensive-design.md Data Models section
  - Set up package initialization with proper imports following comprehensive-design.md Components and Interfaces
  - Create configs/ directory with YAML files for each dataset as specified in comprehensive-design.md
  - _Requirements: 1.1, 10.1, 10.3_
  - _Design References: Architecture section, Components and Interfaces, Data Models - Configuration Schema_

- [ ] 2. Implement core MultimodalCoconut model architecture
- [ ] 2.1 Create MultimodalCoconut class with proper initialization
  - Implement __init__ method following comprehensive-design.md MultimodalCoconut Class Interface signature exactly
  - Add validation for required model components (vision_model, language_model, mlp1) as specified in comprehensive-design.md Error Handling section
  - Set up token ID management for both Coconut and InternVL3 special tokens following comprehensive-design.md Components and Interfaces
  - Initialize generation counter (gen_forward_cnt) and configuration parameters as per comprehensive-design.md Architecture section
  - Follow comprehensive-design.md Implementation Notes for critical token replacement logic requirements
  - _Requirements: 1.1, 2.1, 2.3_
  - _Design References: MultimodalCoconut Class Interface, Error Handling - Model Initialization Errors, Implementation Notes - Critical Implementation Requirements_

- [ ] 2.2 Implement special token management and embedding initialization
  - Add both Coconut tokens (<|start-latent|>, <|end-latent|>, <|latent|>) and InternVL3 tokens (<img>, </img>, <IMG_CONTEXT>) as specified in comprehensive-design.md Components and Interfaces
  - Use add_special_tokens() method to properly register tokens and prevent tokenization splitting following comprehensive-design.md Implementation Notes
  - Initialize new token embeddings using averaged embeddings from existing tokens as per comprehensive-design.md Error Handling section
  - Resize model embeddings and handle both input and output embeddings properly following comprehensive-design.md MultimodalCoconut Class Interface
  - Implement token ID validation and error handling as specified in comprehensive-design.md Error Handling - Model Initialization Errors
  - _Requirements: 2.1, 2.2, 2.4_
  - _Design References: Components and Interfaces, Implementation Notes - Critical Implementation Requirements, Error Handling - Model Initialization Errors_

- [ ] 2.3 Implement visual processing delegation to InternVL3
  - Create extract_feature method following comprehensive-design.md MultimodalCoconut Class Interface that delegates to base InternVL3 model's extract_feature
  - Ensure proper handling of select_layer, pixel_shuffle, and downsample_ratio parameters as specified in comprehensive-design.md Visual Processing Pipeline
  - Validate visual feature extraction with proper error handling for None inputs following comprehensive-design.md Error Handling - Runtime Errors
  - Follow comprehensive-design.md Implementation Notes for visual token integration requirements
  - _Requirements: 3.1, 3.2_
  - _Design References: MultimodalCoconut Class Interface, Visual Processing Pipeline, Error Handling - Runtime Errors, Implementation Notes - Critical Implementation Requirements_

- [ ] 2.4 Implement multimodal forward pass with token replacement
  - Create forward method following comprehensive-design.md MultimodalCoconut Class Interface signature exactly with input_ids, attention_mask, labels, pixel_values, and image_flags
  - Implement InternVL3's exact token replacement pattern for IMG_CONTEXT tokens as specified in comprehensive-design.md Visual Processing Pipeline
  - Handle variable batch sizes and tile counts using concatenated pixel_values following comprehensive-design.md Training Batch Schema
  - Generate position_ids dynamically as torch.arange(0, seq_len) after processing per comprehensive-design.md Implementation Notes
  - Follow comprehensive-design.md Error Handling - Runtime Errors for shape mismatch handling
  - _Requirements: 1.1, 1.5, 3.3, 3.4, 6.4_
  - _Design References: MultimodalCoconut Class Interface, Visual Processing Pipeline, Training Batch Schema, Implementation Notes - Critical Implementation Requirements, Error Handling - Runtime Errors_

- [ ] 2.5 Implement Coconut's continuous reasoning with multi-pass processing
  - Create _coconut_continuous_reasoning method following comprehensive-design.md MultimodalCoconut Class Interface with iterative latent token processing
  - Process one latent token per pass to preserve dependency chains as specified in comprehensive-design.md Implementation Notes - Critical Implementation Requirements
  - Implement proper KV cache management for efficient computation reuse following comprehensive-design.md Performance Considerations
  - Use .detach() to prevent O(N²) memory growth in autograd graph per comprehensive-design.md Implementation Notes - Memory Optimization
  - Handle computation range management to minimize redundant forward passes as detailed in comprehensive-design.md Architecture section
  - _Requirements: 1.3, 6.3, 6.4_
  - _Design References: MultimodalCoconut Class Interface, Implementation Notes - Critical Implementation Requirements, Performance Considerations, Architecture section_

- [ ] 2.6 Implement generation capabilities with dynamic latent handling
  - Create generate method following comprehensive-design.md MultimodalCoconut Class Interface supporting both sampling and greedy decoding
  - Implement autoregressive generation with proper KV cache usage as specified in comprehensive-design.md Performance Considerations
  - Handle dynamically generated latent tokens during generation per comprehensive-design.md Implementation Notes - Critical Implementation Requirements
  - Add FSDP synchronization support with proper forward pass counting following comprehensive-design.md Implementation Notes - Distributed Training
  - Support batch processing with per-item completion tracking as detailed in comprehensive-design.md Architecture section
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  - _Design References: MultimodalCoconut Class Interface, Performance Considerations, Implementation Notes - Critical Implementation Requirements, Implementation Notes - Distributed Training, Architecture section_

- [ ] 3. Implement VQA data processing pipeline
- [ ] 3.1 Create VQAMultimodalProcessor class for dataset handling
  - Implement __init__ method following comprehensive-design.md VQAMultimodalProcessor Interface with tokenizer, image_processor, and config
  - Add Coconut special tokens to tokenizer with proper token ID retrieval as specified in comprehensive-design.md Data Processing System
  - Set up configuration parameters for image processing and token management following comprehensive-design.md Data Models - Configuration Schema
  - Follow comprehensive-design.md Implementation Notes for token management requirements
  - _Requirements: 5.1, 2.1_
  - _Design References: VQAMultimodalProcessor Interface, Data Processing System, Data Models - Configuration Schema, Implementation Notes - Critical Implementation Requirements_

- [ ] 3.2 Implement dynamic image preprocessing with tiling
  - Create load_image_with_dynamic_tiling method following comprehensive-design.md VQAMultimodalProcessor Interface using InternVL3's dynamic_preprocess
  - Support 1-12 tiles based on image aspect ratio and size as specified in comprehensive-design.md Visual Processing Pipeline
  - Use InternVL3's build_transform for consistent image preprocessing following comprehensive-design.md Data Processing System
  - Return properly formatted pixel_values tensor with correct tile count per comprehensive-design.md Data Models - VQASample Schema
  - _Requirements: 3.1, 3.3_
  - _Design References: VQAMultimodalProcessor Interface, Visual Processing Pipeline, Data Processing System, Data Models - VQASample Schema_

- [ ] 3.3 Implement VQA sample processing with stage-based curriculum
  - Create process_sample method following comprehensive-design.md VQAMultimodalProcessor Interface that converts HuggingFace VQA samples to training format
  - Format questions with multiple choice options using A, B, C, D structure as specified in comprehensive-design.md Data Processing System
  - Replace <image> placeholders with proper InternVL3 token sequences BEFORE tokenization following comprehensive-design.md Data Models - VQASample Schema
  - Apply stage-based reasoning step replacement with visual-aware adaptations per comprehensive-design.md Data Processing System
  - Create proper labels that supervise only reasoning and answer portions as detailed in comprehensive-design.md Data Models - Training Batch Schema
  - _Requirements: 5.2, 5.3, 5.4, 4.2, 4.4_
  - _Design References: VQAMultimodalProcessor Interface, Data Processing System, Data Models - VQASample Schema, Data Models - Training Batch Schema_

- [ ] 3.4 Implement Coconut stage-based training logic
  - Create _apply_coconut_stage_training method following comprehensive-design.md VQAMultimodalProcessor Interface for curriculum learning
  - Implement Stage 0 with full Chain-of-Thought reasoning as specified in comprehensive-design.md Data Processing System
  - Implement Stage N with first N reasoning steps replaced by latent tokens per comprehensive-design.md Data Processing System
  - Add visual grounding cues for multimodal reasoning coherence following comprehensive-design.md Data Processing System
  - Format latent sequences as <|start-latent|> + <|latent|> * n_tokens + <|end-latent|> per comprehensive-design.md Components and Interfaces
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  - _Design References: VQAMultimodalProcessor Interface, Data Processing System, Components and Interfaces_

- [ ] 4. Implement multimodal data collation with variable tile support
- [ ] 4.1 Create MultimodalCoconutCollator class
  - Implement __init__ method following comprehensive-design.md MultimodalCoconutCollator Interface with tokenizer, latent_id, and label_pad_token_id
  - Set up collation parameters following Coconut's approach as specified in comprehensive-design.md Data Processing System
  - Follow comprehensive-design.md Implementation Notes for batch processing requirements
  - _Requirements: 6.1_
  - _Design References: MultimodalCoconutCollator Interface, Data Processing System, Implementation Notes - Critical Implementation Requirements_

- [ ] 4.2 Implement Coconut's latent token alignment padding
  - Find latest earliest latent position across batch items following comprehensive-design.md MultimodalCoconutCollator Interface
  - Pad sequences to align latent tokens for KV cache efficiency as specified in comprehensive-design.md Performance Considerations
  - Apply padding to input_ids, attention_mask, and labels appropriately per comprehensive-design.md Data Models - Training Batch Schema
  - Skip position_ids padding to avoid duplicate position 0s following comprehensive-design.md Implementation Notes - Critical Implementation Requirements
  - _Requirements: 6.2, 6.5_
  - _Design References: MultimodalCoconutCollator Interface, Performance Considerations, Data Models - Training Batch Schema, Implementation Notes - Critical Implementation Requirements_

- [ ] 4.3 Implement variable-sized visual input handling
  - Concatenate pixel_values across batch items using torch.cat() following comprehensive-design.md MultimodalCoconutCollator Interface
  - Concatenate image_flags for InternVL3's native batch processing as specified in comprehensive-design.md Data Processing System
  - Handle edge cases with proper error handling and fallbacks per comprehensive-design.md Error Handling - Data Processing Errors
  - Use standard sequence padding for text components following comprehensive-design.md Data Models - Training Batch Schema
  - _Requirements: 6.1, 3.3_
  - _Design References: MultimodalCoconutCollator Interface, Data Processing System, Error Handling - Data Processing Errors, Data Models - Training Batch Schema_

- [ ] 5. Implement training infrastructure and pipeline
- [ ] 5.1 Create MultimodalCoconutTrainer class with model setup
  - Implement __init__ method following comprehensive-design.md MultimodalCoconutTrainer Interface with configuration management
  - Create setup_model_and_data method for model and tokenizer initialization as specified in comprehensive-design.md MultimodalCoconutTrainer Interface
  - Load InternVL3 base model with proper configuration parameters following comprehensive-design.md Data Models - Configuration Schema
  - Initialize MultimodalCoconut model with all required components per comprehensive-design.md Architecture section
  - Follow comprehensive-design.md Implementation Notes for model initialization requirements
  - _Requirements: 8.1, 8.2_
  - _Design References: MultimodalCoconutTrainer Interface, Data Models - Configuration Schema, Architecture section, Implementation Notes - Critical Implementation Requirements_

- [ ] 5.2 Implement stage-based training curriculum
  - Create train_all_stages method following comprehensive-design.md MultimodalCoconutTrainer Interface for complete multi-stage training
  - Implement train_single_stage method for individual stage training as specified in comprehensive-design.md MultimodalCoconutTrainer Interface
  - Handle optimizer reset between stages when configured per comprehensive-design.md Data Models - Configuration Schema
  - Process datasets with stage-specific transformations following comprehensive-design.md Data Processing System
  - Track best accuracy and save checkpoints appropriately as detailed in comprehensive-design.md MultimodalCoconutTrainer Interface
  - _Requirements: 4.1, 8.1, 8.3_
  - _Design References: MultimodalCoconutTrainer Interface, Data Models - Configuration Schema, Data Processing System_

- [ ] 5.3 Implement training loop with gradient accumulation
  - Create training loop with proper batch processing following comprehensive-design.md MultimodalCoconutTrainer Interface
  - Implement gradient accumulation and mixed precision training as specified in comprehensive-design.md Performance Considerations
  - Add progress tracking and loss monitoring per comprehensive-design.md Data Models - Configuration Schema
  - Handle distributed training synchronization following comprehensive-design.md Implementation Notes - Distributed Training
  - _Requirements: 8.2, 8.4_
  - _Design References: MultimodalCoconutTrainer Interface, Performance Considerations, Data Models - Configuration Schema, Implementation Notes - Distributed Training_

- [ ] 5.4 Implement validation and checkpoint management
  - Create evaluate method following comprehensive-design.md MultimodalCoconutTrainer Interface for validation set evaluation
  - Implement checkpoint saving with stage and accuracy information as specified in comprehensive-design.md MultimodalCoconutTrainer Interface
  - Add model state management and recovery capabilities per comprehensive-design.md Error Handling - Error Recovery Strategies
  - _Requirements: 8.3, 9.1_
  - _Design References: MultimodalCoconutTrainer Interface, Error Handling - Error Recovery Strategies_

- [ ] 6. Implement evaluation and metrics system
- [ ] 6.1 Create MultimodalCoconutEvaluator class
  - Implement __init__ method with model and configuration setup following comprehensive-design.md Testing Strategy - Validation Tests
  - Set up evaluation pipeline for different VQA datasets as specified in comprehensive-design.md Data Models - Configuration Schema
  - Follow comprehensive-design.md Extensibility Design for pluggable metric system
  - _Requirements: 9.1, 9.2_
  - _Design References: Testing Strategy - Validation Tests, Data Models - Configuration Schema, Extensibility Design_

- [ ] 6.2 Implement answer extraction and parsing
  - Create extract_choice_from_text method for multiple choice parsing following comprehensive-design.md Testing Strategy - Validation Tests
  - Handle various answer formats (A, B, C, D) with regex patterns as specified in comprehensive-design.md Data Processing System
  - Implement fallback logic for unparseable responses per comprehensive-design.md Error Handling - Data Processing Errors
  - _Requirements: 9.4_
  - _Design References: Testing Strategy - Validation Tests, Data Processing System, Error Handling - Data Processing Errors_

- [ ] 6.3 Implement dataset-specific evaluation metrics
  - Add A-OKVQA accuracy and choice accuracy computation following comprehensive-design.md Extensibility Design for pluggable metric system
  - Add VQAv2 accuracy following standard evaluation protocols as specified in comprehensive-design.md Data Models - Configuration Schema
  - Support explanation generation and quality assessment per comprehensive-design.md Testing Strategy - Validation Tests
  - Provide detailed performance breakdowns by question type following comprehensive-design.md Extensibility Design
  - _Requirements: 9.1, 9.2, 9.3, 9.5_
  - _Design References: Extensibility Design, Data Models - Configuration Schema, Testing Strategy - Validation Tests_

- [ ] 7. Implement configuration management system
- [ ] 7.1 Create base configuration classes and YAML support
  - Define MultiCoCoConfig dataclass with all required parameters following comprehensive-design.md Data Models - Configuration Schema exactly
  - Implement YAML configuration loading and validation as specified in comprehensive-design.md Extensibility Design
  - Set up dataset-specific configuration files (A-OKVQA, VQAv2, GQA, TextVQA) per comprehensive-design.md Data Models - Configuration Schema
  - Follow comprehensive-design.md Implementation Notes for configuration management requirements
  - _Requirements: 10.1, 10.2_
  - _Design References: Data Models - Configuration Schema, Extensibility Design, Implementation Notes - Critical Implementation Requirements_

- [ ] 7.2 Implement configuration validation and defaults
  - Add parameter validation with meaningful error messages following comprehensive-design.md Error Handling - Model Initialization Errors
  - Set up proper default values for all configuration options as specified in comprehensive-design.md Data Models - Configuration Schema
  - Handle hardware detection and distributed training setup per comprehensive-design.md Implementation Notes - Distributed Training
  - _Requirements: 10.2, 10.5_
  - _Design References: Error Handling - Model Initialization Errors, Data Models - Configuration Schema, Implementation Notes - Distributed Training_

- [ ] 8. Implement main orchestrator and CLI interface
- [ ] 8.1 Create main run.py script with argument parsing
  - Implement command-line interface following Coconut's pattern as specified in comprehensive-design.md Extensibility Design
  - Support train, eval, and generate modes per comprehensive-design.md Data Models - Configuration Schema
  - Add configuration file loading and override capabilities following comprehensive-design.md Data Models - Configuration Schema
  - Set up logging and reproducibility (seed setting) as detailed in comprehensive-design.md Implementation Notes
  - _Requirements: 10.4_
  - _Design References: Extensibility Design, Data Models - Configuration Schema, Implementation Notes - Critical Implementation Requirements_

- [ ] 8.2 Implement mode-specific execution logic
  - Add training mode execution with stage-based curriculum following comprehensive-design.md MultimodalCoconutTrainer Interface
  - Add evaluation mode with comprehensive metrics as specified in comprehensive-design.md Testing Strategy - Validation Tests
  - Add generation mode for sample output creation per comprehensive-design.md MultimodalCoconut Class Interface
  - Handle checkpoint loading and model initialization following comprehensive-design.md Error Handling - Error Recovery Strategies
  - _Requirements: 10.4_
  - _Design References: MultimodalCoconutTrainer Interface, Testing Strategy - Validation Tests, MultimodalCoconut Class Interface, Error Handling - Error Recovery Strategies_

- [ ] 9. Implement utility functions and helper modules
- [ ] 9.1 Create logging and debugging utilities
  - Implement comprehensive logging system with different levels
  - Add debugging utilities for tensor shapes and model states
  - Create error handling with meaningful messages
  - _Requirements: 10.4_

- [ ] 9.2 Create data preparation and caching utilities
  - Implement dataset downloading and preprocessing scripts
  - Add data caching for processed samples
  - Create utilities for dataset statistics and analysis
  - _Requirements: 5.1_

- [ ] 10. Implement comprehensive testing suite
- [ ] 10.1 Create unit tests for core model components
  - Test MultimodalCoconut initialization and component validation
  - Test token management and embedding initialization
  - Test visual processing and feature extraction
  - Test continuous reasoning and multi-pass processing
  - Test generation capabilities with various configurations

- [ ] 10.2 Create integration tests for data processing
  - Test VQA sample processing with different datasets
  - Test stage-based curriculum learning progression
  - Test batch collation with variable inputs
  - Test dynamic image tiling and preprocessing

- [ ] 10.3 Create end-to-end pipeline tests
  - Test complete training pipeline from data loading to model saving
  - Test evaluation pipeline with metric computation
  - Test generation pipeline with sample creation
  - Test distributed training with multi-GPU setup

- [ ] 11. Create documentation and examples
- [ ] 11.1 Write comprehensive API documentation
  - Document all classes and methods with Google-style docstrings
  - Create usage examples for each major component
  - Add architecture diagrams and flow charts
  - Document configuration options and their effects

- [ ] 11.2 Create training and evaluation guides
  - Write step-by-step training instructions
  - Create evaluation guides for different datasets
  - Add troubleshooting guides for common issues
  - Document performance optimization tips

- [ ] 12. Implement performance optimization and deployment features
- [ ] 12.1 Add memory optimization features
  - Implement gradient checkpointing for memory efficiency
  - Add mixed precision training support
  - Optimize batch processing for variable inputs
  - Add memory profiling and monitoring tools

- [ ] 12.2 Add distributed training and deployment support
  - Implement multi-GPU training with proper synchronization
  - Add FSDP support for large model training
  - Create deployment scripts for inference serving
  - Add model quantization and optimization for deployment