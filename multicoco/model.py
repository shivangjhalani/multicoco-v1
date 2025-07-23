"""
MultimodalCoconut Model Implementation

This module implements the core MultimodalCoconut model that extends Coconut's 
continuous reasoning capabilities to support multimodal visual question answering
through integration with InternVL3-1B-Pretrained.

Key Features:
- Extends Coconut's multi-pass reasoning to handle visual inputs
- Integrates InternVL3's exact token replacement pattern for IMG_CONTEXT tokens
- Supports variable batch sizes and dynamic image tiling
- Maintains proper dependency chains through iterative latent token processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer
from collections import namedtuple
from typing import Optional, List, Union, Tuple, Dict, Any

# Define output structure following Coconut pattern
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])


def setup_special_tokens(tokenizer, model):
    """
    Add both Coconut and InternVL3 special tokens to tokenizer and model.
    
    This function handles the critical task of adding all required special tokens
    for multimodal Coconut operation, including proper embedding initialization.
    
    Args:
        tokenizer: AutoTokenizer instance to add tokens to
        model: Model instance to resize embeddings for
        
    Returns:
        Dictionary containing token IDs for all special tokens
    """
    # Define all special tokens needed for multimodal Coconut
    coconut_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]
    internvl_tokens = ["<img>", "</img>", "<IMG_CONTEXT>"]
    
    # Combine all special tokens
    all_special_tokens = coconut_tokens + internvl_tokens
    
    # Check which tokens are already present
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [token for token in all_special_tokens if token not in existing_tokens]
    
    if new_tokens:
        print(f"Adding {len(new_tokens)} new special tokens: {new_tokens}")
        
        # Add new special tokens to tokenizer
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        
        # Resize model embeddings to accommodate new tokens
        model.resize_token_embeddings(len(tokenizer))
        
        # Initialize new token embeddings with averaged embeddings from existing tokens
        # This follows Coconut's approach of using "the" token as reference
        with torch.no_grad():
            # Get embeddings for common tokens to average
            reference_tokens = ["the", "a", "an", "is", "are", "was", "were"]
            reference_ids = []
            
            for token in reference_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    reference_ids.append(token_id)
            
            if reference_ids:
                # Get input embeddings
                input_embeddings = model.get_input_embeddings()
                
                # Calculate average embedding from reference tokens
                reference_embeddings = input_embeddings.weight[reference_ids]
                avg_embedding = reference_embeddings.mean(dim=0)
                
                # Initialize new token embeddings
                for token in new_tokens:
                    token_id = tokenizer.convert_tokens_to_ids(token)
                    input_embeddings.weight[token_id] = avg_embedding.clone()
                
                # Also initialize output embeddings if they exist and are tied
                if hasattr(model, 'get_output_embeddings') and model.get_output_embeddings() is not None:
                    output_embeddings = model.get_output_embeddings()
                    if output_embeddings.weight.shape[0] == input_embeddings.weight.shape[0]:
                        for token in new_tokens:
                            token_id = tokenizer.convert_tokens_to_ids(token)
                            output_embeddings.weight[token_id] = avg_embedding.clone()
                
                print(f"Initialized embeddings for new tokens using averaged reference embeddings")
            else:
                print("Warning: Could not find reference tokens for embedding initialization")
    else:
        print("All required special tokens already present in tokenizer")
    
    # Get token IDs for all special tokens
    token_ids = {}
    for token in all_special_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            print(f"Warning: Token '{token}' not found in tokenizer vocabulary")
        token_ids[token] = token_id
    
    return token_ids


class MultimodalCoconut(nn.Module):
    """Extended Coconut model supporting vision-language continuous reasoning

    Based on analysis of coconut.py and InternVL3 architecture, this implementation
    maintains Coconut's multi-stage reasoning while following InternVL's exact
    token replacement pattern for multimodal fusion.

    Key implementation requirements:
    1. MUST support variable batch sizes (not restricted to batch_size=1)
    2. MUST handle variable number of image tiles per batch item
    3. MUST maintain exact Coconut reasoning logic with proper tensor operations
    4. MUST use InternVL3's exact token replacement pattern for IMG_CONTEXT tokens
    5. All tensor operations must be batch-aware and handle edge cases gracefully

    Critical tensor shapes to maintain:
    - input_ids: [batch_size, seq_len]
    - pixel_values: [total_tiles_in_batch, 3, 448, 448] (concatenated, not stacked)
    - image_flags: [total_tiles_in_batch] (boolean mask for valid tiles)
    - input_embeds: [batch_size, seq_len, hidden_size]
    """

    def __init__(self,
                 base_internvl_model,           # Complete InternVL3 model for delegation
                 tokenizer,                     # Tokenizer with special tokens
                 config,                        # Configuration object with model parameters
                 latent_token_id: Optional[int] = None,          # Token ID for <|latent|> tokens
                 start_latent_id: Optional[int] = None,          # Token ID for <|start-latent|> tokens
                 end_latent_id: Optional[int] = None,            # Token ID for <|end-latent|> tokens
                 eos_token_id: Optional[int] = None,             # Token ID for end-of-sequence
                 img_context_token_id: Optional[int] = None,     # Token ID for <IMG_CONTEXT> tokens
                 **kwargs):
        super().__init__()
        
        # CRITICAL: Model loading validation following InternVL pattern
        if base_internvl_model is None:
            raise ValueError("base_internvl_model cannot be None")
        
        if not hasattr(base_internvl_model, 'vision_model'):
            raise ValueError("Invalid InternVL model: missing vision_model component")
        
        if not hasattr(base_internvl_model, 'language_model'):
            raise ValueError("Invalid InternVL model: missing language_model component")
        
        if not hasattr(base_internvl_model, 'mlp1'):
            raise ValueError("Invalid InternVL model: missing mlp1 (visual projector) component")
        
        # Tokenizer validation
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None")
        
        # Configuration validation
        if config is None:
            raise ValueError("config cannot be None")
        
        # Store configuration and tokenizer
        self.config = config
        self.tokenizer = tokenizer
        
        # Store reference to base InternVL model for delegation
        self.base_internvl_model = base_internvl_model
        
        # Extract individual components with error handling
        try:
            self.vision_model = base_internvl_model.vision_model
            self.visual_projector = base_internvl_model.mlp1  # InternVL uses mlp1
            self.language_model = base_internvl_model.language_model
        except AttributeError as e:
            raise ValueError(f"Failed to extract model components: {e}. Ensure base_internvl_model is a valid InternVL model.")

        # Compute token IDs from tokenizer with validation (following Coconut pattern)
        try:
            if latent_token_id is None:
                latent_token_id = tokenizer.convert_tokens_to_ids("<|latent|>")
                if latent_token_id == tokenizer.unk_token_id:
                    print("Warning: <|latent|> token not found in tokenizer, using UNK token ID. Consider adding special tokens.")
            
            if start_latent_id is None:
                start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
                if start_latent_id == tokenizer.unk_token_id:
                    print("Warning: <|start-latent|> token not found in tokenizer, using UNK token ID. Consider adding special tokens.")
            
            if end_latent_id is None:
                end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
                if end_latent_id == tokenizer.unk_token_id:
                    print("Warning: <|end-latent|> token not found in tokenizer, using UNK token ID. Consider adding special tokens.")
            
            if eos_token_id is None:
                eos_token_id = tokenizer.eos_token_id
                if eos_token_id is None:
                    raise ValueError("Tokenizer missing eos_token_id. This is required for generation.")
            
            if img_context_token_id is None:
                img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
                if img_context_token_id == tokenizer.unk_token_id:
                    print("Warning: <IMG_CONTEXT> token not found in tokenizer, using UNK token ID. Consider adding InternVL special tokens.")
        
        except Exception as e:
            raise ValueError(f"Failed to compute token IDs from tokenizer: {e}. Ensure tokenizer has required special tokens.")

        # Coconut reasoning components (from coconut.py analysis)
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id

        # InternVL3 configuration parameters (with proper defaults)
        self.img_context_token_id = img_context_token_id
        self.num_image_token = getattr(config, 'num_image_token', 256)  # Default 256 tokens per tile
        self.downsample_ratio = getattr(config, 'downsample_ratio', 0.5)  # Default 0.5
        self.ps_version = getattr(config, 'ps_version', 'v2')  # Default v2
        self.select_layer = getattr(config, 'select_layer', -1)  # Default -1 (last layer)
        self.template = getattr(config, 'template', 'internvl2_5')  # Conversation template
        self.max_dynamic_patch = getattr(config, 'max_dynamic_patch', 12)  # Max tiles
        self.min_dynamic_patch = getattr(config, 'min_dynamic_patch', 1)   # Min tiles
        self.use_thumbnail = getattr(config, 'use_thumbnail', True)  # Thumbnail processing
        self.pad2square = getattr(config, 'pad2square', False)  # Padding strategy
        self.dynamic_image_size = getattr(config, 'dynamic_image_size', True)  # Dynamic resolution
        self.force_image_size = getattr(config, 'force_image_size', 448)  # Image size

        # Calculate patch size from config
        if hasattr(config, 'vision_config') and hasattr(config.vision_config, 'patch_size'):
            self.patch_size = config.vision_config.patch_size
        else:
            self.patch_size = 14  # Default InternVL3 patch size

        # Generation counter for FSDP synchronization (from Coconut)
        # Critical for distributed training - tracks forward passes across GPUs
        self.gen_forward_cnt = 0

        # Memory optimization features
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        if self.use_gradient_checkpointing:
            self.vision_model.gradient_checkpointing_enable()
            self.language_model.gradient_checkpointing_enable()

    def extract_feature(self, pixel_values: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Delegate to InternVL's built-in extract_feature method for visual feature extraction.
        
        This approach ensures exact compatibility with InternVL's behavior, including proper 
        select_layer support, spatial reshaping, ps_version handling, and scale_factor parameters.
        
        Args:
            pixel_values: Tensor of shape [total_tiles, 3, 448, 448]
            
        Returns:
            Visual features tensor of shape [total_tiles, num_image_token, hidden_size] or None
        """
        if pixel_values is None:
            return None
        
        # Validate input tensor
        if not isinstance(pixel_values, torch.Tensor):
            print(f"Warning: pixel_values must be torch.Tensor, got {type(pixel_values)}")
            return None
        
        if len(pixel_values.shape) != 4:
            print(f"Warning: pixel_values must be 4D tensor [tiles, 3, H, W], got shape {pixel_values.shape}")
            return None
        
        if pixel_values.shape[1] != 3:
            print(f"Warning: pixel_values must have 3 channels, got {pixel_values.shape[1]}")
            return None
        
        try:
            # Delegate directly to the base InternVL model's tested extract_feature method
            # This ensures we get exact InternVL behavior with all edge cases handled correctly
            vit_embeds = self.base_internvl_model.extract_feature(pixel_values)
            
            # Validate output
            if vit_embeds is not None:
                expected_shape = (pixel_values.shape[0], self.num_image_token, -1)
                if vit_embeds.shape[:2] != expected_shape[:2]:
                    print(f"Warning: Unexpected visual feature shape {vit_embeds.shape}, expected {expected_shape}")
            
            return vit_embeds
            
        except Exception as e:
            print(f"Error in visual feature extraction: {e}")
            return None

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: torch.Tensor,
                position_ids: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.Tensor] = None,
                image_flags: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with multimodal continuous reasoning and variable tile support

        Follows InternVL3's exact token replacement pattern with proper handling
        of variable-sized pixel_values from dynamic tiling.

        This is the main forward method. Critical requirements:
        1. MUST handle batch processing with variable image tile counts per batch item
        2. Input tensor shapes:
           - input_ids: [batch_size, seq_len]
           - pixel_values: [total_tiles_in_batch, 3, 448, 448] (concatenated across batch)
           - image_flags: [total_tiles_in_batch] (boolean mask for valid tiles)
        3. MUST follow InternVL3's exact IMG_CONTEXT token replacement pattern
        4. MUST maintain Coconut's continuous reasoning logic with proper batching
        5. Error handling for mismatched tensor shapes is essential
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for loss computation [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len] (optional)
            pixel_values: Image pixels [total_tiles, 3, 448, 448] (optional)
            image_flags: Valid tile mask [total_tiles] (optional)
            
        Returns:
            Dictionary containing loss, logits, and input embeddings
        """

        # Step 1: Get initial text embeddings (following InternVL3)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        # Step 2: Process visual inputs using InternVL's native image_flags mechanism
        if pixel_values is not None and image_flags is not None:
            try:
                # Validate input shapes
                if len(pixel_values.shape) != 4:
                    raise ValueError(f"pixel_values must be 4D tensor [total_tiles, 3, 448, 448], got shape {pixel_values.shape}")
                
                if len(image_flags.shape) != 1:
                    raise ValueError(f"image_flags must be 1D tensor [total_tiles], got shape {image_flags.shape}")
                
                if pixel_values.shape[0] != image_flags.shape[0]:
                    raise ValueError(f"pixel_values and image_flags must have same first dimension, got {pixel_values.shape[0]} vs {image_flags.shape[0]}")
                
                # Process all tiles through vision encoder (InternVL approach)
                vit_embeds = self.extract_feature(pixel_values)  # [total_tiles, num_image_token, hidden_size]
                
                if vit_embeds is None:
                    print("Warning: extract_feature returned None, skipping visual processing")
                    return self._coconut_continuous_reasoning(input_embeds, input_ids, attention_mask, labels, position_ids)
                
                # CRITICAL: Use InternVL's native image_flags filtering mechanism
                # This elegantly handles variable tile counts, text-only samples, and padding
                image_flags = image_flags.squeeze(-1)  # Remove any extra dimensions
                vit_embeds = vit_embeds[image_flags == 1]  # Filter out padding/text-only patches
                
            except Exception as e:
                print(f"Error in visual feature extraction: {e}. Skipping visual processing.")
                return self._coconut_continuous_reasoning(input_embeds, input_ids, attention_mask, labels, position_ids)
            
            # Apply InternVL's exact token replacement pattern
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids_flat = input_ids.reshape(B * N)
            
            # Find IMG_CONTEXT token positions
            selected = (input_ids_flat == self.img_context_token_id)
            
            if selected.any():
                try:
                    # CRITICAL: Use InternVL's exact pattern to preserve gradient flow
                    # Mathematical operations preserve gradients, direct assignment breaks them
                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
                except Exception as e:
                    # Handle shape mismatch with InternVL's fallback pattern
                    vit_embeds_flat = vit_embeds.reshape(-1, C)
                    n_token = selected.sum()
                    print(f'warning: {e}, selected.sum()={n_token}, vit_embeds.shape={vit_embeds.shape}')
                    input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds_flat[:n_token]
            
            # Reshape back to batch format
            input_embeds = input_embeds.reshape(B, N, C)

        # Step 3: Generate position IDs (following InternVL3's actual implementation)
        if position_ids is None:
            batch_size, seq_len = input_ids.shape
            # Standard sequential position IDs (proven InternVL approach)
            # Note: Despite V2PE being mentioned in InternVL3 paper, the actual codebase
            # uses standard sequential position encoding, which works excellently in practice
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Step 4: Apply Coconut's continuous reasoning to the multimodal sequence
        return self._coconut_continuous_reasoning(
            input_embeds, input_ids, attention_mask, labels, position_ids
        )

    def _coconut_continuous_reasoning(self, 
                                    inputs_embeds: torch.Tensor, 
                                    input_ids: torch.Tensor, 
                                    attention_mask: torch.Tensor, 
                                    labels: torch.Tensor, 
                                    position_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Implements Coconut's multi-pass continuous reasoning with proper chained latent token handling
        
        This method follows the original Coconut approach, using iterative multi-pass processing
        to handle chained latent tokens correctly. Each latent token's replacement uses hidden states that
        include the effects of all previously processed latent tokens, preserving dependency chains and
        enabling emergent BFS-like reasoning patterns.

        Critical Multi-Pass Requirements from Original Coconut:
        1. **Iterative Processing**: Process one latent token per pass to preserve dependencies
        2. **KV Cache Reuse**: Efficiently reuse computation from previous passes
        3. **Sequential Replacement**: Each latent uses hidden states from updated embeddings
        4. **Range Management**: Track computation ranges to minimize redundant forward passes
        5. **Dependency Preservation**: Maintain the chain of continuous thoughts for BFS reasoning
        
        Args:
            inputs_embeds: Input embeddings [batch_size, seq_len, hidden_size]
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            
        Returns:
            Dictionary containing loss, logits, and input embeddings
        """

        # Find latent token positions (following original Coconut)
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        
        if len(latent_indices) == 0:
            # No latent tokens, do standard forward pass
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
            )
            return {"loss": outputs.loss, "logits": outputs.logits, "inputs_embeds": inputs_embeds}

        # Organize latent positions by batch (exact Coconut logic)
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max([len(l) for l in latent_lists])
        
        # Initialize computation range (before earliest latent token)
        next_compute_range = (0, latent_indices[:, 1].min().item())
        kv_cache = None
        logits = []

        # Multi-pass iterative processing with corrected range management
        for pass_idx in range(max_n_latents):
            
            if kv_cache is None:
                # First forward pass - compute up to first latent token
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    output_hidden_states=True,
                    use_cache=True
                )
                hidden_states_offset = 0
            else:
                # Subsequent passes - reuse KV cache for efficiency
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True
                )
                hidden_states_offset = next_compute_range[0]

            # Extract hidden states and update KV cache
            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values
            logits.append(outputs.logits)

            # Replace latent tokens with preceding hidden states (Coconut core logic)
            for batch_idx in range(input_ids.shape[0]):
                if pass_idx < len(latent_lists[batch_idx]):
                    token_idx = latent_lists[batch_idx][pass_idx]
                    
                    # Use .detach() to prevent O(N²) memory growth in autograd graph
                    inputs_embeds[batch_idx, token_idx, :] = hidden_states[
                        batch_idx, token_idx - 1 - hidden_states_offset, :
                    ].detach()

            # Update computation range for next pass
            if pass_idx < max_n_latents - 1:
                # Find next latent token position
                remaining_latents = latent_indices[latent_indices[:, 1] > latent_lists[0][pass_idx]]
                if len(remaining_latents) > 0:
                    next_start = latent_lists[0][pass_idx] + 1
                    next_end = remaining_latents[0, 1].item()
                    next_compute_range = (next_start, next_end)
                else:
                    # No more latent tokens, compute to end
                    next_compute_range = (latent_lists[0][pass_idx] + 1, input_ids.shape[1])

        # Final forward pass from last latent to end of sequence
        if next_compute_range[1] < input_ids.shape[1]:
            past_key_values = [
                (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                for k, v in kv_cache
            ]
            
            final_outputs = self.language_model(
                inputs_embeds=inputs_embeds[:, next_compute_range[0]:, :],
                attention_mask=attention_mask[:, next_compute_range[0]:],
                labels=labels[:, next_compute_range[0]:] if labels is not None else None,
                position_ids=position_ids[:, next_compute_range[0]:],
                past_key_values=past_key_values,
            )
            
            logits.append(final_outputs.logits)
            final_loss = final_outputs.loss
        else:
            # Compute loss from accumulated logits
            if labels is not None:
                all_logits = torch.cat(logits, dim=1)
                loss_fct = CrossEntropyLoss()
                shift_logits = all_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                final_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                final_loss = None

        # Concatenate all logits
        all_logits = torch.cat(logits, dim=1) if logits else None

        return {
            "loss": final_loss,
            "logits": all_logits,
            "inputs_embeds": inputs_embeds
        }

    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 pixel_values: Optional[torch.Tensor] = None,
                 image_flags: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 50,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 synced_gpus: bool = False,
                 **kwargs) -> Dict[str, torch.Tensor]:
        """Generate responses with multimodal continuous reasoning support
        
        This method implements autoregressive generation with proper handling of
        dynamically generated latent tokens and KV cache management for efficiency.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            pixel_values: Image pixels [total_tiles, 3, 448, 448] (optional)
            image_flags: Valid tile mask [total_tiles] (optional)
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling or greedy decoding
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            synced_gpus: Whether to synchronize across GPUs for FSDP
            
        Returns:
            Dictionary containing generated sequences and attention masks
        """
        self.eval()
        
        # Validate inputs
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize generation state
        generated_ids = input_ids.clone()
        generated_attention_mask = attention_mask.clone()
        
        # Track completion status for each batch item
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # FSDP synchronization counter
        if synced_gpus:
            self.gen_forward_cnt = 0
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Increment forward counter for FSDP synchronization
                if synced_gpus:
                    self.gen_forward_cnt += 1
                
                # Prepare inputs for current step
                current_input_ids = generated_ids
                current_attention_mask = generated_attention_mask
                
                # Forward pass with multimodal continuous reasoning
                outputs = self.forward(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    labels=None,  # No labels during generation
                    pixel_values=pixel_values,
                    image_flags=image_flags
                )
                
                # Get logits for next token prediction
                logits = outputs["logits"][:, -1, :]  # [batch_size, vocab_size]
                
                # Apply temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Generate next tokens
                if do_sample:
                    # Apply top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        # Set logits to -inf for tokens to remove
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                    
                    # Sample from the filtered distribution
                    probs = F.softmax(logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(logits, dim=-1)
                
                # Handle dynamically generated latent tokens
                # If the model generates a latent token, we need to process it immediately
                latent_mask = (next_tokens == self.latent_token_id)
                if latent_mask.any():
                    # For latent tokens, we need to replace them with continuous thoughts
                    # This is handled automatically in the next forward pass
                    pass
                
                # Update finished status (check for EOS tokens)
                eos_mask = (next_tokens == self.eos_token_id)
                finished = finished | eos_mask
                
                # Mask out finished sequences (set to pad token)
                if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                    next_tokens = torch.where(finished, self.tokenizer.pad_token_id, next_tokens)
                else:
                    next_tokens = torch.where(finished, self.eos_token_id, next_tokens)
                
                # Append new tokens to sequences
                generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(1)], dim=1)
                
                # Update attention mask
                new_attention = torch.where(finished.unsqueeze(1), 0, 1)
                generated_attention_mask = torch.cat([generated_attention_mask, new_attention], dim=1)
                
                # Check if all sequences are finished
                if finished.all():
                    break
                
                # FSDP synchronization point
                if synced_gpus and step % 10 == 0:
                    # Synchronize across GPUs every 10 steps
                    torch.distributed.barrier()
        
        return {
            "sequences": generated_ids,
            "attention_mask": generated_attention_mask,
            "num_generated_tokens": generated_ids.shape[1] - seq_len
        }