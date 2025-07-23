#!/usr/bin/env python3
"""
Test script for MultimodalCoconut model implementation

This script tests the core functionality of the MultimodalCoconut model
to ensure it works as expected before proceeding with full implementation.
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing imports...")
    try:
        from multicoco import MultimodalCoconut, setup_special_tokens, MultiCoCoConfig
        from transformers import AutoTokenizer
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_configuration():
    """Test configuration loading and validation."""
    print("\n🧪 Testing configuration...")
    try:
        from multicoco import MultiCoCoConfig
        
        # Test default configuration
        config = MultiCoCoConfig()
        print(f"✅ Default config created: {config.model_id}")
        
        # Test YAML loading
        config_yaml = MultiCoCoConfig.from_yaml('configs/aokvqa.yaml')
        print(f"✅ YAML config loaded: {config_yaml.dataset_name}")
        
        # Test type validation
        assert isinstance(config_yaml.learning_rate, float), "learning_rate should be float"
        assert isinstance(config_yaml.batch_size, int), "batch_size should be int"
        assert isinstance(config_yaml.use_flash_attn, bool), "use_flash_attn should be bool"
        print("✅ Type validation passed")
        
        return True, config_yaml
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False, None

def test_special_tokens():
    """Test special token management."""
    print("\n🧪 Testing special token management...")
    try:
        from multicoco import setup_special_tokens
        from transformers import AutoTokenizer
        
        # Create a simple tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        
        # Create a dummy model with embeddings
        class DummyModel:
            def __init__(self):
                self.embeddings = torch.nn.Embedding(tokenizer.vocab_size, 768)
            
            def get_input_embeddings(self):
                return self.embeddings
            
            def get_output_embeddings(self):
                return None
            
            def resize_token_embeddings(self, new_size):
                old_embeddings = self.embeddings
                self.embeddings = torch.nn.Embedding(new_size, 768)
                # Copy old embeddings
                with torch.no_grad():
                    self.embeddings.weight[:old_embeddings.num_embeddings] = old_embeddings.weight
        
        dummy_model = DummyModel()
        original_vocab_size = len(tokenizer)
        
        # Test token setup
        token_ids = setup_special_tokens(tokenizer, dummy_model)
        
        # Verify tokens were added
        new_vocab_size = len(tokenizer)
        print(f"✅ Vocabulary expanded from {original_vocab_size} to {new_vocab_size}")
        
        # Check specific tokens
        required_tokens = ["<|start-latent|>", "<|end-latent|>", "<|latent|>", "<img>", "</img>", "<IMG_CONTEXT>"]
        for token in required_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            assert token_id != tokenizer.unk_token_id, f"Token {token} not properly added"
            assert token in token_ids, f"Token {token} not in returned token_ids"
        
        print("✅ All required tokens properly added")
        return True, tokenizer, dummy_model, token_ids
        
    except Exception as e:
        print(f"❌ Special token test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None

def test_model_initialization():
    """Test MultimodalCoconut model initialization with real InternVL3-1B-Pretrained."""
    print("\n🧪 Testing model initialization with real InternVL3-1B-Pretrained...")
    try:
        from multicoco import MultimodalCoconut, MultiCoCoConfig, setup_special_tokens
        from transformers import AutoModel, AutoTokenizer
        
        print("   Loading InternVL3-1B-Pretrained model (this may take a moment)...")
        
        # Load real InternVL3 model and tokenizer
        model_id = "OpenGVLab/InternVL3-1B-Pretrained"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False
        )
        print("   ✅ Tokenizer loaded")
        
        # Load base InternVL model
        base_internvl_model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        print("   ✅ Base InternVL3 model loaded")
        
        # Verify model components
        assert hasattr(base_internvl_model, 'vision_model'), "Missing vision_model"
        assert hasattr(base_internvl_model, 'language_model'), "Missing language_model"
        assert hasattr(base_internvl_model, 'mlp1'), "Missing mlp1 (visual projector)"
        print("   ✅ Model components verified")
        
        # Setup special tokens
        token_ids = setup_special_tokens(tokenizer, base_internvl_model)
        print("   ✅ Special tokens added")
        
        # Create configuration
        config = MultiCoCoConfig()
        
        # Test MultimodalCoconut initialization
        model = MultimodalCoconut(
            base_internvl_model=base_internvl_model,
            tokenizer=tokenizer,
            config=config,
            **token_ids
        )
        
        print("✅ MultimodalCoconut model initialized successfully")
        print(f"   - Model device: {next(model.parameters()).device}")
        print(f"   - Model dtype: {next(model.parameters()).dtype}")
        print(f"   - Latent token ID: {model.latent_token_id}")
        print(f"   - IMG_CONTEXT token ID: {model.img_context_token_id}")
        print(f"   - Vocabulary size: {len(tokenizer)}")
        print(f"   - Vision model: {type(model.vision_model).__name__}")
        print(f"   - Language model: {type(model.language_model).__name__}")
        
        return True, model, tokenizer
        
    except Exception as e:
        print(f"❌ Model initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_forward_pass():
    """Test model forward pass with real data."""
    print("\n🧪 Testing forward pass...")
    try:
        # Get model from previous test
        success, model, tokenizer = test_model_initialization()
        if not success:
            return False
        
        # Get model device and dtype
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        # Create input data on correct device
        batch_size = 2
        seq_len = 10
        
        # Create input tensors on correct device
        input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        labels = input_ids.clone()
        
        # Add some latent tokens to test continuous reasoning
        input_ids[0, 5] = model.latent_token_id  # Add latent token
        input_ids[1, 3] = model.latent_token_id  # Add latent token
        
        print("   Testing text-only forward pass...")
        # Test text-only forward pass
        outputs = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        print("✅ Text-only forward pass successful")
        print(f"   - Loss: {outputs['loss']}")
        print(f"   - Logits shape: {outputs['logits'].shape}")
        print(f"   - Logits device: {outputs['logits'].device}")
        
        print("   Testing multimodal forward pass...")
        # Test multimodal forward pass
        num_tiles = 2  # Reduce for memory efficiency
        pixel_values = torch.randn(num_tiles, 3, 448, 448, device=device, dtype=dtype)
        image_flags = torch.ones(num_tiles, dtype=torch.bool, device=device)
        
        # Add IMG_CONTEXT tokens to input (more realistic placement)
        input_ids_mm = input_ids.clone()
        # Replace some tokens with IMG_CONTEXT tokens
        input_ids_mm[0, 1:3] = model.img_context_token_id  # 2 IMG_CONTEXT tokens for first batch
        input_ids_mm[1, 1:2] = model.img_context_token_id  # 1 IMG_CONTEXT token for second batch
        
        outputs_mm = model.forward(
            input_ids=input_ids_mm,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_flags=image_flags
        )
        
        print("✅ Multimodal forward pass successful")
        print(f"   - Loss: {outputs_mm['loss']}")
        print(f"   - Logits shape: {outputs_mm['logits'].shape}")
        print(f"   - Visual features processed: {num_tiles} tiles")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation():
    """Test model generation capabilities."""
    print("\n🧪 Testing generation...")
    try:
        # Get model from previous test
        success, model, tokenizer = test_model_initialization()
        if not success:
            return False
        
        # Get model device
        device = next(model.parameters()).device
        
        # Create input for generation on correct device
        batch_size = 1
        seq_len = 5
        
        input_ids = torch.randint(0, len(tokenizer), (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        print("   Testing text-only generation...")
        # Test generation
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=5,  # Reduce for faster testing
            do_sample=False  # Greedy decoding for deterministic test
        )
        
        print("✅ Text-only generation successful")
        print(f"   - Original length: {seq_len}")
        print(f"   - Generated length: {generated['sequences'].shape[1]}")
        print(f"   - New tokens: {generated['num_generated_tokens']}")
        
        # Test multimodal generation
        print("   Testing multimodal generation...")
        
        # Create input with IMG_CONTEXT tokens
        input_ids_mm = input_ids.clone()
        input_ids_mm[0, 1] = model.img_context_token_id  # Add IMG_CONTEXT token
        
        # Create visual input
        num_tiles = 1
        pixel_values = torch.randn(num_tiles, 3, 448, 448, device=device, dtype=next(model.parameters()).dtype)
        image_flags = torch.ones(num_tiles, dtype=torch.bool, device=device)
        
        generated_mm = model.generate(
            input_ids=input_ids_mm,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_flags=image_flags,
            max_new_tokens=5,
            do_sample=False
        )
        
        print("✅ Multimodal generation successful")
        print(f"   - Generated length: {generated_mm['sequences'].shape[1]}")
        print(f"   - New tokens: {generated_mm['num_generated_tokens']}")
        print(f"   - Visual tiles processed: {num_tiles}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting MultimodalCoconut Model Tests\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", lambda: test_configuration()[0]),
        ("Special Tokens", lambda: test_special_tokens()[0]),
        ("Model Initialization", lambda: test_model_initialization()[0]),
        ("Forward Pass", test_forward_pass),
        ("Generation", test_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! MultimodalCoconut model is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)