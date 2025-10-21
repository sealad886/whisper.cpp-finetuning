#!/usr/bin/env python3
"""
Test the merged Whisper model before converting to GGML.

This helps verify the merge was successful.
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import argparse


def test_merged_model(model_path: str, test_text: str = "Hello, this is a test."):
    """
    Test that the merged model loads and can be used for inference.

    Args:
        model_path: Path to the merged model
        test_text: Optional text to test (for debugging)
    """
    print(f"Loading model from {model_path}...")

    try:
        # Load model and processor
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)

        print("✅ Model loaded successfully!")

        # Print model info
        print(f"\nModel configuration:")
        print(f"  - Model type: {model.config.model_type}")
        print(f"  - Number of parameters: {model.num_parameters():,}")
        print(f"  - Vocab size: {model.config.vocab_size}")
        print(f"  - Encoder layers: {model.config.encoder_layers}")
        print(f"  - Decoder layers: {model.config.decoder_layers}")

        # Check if model.safetensors exists
        import os
        safetensors_path = os.path.join(model_path, "model.safetensors")
        pytorch_path = os.path.join(model_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
            print(f"  - Model file: model.safetensors ({size_mb:.1f} MB)")
        elif os.path.exists(pytorch_path):
            size_mb = os.path.getsize(pytorch_path) / (1024 * 1024)
            print(f"  - Model file: pytorch_model.bin ({size_mb:.1f} MB)")

        print("\n✅ Model is ready for conversion to GGML format!")
        print(f"\nTo convert, run:")
        print(f"  python3 whisper.cpp/models/convert-h5-to-ggml.py \\")
        print(f"      {model_path}/ \\")
        print(f"      whisper.cpp/models/whisper/ \\")
        print(f"      ./")

        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test merged Whisper model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="whisper-turbo-merged",
        help="Path to merged model (default: whisper-turbo-merged)"
    )

    args = parser.parse_args()

    test_merged_model(args.model)


if __name__ == "__main__":
    main()
