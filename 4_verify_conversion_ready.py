#!/usr/bin/env python3
"""
Verify that a Whisper model directory has all necessary files for whisper.cpp conversion.
"""

import os
import json
import argparse
from pathlib import Path


def verify_model_directory(model_path: str) -> tuple[bool, list[str]]:
    """Verify model directory has all necessary files for conversion."""
    model_dir = Path(model_path)
    issues = []
    
    print(f"Verifying model directory: {model_dir}")
    print("=" * 60)
    
    # Check model weights
    print("\n1. Checking model weights...")
    safetensors = model_dir / "model.safetensors"
    pytorch_bin = model_dir / "pytorch_model.bin"
    
    if safetensors.exists():
        size_mb = safetensors.stat().st_size / (1024 * 1024)
        print(f"   ✓ model.safetensors found ({size_mb:.1f} MB)")
    elif pytorch_bin.exists():
        size_mb = pytorch_bin.stat().st_size / (1024 * 1024)
        print(f"   ✓ pytorch_model.bin found ({size_mb:.1f} MB)")
    else:
        issues.append("Missing model weights (model.safetensors or pytorch_model.bin)")
        print("   ✗ No model weights found")
    
    # Check config.json
    print("\n2. Checking config.json...")
    config_file = model_dir / "config.json"
    if config_file.exists():
        print("   ✓ config.json found")
        try:
            with open(config_file) as f:
                config = json.load(f)
            required_keys = ["model_type", "vocab_size", "encoder_layers", "decoder_layers"]
            missing = [k for k in required_keys if k not in config]
            if missing:
                issues.append(f"config.json missing keys: {missing}")
                print(f"   ⚠ Missing keys: {missing}")
            else:
                print(f"   ✓ Model type: {config.get('model_type')}")
                print(f"   ✓ Encoder layers: {config.get('encoder_layers')}")
                print(f"   ✓ Decoder layers: {config.get('decoder_layers')}")
        except Exception as e:
            issues.append(f"Failed to parse config.json: {e}")
            print(f"   ✗ Failed to parse: {e}")
    else:
        issues.append("Missing config.json")
        print("   ✗ config.json not found")
    
    # Check preprocessor_config.json (CRITICAL for conversion)
    print("\n3. Checking preprocessor_config.json...")
    preprocessor_file = model_dir / "preprocessor_config.json"
    if preprocessor_file.exists():
        print("   ✓ preprocessor_config.json found")
        try:
            with open(preprocessor_file) as f:
                preproc_config = json.load(f)
            
            # Check critical parameters for mel filter computation
            required = {
                "feature_size": "number of mel bins",
                "n_fft": "FFT window size",
                "hop_length": "hop length",
                "sampling_rate": "sampling rate"
            }
            
            all_present = True
            for key, desc in required.items():
                if key in preproc_config:
                    print(f"   ✓ {key}: {preproc_config[key]} ({desc})")
                else:
                    issues.append(f"preprocessor_config.json missing {key} ({desc})")
                    print(f"   ✗ Missing {key} ({desc})")
                    all_present = False
            
            if all_present:
                print("\n   ✓ All mel filter parameters present!")
                print("   (Mel filters will be computed from these parameters during conversion)")
        except Exception as e:
            issues.append(f"Failed to parse preprocessor_config.json: {e}")
            print(f"   ✗ Failed to parse: {e}")
    else:
        issues.append("Missing preprocessor_config.json - REQUIRED for conversion!")
        print("   ✗ preprocessor_config.json not found")
        print("   ✗ This file is REQUIRED for whisper.cpp conversion!")
    
    # Check tokenizer files
    print("\n4. Checking tokenizer files...")
    tokenizer_files = ["tokenizer_config.json", "vocab.json", "merges.txt"]
    missing_tokenizer = []
    for fname in tokenizer_files:
        if (model_dir / fname).exists():
            print(f"   ✓ {fname} found")
        else:
            missing_tokenizer.append(fname)
            print(f"   ⚠ {fname} not found (may cause issues)")
    
    if missing_tokenizer:
        issues.append(f"Missing tokenizer files: {missing_tokenizer}")
    
    # Summary
    print("\n" + "=" * 60)
    if not issues:
        print("✅ SUCCESS: Model directory is ready for conversion!")
        print("\nNext steps:")
        print("1. Clone whisper.cpp and OpenAI Whisper repo (if not done):")
        print("   git clone https://github.com/ggerganov/whisper.cpp")
        print("   cd whisper.cpp/models")
        print("   git clone https://github.com/openai/whisper")
        print("   cd ../..")
        print("\n2. Convert to GGML:")
        print(f"   cd whisper.cpp")
        print(f"   python3 models/convert-h5-to-ggml.py \\")
        print(f"       {model_dir.absolute()}/ \\")
        print(f"       models/whisper/ \\")
        print(f"       ./models/")
        print("\n3. Test:")
        print("   ./build/bin/whisper-cli -m models/ggml-model.bin -f audio.wav")
        return True, []
    else:
        print(f"❌ FAILED: Found {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\nPlease fix these issues before attempting conversion.")
        return False, issues


def main():
    parser = argparse.ArgumentParser(
        description="Verify Whisper model directory is ready for whisper.cpp conversion"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to model directory to verify"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.model_path):
        print(f"Error: {args.model_path} is not a directory")
        return 1
    
    is_valid, issues = verify_model_directory(args.model_path)
    return 0 if is_valid else 1


if __name__ == "__main__":
    exit(main())
