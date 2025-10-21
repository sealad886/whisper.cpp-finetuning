#!/usr/bin/env python3
"""
Merge PEFT LoRA adapter with base Whisper model and prepare for whisper.cpp conversion.

This script:
1. Loads the base Whisper model
2. Loads and merges the LoRA adapter weights
3. Saves the merged model in HuggingFace format
4. Provides instructions for converting to GGML format
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Any, Optional


def merge_lora_adapter(
    adapter_path: str,
    output_path: str,
    device: str = "cpu",
    convert: bool = False,
    whisper_cpp_dir: Optional[str] = None,
    whisper_repo_dir: Optional[str] = None,
    ggml_out: Optional[str] = None,
    use_f32: bool = False,
    quantize: Optional[str] = None,
):
    """
    Merge PEFT LoRA adapter with base Whisper model.

    Args:
        adapter_path: Path to the LoRA adapter checkpoint
        output_path: Path to save the merged model
        device: Device to load model on ('cpu', 'cuda', or 'mps' for Mac)
    """
    # Lazy-import heavy deps to allow --help without ML stack installed
    from peft import PeftModel, PeftConfig
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    import torch

    print(f"Loading PEFT config from {adapter_path}...")
    peft_config = PeftConfig.from_pretrained(adapter_path)

    print(f"Base model: {peft_config.base_model_name_or_path}")
    print(f"Loading base model...")

    # Choose dtype based on device to avoid CPU fp16 pitfalls
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    # Load base model with appropriate dtype
    model = WhisperForConditionalGeneration.from_pretrained(
        peft_config.base_model_name_or_path,
        dtype=dtype,
    )
    if device != "cpu":
        model.to(device)

    print(f"Loading LoRA adapter from {adapter_path}...")
    model_peft: Any = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter weights with base model...")
    merged_model = model_peft.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)

    # Also save the processor/tokenizer (includes preprocessor_config.json with mel filter parameters)
    print("Saving processor/tokenizer...")
    processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)
    processor.save_pretrained(output_path)

    # Verify preprocessor_config.json was saved
    import json
    config_path = os.path.join(output_path, "preprocessor_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Preprocessor config saved (feature_size: {config.get('feature_size', 'unknown')} mel bins)")
    else:
        print("⚠ Warning: preprocessor_config.json not found!")

    print("\n✅ Successfully merged and saved model!")
    print(f"\nMerged model saved to: {output_path}")

    # Optional: run conversion to GGML via whisper.cpp
    if convert:
        print("\n▶ Converting merged model to GGML using whisper.cpp …")
        if not whisper_cpp_dir:
            raise ValueError("--whisper-cpp-dir is required when --convert is set")
        if not whisper_repo_dir:
            raise ValueError("--whisper-repo-dir is required when --convert is set")

        wcpp = Path(whisper_cpp_dir).expanduser().resolve()
        wrepo = Path(whisper_repo_dir).expanduser().resolve()
        out_dir = Path(ggml_out or (wcpp / "models")).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        conv_script = wcpp / "models" / "convert-h5-to-ggml.py"
        if not conv_script.exists():
            raise FileNotFoundError(f"convert-h5-to-ggml.py not found at {conv_script}")
        if not (wrepo / "whisper" / "assets" / "mel_filters.npz").exists():
            # Basic sanity check that this looks like the OpenAI Whisper repo
            raise FileNotFoundError(
                f"mel_filters.npz not found under {wrepo}/whisper/assets. "
                "Ensure --whisper-repo-dir points to a clone of github.com/openai/whisper"
            )

        cmd = [
            sys.executable or "python3",
            str(conv_script),
            str(Path(output_path).resolve()),
            str(wrepo.resolve()),
            str(out_dir),
        ]
        if use_f32:
            cmd.append("use-f32")

        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(wcpp))

        # By default the converter writes ggml-model.bin (or ggml-model-f32.bin)
        produced = out_dir / ("ggml-model-f32.bin" if use_f32 else "ggml-model.bin")
        if produced.exists():
            print(f"\n✅ GGML model created: {produced}")
            print("You can use it with whisper.cpp, e.g.:")
            print(f"  {wcpp / 'build' / 'bin' / 'whisper-cli'} -m {produced} -f your_audio.wav")
            # Optional quantization
            if quantize:
                quant_bin = wcpp / 'build' / 'bin' / 'quantize'
                if not quant_bin.exists():
                    print("\n\u26a0 Quantize binary not found. Please build whisper.cpp (make) to enable quantization.")
                else:
                    q_out = out_dir / f"ggml-model-{quantize}.bin"
                    q_cmd = [str(quant_bin), str(produced), str(q_out), quantize]
                    print("\nQuantizing:", " ".join(q_cmd))
                    subprocess.check_call(q_cmd, cwd=str(wcpp))
                    if q_out.exists():
                        print(f"\n\u2705 Quantized model created: {q_out}")
                    else:
                        print("\n\u26a0 Quantization finished but output not found:", q_out)
        else:
            print("\n⚠ Conversion completed but expected GGML file not found:", produced)
            print("Check converter output for details.")
    else:
        print("\nNext steps (manual conversion):")
        print("1) whisper.cpp setup (one-time):")
        print("   git clone https://github.com/ggerganov/whisper.cpp && cd whisper.cpp && make")
        print("   cd models && git clone https://github.com/openai/whisper && cd ..")
        print("2) Convert to GGML:")
        print(f"   python3 models/convert-h5-to-ggml.py {Path(output_path).resolve()}/ models/whisper/ ./models/")
        print("3) Run:")
        print("   ./build/bin/whisper-cli -m models/ggml-model.bin -f your_audio.wav")


def main():
    parser = argparse.ArgumentParser(
        description="Merge PEFT LoRA adapter with Whisper base model"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="whisper-turbo-lora-out/checkpoint-50",
        help="Path to LoRA adapter checkpoint (default: whisper-turbo-lora-out/checkpoint-50)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="whisper-turbo-merged",
        help="Path to save merged model (default: whisper-turbo-merged)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use (default: cpu). Use 'mps' for Mac M1/M2, 'cuda' for NVIDIA GPU"
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="After merge, convert to GGML using whisper.cpp conversion script"
    )
    parser.add_argument(
        "--whisper-cpp-dir",
        type=str,
        help="Path to whisper.cpp repository (required with --convert)"
    )
    parser.add_argument(
        "--whisper-repo-dir",
        type=str,
        help="Path to OpenAI Whisper repository clone (required with --convert)"
    )
    parser.add_argument(
        "--ggml-out",
        type=str,
        help="Directory to write ggml-model.bin (default: <whisper-cpp-dir>/models)"
    )
    parser.add_argument(
        "--use-f32",
        action="store_true",
        help="Export GGML in float32 instead of float16"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=[
            "q8_0","q6_k","q5_k_m","q5_k_s","q5_0","q4_k_m","q4_k_s","q4_0","q3_k_m","q3_k_s","q2_k"],
        help="Quantize the GGML model with the given method (e.g., q5_0)"
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    merge_lora_adapter(
        adapter_path=args.adapter,
        output_path=args.output,
        device=args.device,
        convert=args.convert,
        whisper_cpp_dir=args.whisper_cpp_dir,
        whisper_repo_dir=args.whisper_repo_dir,
        ggml_out=args.ggml_out,
        use_f32=args.use_f32,
        quantize=args.quantize,
    )


if __name__ == "__main__":
    main()
