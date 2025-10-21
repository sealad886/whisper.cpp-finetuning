# Getting Started

This guide walks you through fine-tuning Whisper models with LoRA/DoRA adapters and exporting them for [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp).

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU with at least 12 GB VRAM for medium models.
- [ffmpeg](https://ffmpeg.org/download.html) installed and available on `PATH`.
- Access to training audio/transcript pairs (CSV manifest).

## Installation

```bash
git clone https://github.com/your-org/whisper.cpp-finetuning.git
cd whisper.cpp-finetuning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare Manifests

Use the helper script to convert a directory of audio files into a CSV manifest compatible with the training pipeline:

```bash
python build_manifest_from_names.py \
  --audio-root /path/to/audio \
  --transcript-root /path/to/transcripts \
  --output-manifest data/train.csv
```

Repeat the process for validation data. Each CSV must contain at least the `path` and `text` columns.

## Run Training

Edit `configs/train_whisper_lora.json` to match your dataset and hardware. Then launch training:

```bash
python train_whisper_lora.py --config configs/train_whisper_lora.json
```

Key config knobs:

- `base_model`: Whisper checkpoint from Hugging Face (`openai/whisper-medium.en`, etc.).
- `lora_rank`: Adapter rank, balancing accuracy vs. parameter count.
- `merge_lora`: Automatically merge adapters after training for export.

## Merge & Export

After training completes, generate a merged checkpoint for deployment:

```bash
python merge_lora_and_convert.py \
  --experiment-dir outputs/your-run \
  --whisper-cpp-repo ../whisper.cpp \
  --output-dir exports/your-run
```

Verify the exported model with the provided sanity checks:

```bash
python test_merged_model.py --export-dir exports/your-run
python verify_conversion_ready.py --export-dir exports/your-run
```

The resulting `.bin` files drop into `whisper.cpp`'s `models/` directory for immediate inference.
