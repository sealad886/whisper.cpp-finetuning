# Converting LoRA-fine-tuned Whisper Model to whisper.cpp Format

This guide explains how to use your PEFT LoRA adapter with whisper.cpp.

## Background

**whisper.cpp** is a C++ implementation of OpenAI's Whisper that uses the GGML format for efficient inference. It **does not natively support LoRA adapters**, so you need to:

1. ✅ Merge the LoRA adapter with the base model (creates a full model)
2. ✅ Convert the merged model to GGML format

## Prerequisites

You already have the required dependencies in `requirements.txt`:
- `torch`
- `peft`
- `transformers`
- `accelerate`

## Quick Start

### Step 1: Merge LoRA Adapter with Base Model

Run the provided script:

```bash
python3 merge_lora_and_convert.py \
    --adapter whisper-turbo-lora-out/checkpoint-50 \
    --output whisper-turbo-merged \
    --device cpu
```

**Options:**
- `--device cpu`: Use CPU (slower but works everywhere)
- `--device mps`: Use Apple Silicon GPU (M1/M2/M3 Macs - much faster!)
- `--device cuda`: Use NVIDIA GPU

This creates a `whisper-turbo-merged/` directory with the full merged model.

### Step 2 (Option A - automatic): Merge + Convert in one command

If you already have both repositories cloned locally, you can produce the GGML model in a single step:

```bash
python3 merge_lora_and_convert.py \
    --adapter whisper-turbo-lora-out/checkpoint-50 \
    --output whisper-turbo-merged \
    --device cpu \
    --convert \
    --whisper-cpp-dir /path/to/whisper.cpp \
    --whisper-repo-dir /path/to/openai/whisper \
    --ggml-out /path/to/whisper.cpp/models \  # optional, defaults to <whisper-cpp-dir>/models
    --use-f32 \                               # optional, export in float32 instead of float16
    --quantize q5_0                           # optional, quantize after conversion (e.g., q5_0)
```

Flags:
- `--convert` run the whisper.cpp conversion right after merge
- `--whisper-cpp-dir` absolute path to your whisper.cpp repo
- `--whisper-repo-dir` absolute path to your OpenAI whisper repo clone
- `--ggml-out` optional output directory for the GGML file (defaults to `<whisper-cpp-dir>/models`)
- `--use-f32` export in float32 instead of float16 (bigger, sometimes more accurate)
- `--quantize` run whisper.cpp quantizer automatically with your chosen method (examples: `q5_0`, `q4_0`, `q6_k`)

The output will be:
- `whisper-turbo-merged/` (HF format, for reference)
- `ggml-model.bin` or `ggml-model-f32.bin` in `--ggml-out`
 - Optionally, a quantized file such as `ggml-model-q5_0.bin` in the same directory

Note on paths:
- The converter writes to the `models` folder of the whisper.cpp repo you pass via `--whisper-cpp-dir` by default, or to the `--ggml-out` directory if provided.
- The script prints the exact output path(s) when it finishes. Use those paths with `whisper-cli`.

### Step 2 (Option B - manual): Set Up whisper.cpp

If you haven't already:

```bash
# Clone whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp

# Build it
make

# Clone OpenAI Whisper repo (needed for conversion scripts)
cd models
git clone https://github.com/openai/whisper
cd ..
```

### Step 3: Convert to GGML Format

From the whisper.cpp directory:

```bash
python3 models/convert-h5-to-ggml.py \
    /Volumes/scritch/CHAINS/whisper-turbo-merged/ \
    models/whisper/ \
    ./models/
```

This creates `models/ggml-model.bin` - your whisper.cpp-compatible model!

### Step 4: Use Your Model

```bash
./build/bin/whisper-cli -m models/ggml-model.bin -f your_audio.wav
```

Smoke test example (after conversion/quantization):

```bash
# Use the quantized model on the provided sample
./build/bin/whisper-cli \
    -m models/ggml-model-q5_0.bin \
    -f samples/jfk.wav \
    -otxt -of test_out
```
This should produce `test_out.txt` with the well-known JFK quote.

## Manual Method (If You Want to Understand the Process)

### 1. Merge Adapter Manually

```python
import torch
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Load PEFT config to get base model path
peft_config = PeftConfig.from_pretrained("whisper-turbo-lora-out/checkpoint-50")

# Load base model
model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path,  # Should be "openai/whisper-large-v3-turbo"
    torch_dtype=torch.float16,
    device_map="cpu"  # or "mps" for Mac, "cuda" for NVIDIA
)

# Load and merge LoRA adapter
model = PeftModel.from_pretrained(model, "whisper-turbo-lora-out/checkpoint-50")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("whisper-turbo-merged")

# Save processor/tokenizer
processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path)
processor.save_pretrained("whisper-turbo-merged")
```

### 2. Convert to GGML

Follow Step 3 above.

## Understanding the Files

### Your LoRA Checkpoint Contains:
- `adapter_config.json` - LoRA configuration
- `adapter_model.safetensors` - LoRA weights (small, only the adapters)
- Other tokenizer/config files

### After Merging:
- `model.safetensors` - Full model weights (base + LoRA merged)
- All tokenizer and config files needed for conversion

### After GGML Conversion:
- `ggml-model.bin` - Single file for whisper.cpp (efficient binary format)

## Performance Tips

1. **Use Apple Silicon GPU on Mac:**
   ```bash
   python3 merge_lora_and_convert.py --device mps
   ```

2. **Quantize the GGML model** for smaller size and faster inference:
   ```bash
   # After conversion, quantize (in whisper.cpp directory)
   ./build/bin/quantize models/ggml-model.bin models/ggml-model-q5_0.bin q5_0
   ```

You can also let the merge script do this automatically with `--quantize q5_0`.

3. **Use threads for faster transcription:**
   ```bash
   ./build/bin/whisper-cli -m models/ggml-model.bin -f audio.wav -t 4
   ```

## Troubleshooting

### "RuntimeError: No CUDA GPUs are available"
Use `--device cpu` or `--device mps` (Mac) instead.

### "Where are the mel filters?"
The mel filters are **not stored in the model files**. They are computed on-the-fly from the parameters in `preprocessor_config.json`:
- `feature_size`: Number of mel bins (usually 80 or 128)
- `n_fft`: FFT window size
- `hop_length`: Hop length
- `sampling_rate`: Audio sampling rate

During whisper.cpp conversion, the script loads pre-computed mel filters from the OpenAI Whisper repository's `mel_filters.npz` file. This is why you need to clone the OpenAI Whisper repo for conversion.

**To verify your model has the necessary parameters:**
```bash
python3 verify_conversion_ready.py whisper-turbo-merged
```

### "Out of memory" during merge
The large-v3-turbo model is ~1.5GB. Try:
- Using CPU with sufficient RAM
- Closing other applications
- Using float32 instead of float16 (edit the script)

### Conversion script not found
Make sure you cloned the OpenAI Whisper repo into `whisper.cpp/models/whisper/`

### Model not transcribing correctly
- Ensure you're using the right checkpoint (checkpoint-50 seems to be your best)
- Check if your training converged properly
- Test with audio similar to your training data

## Additional Resources

- [whisper.cpp Repository](https://github.com/ggerganov/whisper.cpp)
- [whisper.cpp Models README](https://github.com/ggerganov/whisper.cpp/tree/master/models)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Whisper Fine-tuning Guide](https://huggingface.co/blog/fine-tune-whisper)

## Alternative: Use the Merged Model Directly

If you don't need whisper.cpp specifically, you can use the merged model directly with transformers:

```python
from transformers import pipeline

# After merging
pipe = pipeline("automatic-speech-recognition", model="whisper-turbo-merged")
result = pipe("audio.wav")
print(result["text"])
```

This is easier but requires Python and more memory/compute than whisper.cpp.
