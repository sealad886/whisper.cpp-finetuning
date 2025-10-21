#!/usr/bin/env python3
# train_whisper_lora.py
# LoRA fine-tuning for openai/whisper-large-v3-turbo using CSV manifests from build_manifest.py
#
# Usage:
#   python train_whisper_lora.py \
#     --train_csv ./manifests/train.csv \
#     --val_csv ./manifests/val.csv \
#     --outdir ./whisper-turbo-lora-out \
#     --language en --epochs 5 --bf16 --merge_lora --freeze_encoder

import argparse, json, os, sys
import pandas as pd
import torch
from dataclasses import dataclass
from types import MethodType
from datasets import Dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora.config import LoraRuntimeConfig
from peft.peft_model import PeftModel
from peft.mixed_model import PeftMixedModel
from jiwer import wer

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object with key/value pairs")
    return data


def _maybe_parse_json(value, *, arg_name: str):
    if value is None:
        return None
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON for {arg_name}: {exc}") from exc
    raise ValueError(f"Unsupported type for {arg_name}: {type(value)!r}")


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", help="Path to a JSON config file overriding CLI defaults")
    config_args, remaining = config_parser.parse_known_args(argv)

    ap = argparse.ArgumentParser(parents=[config_parser])
    ap.add_argument("--base-model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.08)
    ap.add_argument("--per-dev-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--use-dora", action="store_true", help="Enable DoRA adapters instead of plain LoRA")
    ap.add_argument("--dora-ephemeral-offload", action="store_true", help="Enable runtime offload helpful for DoRA initialization")
    ap.add_argument("--lora-runtime-config", type=str, help="JSON dict passed to LoraRuntimeConfig")
    ap.add_argument("--lora-init", type=str, help="Initialization mode for LoRA/DoRA adapters (e.g. 'pissa', 'eva', 'orthogonal')")
    ap.add_argument("--lora-rank-pattern", type=str, help="JSON dict mapping module regex to LoRA ranks")
    ap.add_argument("--lora-alpha-pattern", type=str, help="JSON dict mapping module regex to LoRA alphas")
    ap.add_argument("--lora-target-modules", nargs="+", help="Override list of modules to wrap with LoRA/DoRA")
    ap.add_argument("--modules-to-save", nargs="+", help="Additional module names to mark as trainable and save")
    ap.add_argument("--merge-lora", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true")

    if config_args.config:
        cfg = _load_config(config_args.config)
        valid_keys = {action.dest for action in ap._actions if action.dest is not argparse.SUPPRESS}
        unknown = set(cfg.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown config option(s): {', '.join(sorted(unknown))}")
        for action in ap._actions:
            if action.dest in cfg:
                action.required = False
        ap.set_defaults(**cfg)

    args = ap.parse_args(remaining)
    if config_args.config:
        args.config = config_args.config
    return args

def load_csv(csv_path: str, sr: int = 16000, default_lang: str = "en") -> Dataset:
    df = pd.read_csv(csv_path)
    for col in ["path","text"]:
        if col not in df.columns:
            raise ValueError(f"{csv_path} is missing column: {col}")
    if "language" not in df.columns:
        df["language"] = default_lang
    ds = Dataset.from_pandas(df)
    return ds.cast_column("path", Audio(sampling_rate=sr))

@dataclass
class Proc:
    processor: WhisperProcessor
    def __call__(self, batch):
        audio = [x["array"] for x in batch["path"]]
        inputs = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        labels = self.processor.tokenizer(
            batch["text"],
            padding="longest",
            return_attention_mask=False,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

def compute_metrics(eval_pred, processor: WhisperProcessor):
    preds, labels = eval_pred
    labels = [[(l if l != -100 else processor.tokenizer.pad_token_id) for l in label] for label in labels]
    pred_str = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    pred_norm = [s.lower().strip() for s in pred_str]
    label_norm = [s.lower().strip() for s in label_str]
    return {"wer": wer(label_norm, pred_norm)}

def main():
    def mps_available():
        return torch.backends.mps.is_available() and torch.backends.mps.is_built()

    def cuda_available():
        return torch.cuda.is_available()

    def xpu_available():
        return hasattr(torch.backends, "xpu") and torch.backends.xpu.is_available()

    args = parse_args()
    set_seed(args.seed)

    resume_ckpt: str | None = None
    overwrite_output_dir = False
    if os.path.isdir(args.outdir):
        resume_ckpt = get_last_checkpoint(args.outdir)
        if resume_ckpt:
            print(f"Resuming from checkpoint: {resume_ckpt}")
        elif os.listdir(args.outdir):
            print(f"Existing files detected in {args.outdir}; starting fresh and overwriting outputs.")
            overwrite_output_dir = True

    os.makedirs(args.outdir, exist_ok=True)

    if cuda_available():
        device = "cuda"
    elif xpu_available():
        device = "xpu"
    elif mps_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    processor: WhisperProcessor = WhisperProcessor.from_pretrained(args.base_model)
    model: WhisperForConditionalGeneration | PeftModel | PeftMixedModel = WhisperForConditionalGeneration.from_pretrained(args.base_model, low_cpu_mem_usage=True, use_safetensors=True, token=os.getenv("HF_TOKEN"))

    if args.freeze_encoder:
        for p in model.model.encoder.parameters():
            p.requires_grad = False

    gen_cfg = GenerationConfig.from_pretrained(args.base_model)
    gen_cfg.language = args.language
    gen_cfg.task = "transcribe"
    model.generation_config = gen_cfg

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.use_dora:
        print("Using DoRA adapters")
    else:
        print("Using LoRA adapters")

    target_modules = args.lora_target_modules or ["q_proj","k_proj","v_proj","o_proj","fc1","fc2"]

    rank_pattern = _maybe_parse_json(args.lora_rank_pattern, arg_name="lora_rank_pattern")
    if rank_pattern is not None and not isinstance(rank_pattern, dict):
        raise ValueError("lora_rank_pattern must decode to a JSON object")

    alpha_pattern = _maybe_parse_json(args.lora_alpha_pattern, arg_name="lora_alpha_pattern")
    if alpha_pattern is not None and not isinstance(alpha_pattern, dict):
        raise ValueError("lora_alpha_pattern must decode to a JSON object")

    runtime_cfg = _maybe_parse_json(args.lora_runtime_config, arg_name="lora_runtime_config")
    if runtime_cfg is not None and not isinstance(runtime_cfg, dict):
        raise ValueError("lora_runtime_config must decode to a JSON object")

    if runtime_cfg is None:
        runtime_cfg = {}

    if args.dora_ephemeral_offload:
        runtime_cfg.setdefault("ephemeral_gpu_offload", True)

    accelerator_supports_ephemeral = cuda_available() or xpu_available()
    if args.use_dora and not accelerator_supports_ephemeral:
        if runtime_cfg.get("ephemeral_gpu_offload", False):
            print("Disabling DoRA ephemeral GPU offload: CUDA/XPU backend unavailable.")
        runtime_cfg["ephemeral_gpu_offload"] = False

    runtime_config = LoraRuntimeConfig(**runtime_cfg) if runtime_cfg else None

    init_lora_weights: bool | str | None
    init_lora_weights = args.lora_init
    if isinstance(init_lora_weights, str):
        lowered = init_lora_weights.strip().lower()
        if lowered in {"", "default", "none"}:
            init_lora_weights = None
        elif lowered == "true":
            init_lora_weights = True
        elif lowered == "false":
            init_lora_weights = False
        else:
            init_lora_weights = init_lora_weights.strip()

    modules_to_save = args.modules_to_save
    if isinstance(modules_to_save, str):
        modules_to_save = [modules_to_save]

    lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="SEQ_2_SEQ_LM",
        use_dora=args.use_dora,
    )
    if rank_pattern:
        lora_kwargs["rank_pattern"] = rank_pattern
    if alpha_pattern:
        lora_kwargs["alpha_pattern"] = alpha_pattern
    if modules_to_save:
        lora_kwargs["modules_to_save"] = modules_to_save
    if init_lora_weights is not None:
        lora_kwargs["init_lora_weights"] = init_lora_weights
    if runtime_config is not None:
        lora_kwargs["runtime_config"] = runtime_config

    lora_cfg = LoraConfig(**lora_kwargs)
    model_peft: PeftModel | PeftMixedModel = get_peft_model(model, lora_cfg)

    # PEFT's default seq2seq wrapper always forwards an `input_ids` kwarg, but
    # Whisper models operate on `input_features`. Patch the forward method so
    # audio batches bypass the seq2seq helper and call the generic PEFT path.
    orig_forward = model_peft.forward

    def whisper_peft_forward(self, *args, **kwargs):
        if "input_features" in kwargs and "input_ids" not in kwargs:
            return PeftModel.forward(self, *args, **kwargs)
        return orig_forward(*args, **kwargs)

    model_peft.forward = MethodType(whisper_peft_forward, model_peft)

    # Gradient checkpointing in HF expects the decoder embeddings to emit
    # tensors that require grad, otherwise the LoRA weights receive no grads.
    if hasattr(model_peft, "enable_input_require_grads"):
        model_peft.enable_input_require_grads

    train_ds = load_csv(args.train_csv)
    val_ds   = load_csv(args.val_csv)

    proc = Proc(processor)
    train_p = train_ds.map(proc, batched=True, remove_columns=train_ds.column_names)
    val_p   = val_ds.map(proc, batched=True, remove_columns=val_ds.column_names)

    ta = Seq2SeqTrainingArguments(
        output_dir=args.outdir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_dev_batch,
        per_device_eval_batch_size=args.per_dev_batch,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        predict_with_generate=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=3,
        report_to=["none"],
        fp16=args.fp16,
        bf16=args.bf16,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        label_names=["labels"],
    )
    ta.overwrite_output_dir = overwrite_output_dir

    trainer = Seq2SeqTrainer(
        model=model_peft,
        args=ta,
        train_dataset=train_p,
        eval_dataset=val_p,
        processing_class=processor.feature_extractor,
        compute_metrics=lambda p: compute_metrics(p, processor),
    )

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(args.outdir)

    # Save processor (includes preprocessor_config.json with mel filter parameters)
    processor.save_pretrained(args.outdir)
    print(f"✓ Processor saved to {args.outdir} (includes preprocessor_config.json)")

    if args.merge_lora:
        print("Merging LoRA → full model …")
        merged = model_peft.merge_and_unload()  # type: ignore
        merged_dir = os.path.join(args.outdir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged.save_pretrained(merged_dir)  # type: ignore
        processor.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")

if __name__ == "__main__":
    main()
