#!/usr/bin/env python3
# build_manifest_from_names.py
#
# Deterministically map WAV/MP3 filenames to one of four canonical texts (f01..f04)
# without running ASR. It expects text files f01.txt ... f04.txt in --texts_root.
#
# Examples of supported filename patterns (case-insensitive):
#   …_f01.wav, …-F02.mp3, …f03.WAV, …F04something.flac
#
# Grouping for stratified split:
#   - default: use the immediate parent directory name as a proxy for "speaker"
#   - fallback: use stem up to first '_' or '-' if parent directory is uniform
#
# Output:
#   outdir/train.csv  (columns: path,text,language)
#   outdir/val.csv
#
# Usage:
#   python build_manifest_from_names.py \
#     --audio_root /path/to/repo/audio \
#     --texts_root /path/to/texts \
#     --outdir ./manifests \
#     --val_fraction 0.1 \
#     --language en
#
# Optional:
#   --extensions .wav .mp3 .flac   (space-separated)
#
import argparse, json, os, re, sys, random, glob
import pandas as pd
from collections import defaultdict
from pathlib import Path

F_RE = re.compile(r"(?:^|[_\-\s])f0?([1-4])(?:[_\-\s]|[^0-9a-z]|$)", re.IGNORECASE)

def read_texts(texts_root: str) -> dict:
    texts = {}
    for i in range(1, 5):
        p = Path(texts_root) / f"f0{i}.txt"
        if not p.exists():
            raise FileNotFoundError(f"Missing required text file: {p}")
        texts[f"f0{i}"] = p.read_text(encoding="utf-8").strip()
    return texts

def infer_fid_from_name(path: Path) -> str | None:
    m = F_RE.search(path.stem + "_")  # trailing delimiter to ease end-match
    if not m:
        return None
    idx = int(m.group(1))
    return f"f0{idx}"

def group_key_for_split(path: Path) -> str:
    # Prefer parent directory as "speaker"
    parent = path.parent.name
    if parent and parent.lower() not in (".", "audio", "wav", "wavs", "mp3", "clips", "files"):
        return f"dir:{parent}"
    # Fallback: stem prefix up to first '_' or '-'
    stem = path.stem
    for sep in ["_", "-"]:
        if sep in stem:
            return f"stem:{stem.split(sep, 1)[0]}"
    return f"stem:{stem[:8]}"

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object with key/value pairs")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Path to a JSON config file overriding CLI defaults")
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--texts-root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--language", default="en")
    ap.add_argument("--extensions", nargs="*", default=[".wav", ".mp3", ".flac", ".m4a"])
    ap.add_argument("--seed", type=int, default=13)

    prelim, remaining = ap.parse_known_args()
    if prelim.config:
        cfg = _load_config(prelim.config)
        valid_keys = {action.dest for action in ap._actions if action.dest is not argparse.SUPPRESS}
        unknown = set(cfg.keys()) - valid_keys
        if unknown:
            raise ValueError(f"Unknown config option(s): {', '.join(sorted(unknown))}")
        for action in ap._actions:
            if action.dest in cfg:
                action.required = False
        ap.set_defaults(**cfg)
        ap.set_defaults(config=prelim.config)
    args = ap.parse_args(remaining)

    texts = read_texts(args.texts_root)
    audio_paths = []
    for ext in args.extensions:
        audio_paths.extend(Path(args.audio_root).rglob(f"*{ext}"))
    audio_paths = sorted(set(p.resolve() for p in audio_paths))

    if not audio_paths:
        print("No audio files found. Check --audio_root and --extensions.", file=sys.stderr)
        sys.exit(1)

    rows = []
    missing = []
    per_class = defaultdict(int)
    for p in audio_paths:
        fid = infer_fid_from_name(p)
        if fid is None:
            missing.append(str(p))
            continue
        text = texts[fid]
        rows.append({"path": str(p), "text": text, "language": args.language, "fid": fid, "group": group_key_for_split(p)})
        per_class[fid] += 1

    if not rows:
        print("Could not infer f01–f04 from any filenames. Adjust naming or regex.", file=sys.stderr)
        sys.exit(2)

    print("Class counts:")
    for k in sorted(per_class.keys()):
        print(f"  {k}: {per_class[k]}")
    if missing:
        print(f"Skipped {len(missing)} files with no f01–f04 token (first 10):")
        for s in missing[:10]:
            print("   -", s)

    # Stratified split by group (speaker-ish), while preserving class mix
    random.seed(args.seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)

    keys = list(groups.keys())
    random.shuffle(keys)
    n_val = max(1, int(len(keys) * args.val_fraction))
    val_keys = set(keys[:n_val])

    train, val = [], []
    for k in keys:
        (val if k in val_keys else train).extend(groups[k])

    # Drop helper cols
    for rec in train:
        rec.pop("fid", None); rec.pop("group", None)
    for rec in val:
        rec.pop("fid", None); rec.pop("group", None)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train).to_csv(outdir / "train.csv", index=False)
    pd.DataFrame(val).to_csv(outdir / "val.csv", index=False)
    print(f"Wrote {len(train)} train and {len(val)} val rows to {outdir}")

if __name__ == "__main__":
    main()
