# Contributing Guide

We welcome pull requests and bug reports that improve this finetuning supplement for [whisper.cpp](https://github.com/ggerganov/whisper.cpp). Please follow the steps below to streamline the review process.

## Getting Started

1. Fork the repository and create a feature branch from `main`.
2. Install dependencies with `pip install -r requirements.txt`.
3. Run the provided smoke tests before opening a pull request:
   ```bash
   python test_merged_model.py
   python verify_conversion_ready.py
   ```
4. Ensure any new CLI flags or configuration keys are documented in `docs/`.

## Commit Standards

- Write clear, actionable commit messages in the imperative mood, e.g. `Add GPU health check`.
- Group unrelated changes into separate commits.
- If your change affects training or conversion output, include before/after notes in the pull request description.

## Pull Request Checklist

- [ ] Tests and lint checks pass locally.
- [ ] Relevant docs in `docs/` or `README.md` are updated.
- [ ] Added or updated configuration examples when necessary.
- [ ] Screenshots or logs attached for UI or CLI changes when applicable.

## Code Style

- Follow the existing formatting conventions; use `ruff` or `black` if you prefer auto-formatting, but keep diffs focused.
- Add concise comments when behavior is non-obvious, especially around accelerator or adapter tuning.
- Prefer explicit variable names and typed dictionaries when extending configs.

## Reporting Issues

Please include:

- Python version, GPU model, and driver/CUDA info.
- Steps to reproduce, including dataset manifest snippets if possible.
- Console logs or stack traces.

## Community Expectations

Respect fellow contributors. Review feedback should focus on code and behavior, not individuals. The `docs/CODE_OF_CONDUCT.md` governs all community interactions.
