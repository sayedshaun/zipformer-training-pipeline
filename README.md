# Zipformer Training Pipeline

Train a Zipformer (Yao et al., ICLR 2024) speech recognition model from
scratch on Bengali, end to end: build manifests from Common Voice + OpenSLR
data, train a tokenizer, train the model (CTC or RNNT), and evaluate it.

There is no NeMo/HuggingFace-style pip-installable Zipformer — the reference
implementation ([k2-fsa/icefall](https://github.com/k2-fsa/icefall)) needs
`k2` (a CUDA/torch-version-matched wheel) and `lhotse`, and is a
clone-and-run-scripts recipe collection rather than a library. Rather than
taking on that dependency stack, this repo follows the same approach the
sibling [`conformer-training-pipeline`](../conformer-training-pipeline) repo
took for FastConformer: **the model is implemented from scratch in
[`model.py`](model.py)**, module by module. See [`ARCHITECTURE.md`](ARCHITECTURE.md)
for the full reference this follows and what's simplified vs. the paper
(single-stack encoder, not the full U-Net multi-rate stack).

## Relationship to the other repos in this family

- Data-prep code (`src/config.py`, `src/mcv.py`, `src/openslr.py`,
  `src/download.py`, `src/audio.py`, `prepare_data.py`) is copied unchanged
  from [`asr-training-pipeline`](../asr-training-pipeline) — the manifest
  format (`{"audio_filepath", "text", "duration"}` JSON-lines) and the
  mcv/openslr sources are architecture-agnostic.
- `config.yaml`'s `data.output_dir` points at that repo's `data/` directory
  by **absolute path**, and `manifests.train_manifest`/`val_manifest` point
  at the manifests it already produces there — so this repo reuses the
  already-downloaded/extracted Common Voice (and OpenSLR, once pulled) data
  in place instead of duplicating ~14GB+ on disk. Run `prepare_data.py` here
  only if you want this repo to (re)build manifests independently.
- `model.py`/`dataset.py`/`tokenizer.py`/`train.py`/`eval.py` mirror
  `conformer-training-pipeline`'s from-scratch style and CLI conventions —
  only the encoder internals differ (see `ARCHITECTURE.md`).

## Pipeline overview

```
prepare_data.py   →  train.py (auto-trains a tokenizer   →  eval.py
(build NeMo-style     on first run if none exists)           (WER/CER on
 manifests, optional)  → Zipformer CTC/RNNT training            held-out test set)
```

## Requirements

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

An NVIDIA GPU is assumed for training/evaluation (`train.device`/`eval.device: cuda`).

## Configuration

Every script takes only `-c/--config` (`prepare_data.py`) or `--config`
(`train.py`/`eval.py`), defaulting to `config.yaml`:

| Script | Config section(s) | Purpose |
|---|---|---|
| `prepare_data.py` | `data:` | Dataset download + manifest generation (mcv + openslr sources) |
| `train.py` | `manifests:`, `model:`, `train:`, `wandb:` | Tokenizer (auto-trained on first run) + model training |
| `eval.py` | `manifests:`, `model:`, `eval:` | WER/CER evaluation |

See [`config.yaml`](config.yaml) for the full set of keys and their defaults.

## Usage

```bash
# 1. (Optional) Build manifests independently of the sibling repo
python prepare_data.py

# 2. Train (auto-trains a SentencePiece tokenizer into manifests.output_dir on first run)
python train.py

# 3. Evaluate the best checkpoint
python eval.py
```

`train.py` supports CTC (default) or RNNT via `train.loss: ctc | rnnt` in
`config.yaml` — train CTC to convergence first, per `ARCHITECTURE.md`'s
suggested build order, before attempting RNNT.

## Project layout

```
config.yaml            Single source of truth for all pipeline settings
ARCHITECTURE.md         Zipformer architectural reference (BiasNorm, shared
                        attention weights, non-linear attention, Swoosh, bypass)
prepare_data.py         CLI: runs each configured data source, merges manifests
model.py                Zipformer, built from scratch (see ARCHITECTURE.md)
dataset.py              Manifest-backed PyTorch Dataset/DataLoader
tokenizer.py            SentencePiece BPE tokenizer
train.py                CLI: model training (CTC/RNNT)
eval.py                 CLI: WER/CER evaluation
src/
  config.py             YAML config-section loader
  mcv.py                 Common Voice / Mozilla Data Collective source
  openslr.py             OpenSLR-53 Bengali corpus source
  download.py             Shared resumable-download helper
  audio.py                Shared clip-to-16kHz-mono-WAV conversion helper
```
