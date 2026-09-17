"""Build manifests from a Common Voice corpus that is already on disk.

Unlike src/mcv.py (which downloads from the Mozilla Data Collective API and
transcodes every clip to 16kHz wav), this module assumes the corpus is *mounted
read-only* - the Kaggle dataset case - so it never downloads and never writes
audio. Manifests point straight at the mounted MP3s, and src/dataset.py
resamples them per item.

Splits are cut on `client_id`, not on rows: Common Voice has many clips per
speaker, and a random row split would put the same voice in train and dev,
making val loss look far better than the real WER.
"""
import csv
import json
import random
import sys
from pathlib import Path

SPLIT_SEED = 42

# validated.tsv sentences can be long; the default field limit trips on them.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def load_clip_durations(corpus_dir: Path) -> dict:
    """path -> duration in seconds, from Common Voice's clip_durations.tsv.

    Returns {} when the file is absent; callers then fall back to not filtering
    on duration rather than decoding every clip to measure it.
    """
    durations_path = corpus_dir / "clip_durations.tsv"
    if not durations_path.exists():
        return {}

    durations = {}
    with open(durations_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            # Column is "duration[ms]" in current releases.
            raw = row.get("duration[ms]") or row.get("duration")
            if raw and raw.strip():
                durations[row["clip"]] = float(raw) / 1000.0
    return durations


def read_validated(corpus_dir: Path) -> list:
    rows = []
    with open(corpus_dir / "validated.tsv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sentence = (row.get("sentence") or "").strip()
            path = (row.get("path") or "").strip()
            if sentence and path:
                rows.append((row.get("client_id", ""), path, sentence))
    return rows


def split_by_speaker(rows: list, dev_utterances: int, test_utterances: int) -> dict:
    """Assign whole speakers to dev/test until each holdout is filled."""
    by_speaker = {}
    for client_id, path, sentence in rows:
        by_speaker.setdefault(client_id, []).append((path, sentence))

    speakers = sorted(by_speaker)
    random.Random(SPLIT_SEED).shuffle(speakers)

    splits = {"dev": [], "test": [], "train": []}
    quotas = {"dev": dev_utterances, "test": test_utterances}
    for speaker in speakers:
        target = next(
            (name for name in ("dev", "test") if len(splits[name]) < quotas[name]),
            "train",
        )
        splits[target].extend(by_speaker[speaker])
    return splits


def write_manifest(entries: list, clips_dir: Path, manifest_path: Path, durations: dict,
                   min_duration: float, max_duration: float) -> int:
    written = 0
    with open(manifest_path, "w", encoding="utf-8") as out:
        for path, sentence in entries:
            duration = durations.get(path)
            if duration is not None and not (min_duration <= duration <= max_duration):
                continue
            row = {
                "audio_filepath": str(clips_dir / path),
                "text": sentence,
                "duration": duration if duration is not None else 0.0,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    return written


def prepare_commonvoice_dataset(args):
    corpus_dir = Path(args.corpus_dir)
    if not (corpus_dir / "validated.tsv").exists():
        raise SystemExit(
            f"{corpus_dir}/validated.tsv not found - point corpus_dir at the "
            "directory holding validated.tsv and clips/"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = corpus_dir / "clips"

    rows = read_validated(corpus_dir)
    durations = load_clip_durations(corpus_dir)
    if not durations:
        print("clip_durations.tsv not found - skipping duration filtering")

    max_utterances = getattr(args, "max_utterances", None)
    if max_utterances and len(rows) > max_utterances:
        # Sample whole rows before the speaker split so a capped run still draws
        # from the full speaker pool rather than the head of the file.
        rows = random.Random(SPLIT_SEED).sample(rows, max_utterances)

    splits = split_by_speaker(rows, args.dev_utterances, args.test_utterances)

    prefix = getattr(args, "manifest_prefix", "")
    for split, entries in splits.items():
        manifest_path = output_dir / f"{prefix}{split}_manifest.json"
        count = write_manifest(
            entries, clips_dir, manifest_path, durations,
            getattr(args, "min_duration", 0.0), getattr(args, "max_duration", 30.0),
        )
        print(f"{split}: {count} utterances -> {manifest_path}")
