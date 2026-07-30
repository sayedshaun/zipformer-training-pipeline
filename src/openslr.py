"""Core dataset logic: download OpenSLR-53 (Large Bengali ASR training data set)
shards and build NeMo manifests.

The corpus ships as 16 independent zip shards (hex digits 0-f), each holding
its own copy of the full utt_spk_text.tsv transcript file plus the audio for
utterance IDs whose first hex character matches the shard. There is no
official train/dev/test split, so we hold out a random dev/test sample.
"""
import json
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from src.audio import convert_clip
from src.download import download_with_resume

BASE_URL = "https://openslr.trmal.net/resources/53"
ALL_SHARDS = list("0123456789abcdef")
SPLIT_SEED = 42


def download_shard(shard: str, dest: Path) -> Path:
    archive_path = dest / f"asr_bengali_{shard}.zip"
    return download_with_resume(f"{BASE_URL}/asr_bengali_{shard}.zip", archive_path)


def extract_shard(shard: str, archive_path: Path, dest: Path) -> None:
    marker = dest / f".extracted_{shard}"
    if marker.exists():
        return
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dest)
    marker.touch()


def load_transcripts(corpus_dir: Path) -> dict:
    transcripts = {}
    with open(corpus_dir / "utt_spk_text.tsv", encoding="utf-8") as f:
        for line in f:
            utt_id, _speaker_id, text = line.rstrip("\n").split("\t", 2)
            transcripts[utt_id] = text.strip()
    return transcripts


def collect_utterances(corpus_dir: Path, transcripts: dict) -> list:
    utterances = []
    for flac_path in (corpus_dir / "data").glob("*/*.flac"):
        text = transcripts.get(flac_path.stem)
        if text:
            utterances.append((flac_path.stem, flac_path, text))
    return utterances


def write_manifest(utterances: list, clips_out_dir: Path, manifest_path: Path, workers: int, desc: str) -> int:
    clips_out_dir.mkdir(parents=True, exist_ok=True)

    jobs, rows = [], []
    for utt_id, src, text in utterances:
        dst = clips_out_dir / f"openslr_{utt_id}.wav"
        jobs.append((src, dst))
        rows.append((dst, text))

    written = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, open(manifest_path, "w") as out:
        for (dst, text), duration in zip(rows, tqdm(pool.map(convert_clip, jobs), total=len(jobs), desc=desc)):
            if not text:
                continue
            out.write(json.dumps({"audio_filepath": str(dst), "text": text, "duration": duration}, ensure_ascii=False) + "\n")
            written += 1
    return written


def prepare_openslr_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = getattr(args, "manifest_prefix", "openslr_")
    splits = ["train", "dev", "test"]

    raw_dir = output_dir / "openslr_bengali"
    corpus_dir = raw_dir / "asr_bengali"

    pending_splits = [
        split for split in splits
        if not (output_dir / f"{prefix}{split}_manifest.json").exists()
    ]

    pools = {}
    if pending_splits:
        shards = getattr(args, "shards", None) or ALL_SHARDS
        if shards == "all":
            shards = ALL_SHARDS

        if not args.skip_download:
            for shard in shards:
                archive_path = download_shard(shard, raw_dir)
                extract_shard(shard, archive_path, raw_dir)
        elif not corpus_dir.exists():
            raise SystemExit(f"skip_download is set but no extracted corpus found at {corpus_dir}")

        transcripts = load_transcripts(corpus_dir)
        utterances = collect_utterances(corpus_dir, transcripts)
        random.Random(SPLIT_SEED).shuffle(utterances)

        dev_n = getattr(args, "dev_utterances", 500)
        test_n = getattr(args, "test_utterances", 500)
        pools = {
            "dev": utterances[:dev_n],
            "test": utterances[dev_n:dev_n + test_n],
            "train": utterances[dev_n + test_n:],
        }

    counts = {}
    for split in splits:
        manifest_path = output_dir / f"{prefix}{split}_manifest.json"
        if split not in pending_splits:
            with open(manifest_path) as f:
                counts[split] = sum(1 for _ in f)
            print(f"{split}: manifest already exists -> {manifest_path} ({counts[split]} utterances), skipping")
            continue
        counts[split] = write_manifest(
            pools[split], output_dir / "wavs", manifest_path, args.workers, desc=f"openslr-{split}",
        )
        print(f"{split}: wrote {counts[split]} utterances -> {manifest_path}")
    return counts
