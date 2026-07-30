"""Core dataset logic: download Mozilla Data Collective (Common Voice) releases
and build NeMo manifests."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import tarfile
from tqdm import tqdm

from src.audio import convert_clip
from src.download import download_with_resume


def download_dataset(dataset_id: str, api_key: str, dest: Path) -> Path:
    resp = requests.post(
        f"https://mozilladatacollective.com/api/datasets/{dataset_id}/download",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    download_url = resp.json()["downloadUrl"]

    archive_path = dest / f"{dataset_id}.tar.gz"
    return download_with_resume(download_url, archive_path)


def find_corpus_root(path: Path):
    """Common Voice archives nest the actual corpus (clips/ + *.tsv) inside a
    locale subdirectory, e.g. cv-corpus-26.0-.../bn/clips - search for it."""
    if (path / "clips").is_dir():
        return path
    for p in path.iterdir():
        if p.is_dir():
            found = find_corpus_root(p)
            if found:
                return found
    return None


def extract_dataset(archive_path: Path, dest: Path) -> Path:
    with tarfile.open(archive_path) as tar:
        tar.extractall(dest)
    subdirs = [p for p in dest.iterdir() if p.is_dir()]
    top = subdirs[0] if len(subdirs) == 1 else dest
    corpus_dir = find_corpus_root(top)
    if corpus_dir is None:
        raise RuntimeError(f"Could not find a clips/ directory anywhere under {top}")
    return corpus_dir


def find_existing_corpus(output_dir: Path):
    if not output_dir.exists():
        return None
    for p in output_dir.iterdir():
        if p.is_dir():
            found = find_corpus_root(p)
            if found:
                return found
    return None


def build_manifest(corpus_dir: Path, split: str, clips_out_dir: Path, manifest_path: Path, workers: int) -> int:
    if split == "train":
        # train.tsv is only the official CV subset of validated.tsv; pull in the
        # rest of the validated pool too, excluding whatever dev/test hold out.
        df = pd.read_csv(corpus_dir / "validated.tsv", sep="\t")
        held_out = set()
        for other_split in ("dev", "test"):
            other_tsv = corpus_dir / f"{other_split}.tsv"
            if other_tsv.exists():
                held_out.update(pd.read_csv(other_tsv, sep="\t")["path"])
        df = df[~df["path"].isin(held_out)]
    else:
        df = pd.read_csv(corpus_dir / f"{split}.tsv", sep="\t")
    clips_out_dir.mkdir(parents=True, exist_ok=True)

    jobs, rows = [], []
    for _, row in df.iterrows():
        src = corpus_dir / "clips" / row["path"]
        dst = clips_out_dir / (Path(row["path"]).stem + ".wav")
        jobs.append((src, dst))
        rows.append((dst, str(row["sentence"]).strip()))

    written = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, open(manifest_path, "w") as out:
        for (dst, text), duration in zip(rows, tqdm(pool.map(convert_clip, jobs), total=len(jobs), desc=split)):
            if not text:
                continue
            out.write(json.dumps({"audio_filepath": str(dst), "text": text, "duration": duration}, ensure_ascii=False) + "\n")
            written += 1
    return written


def prepare_mcv_dataset(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = getattr(args, "manifest_prefix", "")

    pending_splits = [
        split for split in args.splits
        if not (output_dir / f"{prefix}{split}_manifest.json").exists()
    ]

    corpus_dir = None
    if pending_splits:
        corpus_dir = find_existing_corpus(output_dir)
        if corpus_dir:
            print(f"Found existing extracted corpus at {corpus_dir}, skipping download/extraction")
        elif args.skip_download:
            corpus_dir = next(p for p in output_dir.iterdir() if p.is_dir())
        else:
            archive_path = download_dataset(args.dataset_id, args.api_key, output_dir)
            corpus_dir = extract_dataset(archive_path, output_dir)

    counts = {}
    for split in args.splits:
        manifest_path = output_dir / f"{prefix}{split}_manifest.json"
        if split not in pending_splits:
            with open(manifest_path) as f:
                counts[split] = sum(1 for _ in f)
            print(f"{split}: manifest already exists -> {manifest_path} ({counts[split]} utterances), skipping")
            continue
        counts[split] = build_manifest(corpus_dir, split, output_dir / "wavs", manifest_path, args.workers)
        print(f"{split}: wrote {counts[split]} utterances -> {manifest_path}")
    return counts
