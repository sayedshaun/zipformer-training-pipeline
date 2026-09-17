"""CLI entrypoint for downloading one or more data sources and building NeMo-style
{audio_filepath, text, duration} manifests that dataset.py / train.py / eval.py read.

All settings come from config.yaml's `data:` section. Each entry under
`data.sources` picks a `type` (mcv or openslr) and its own options; every
source writes its own `{name}_{split}_manifest.json` files, which are then
merged into the final train/dev/test manifests. The Mozilla Data Collective
API key comes from the MDC_API_KEY env var (.env), never from config.yaml.

This is copied unchanged from the asr-training-pipeline (NeMo/FastConformer)
repo's data-prep stage - the manifest format, mcv/openslr sources, and merge
step are architecture-agnostic. `data.output_dir` defaults to that repo's
`data/` directory by absolute path, so the already-downloaded/extracted
Common Voice corpus is reused in place rather than duplicated on disk.
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from src.commonvoice import prepare_commonvoice_dataset
from src.config import load_section
from src.mcv import prepare_mcv_dataset
from src.openslr import prepare_openslr_dataset

load_dotenv()

PREPARERS = {
    "mcv": prepare_mcv_dataset,
    "openslr": prepare_openslr_dataset,
    # An already-extracted Common Voice tree (e.g. a read-only Kaggle mount).
    "commonvoice": prepare_commonvoice_dataset,
}
SPLITS = ("train", "dev", "test")


def merge_manifests(output_dir: Path, source_names: list) -> None:
    for split in SPLITS:
        part_paths = [output_dir / f"{name}_{split}_manifest.json" for name in source_names]
        part_paths = [p for p in part_paths if p.exists()]
        if not part_paths:
            continue
        with open(output_dir / f"{split}_manifest.json", "w") as out:
            for part_path in part_paths:
                with open(part_path) as f:
                    part = f.read()
                # Guarantee the separator between parts: without it a part file
                # missing its trailing newline would splice its last JSON object
                # onto the next part's first one.
                if part and not part.endswith("\n"):
                    part += "\n"
                out.write(part)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config.yaml")
    cli_args = parser.parse_args()

    args = load_section(cli_args.config, "data", required=["output_dir", "sources"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_names = []
    for source_cfg in args.sources:
        source_cfg = dict(source_cfg)
        source_type = source_cfg.pop("type", None)
        if source_type not in PREPARERS:
            raise SystemExit(f"Unknown source type {source_type!r}, expected one of {list(PREPARERS)}")
        name = source_cfg.pop("name", source_type)
        source_names.append(name)

        source_cfg.setdefault("output_dir", args.output_dir)
        source_cfg.setdefault("workers", getattr(args, "workers", 8))
        source_cfg.setdefault("skip_download", getattr(args, "skip_download", False))
        source_cfg["manifest_prefix"] = f"{name}_"

        if source_type == "mcv":
            source_cfg.setdefault("splits", list(SPLITS))
            source_cfg["api_key"] = os.environ.get("MDC_API_KEY")
            if not source_cfg["api_key"] and not source_cfg["skip_download"]:
                raise SystemExit(f"Set MDC_API_KEY in .env, or skip_download: true for the {name!r} source")

        PREPARERS[source_type](argparse.Namespace(**source_cfg))

    merge_manifests(output_dir, source_names)


if __name__ == "__main__":
    main()
