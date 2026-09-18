"""Evaluate the trained checkpoint on the held-out, speaker-disjoint test split.

Separate from the training kernel so scoring does not require retraining: the
training kernel's output is attached as a kernel data source, which is where
best.pt and tokenizer.model come from.
"""
import os
import subprocess
import sys

REPO = "https://github.com/sayedshaun/zipformer-training-pipeline.git"
BRANCH = "kaggle"
WORKDIR = "/tmp/zipformer-training-pipeline"
CONFIG = "config.kaggle.yaml"


def run(cmd, **kw):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kw)


if not os.path.isdir(WORKDIR):
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, WORKDIR])
os.chdir(WORKDIR)
run([sys.executable, "-m", "pip", "install", "-q", "soxr"])


def find(root, name):
    for base, dirs, files in os.walk(root):
        if name in files:
            return os.path.join(base, name)
        dirs[:] = [d for d in dirs if d != "clips"]
    return None


print("\n/kaggle/input tree:", flush=True)
for base, dirs, files in os.walk("/kaggle/input"):
    if base.count(os.sep) <= 4:
        print(f"  {base}  {[f for f in files][:4]}", flush=True)
    dirs[:] = [d for d in dirs if d != "clips"]

ckpt = find("/kaggle/input", "best.pt")
tok = find("/kaggle/input", "tokenizer.model")
corpus = os.path.dirname(find("/kaggle/input", "validated.tsv"))
print(f"\ncheckpoint: {ckpt}\ntokenizer:  {tok}\ncorpus:     {corpus}", flush=True)

import yaml

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)
for s in cfg["data"]["sources"]:
    if s.get("type") == "commonvoice":
        s["corpus_dir"] = corpus
# eval.py resolves the checkpoint and tokenizer from manifests.output_dir, so
# point it at a directory holding the mounted pair.
os.makedirs("/kaggle/working/ckpt", exist_ok=True)
run(["cp", ckpt, "/kaggle/working/ckpt/best.pt"])
run(["cp", tok, "/kaggle/working/ckpt/tokenizer.model"])
cfg["manifests"]["output_dir"] = "/kaggle/working/ckpt"
cfg["eval"]["manifest"] = "/kaggle/working/data/test_manifest.json"
cfg["eval"]["num_workers"] = 2
RESOLVED = "config.eval.resolved.yaml"
with open(RESOLVED, "w") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True)

# Rebuild manifests: the same SPLIT_SEED reproduces the identical test split.
run([sys.executable, "prepare_data.py", "--config", RESOLVED])
run([sys.executable, "eval.py", "--config", RESOLVED])
