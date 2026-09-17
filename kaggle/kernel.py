"""Kaggle kernel entrypoint: train the tiny Zipformer on 2x T4.

Pushed with `kaggle kernels push -p kaggle/`. Requires, in the kernel settings
(kernel-metadata.json already sets these): GPU T4 x2 accelerator and internet
enabled, the latter only so this script can clone the repo.
"""
import os
import subprocess
import sys

REPO = "https://github.com/sayedshaun/zipformer-training-pipeline.git"
# The Kaggle-specific config and loader live on this branch, not on main.
BRANCH = "kaggle"
WORKDIR = "/kaggle/working/zipformer-training-pipeline"
# Flip to False for the full training run.
SMOKE = True
CONFIG = "config.kaggle.smoke.yaml" if SMOKE else "config.kaggle.yaml"


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


if not os.path.isdir(WORKDIR):
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, WORKDIR])
os.chdir(WORKDIR)

# soxr does the 32kHz -> 16kHz resampling; everything else is preinstalled.
run([sys.executable, "-m", "pip", "install", "-q", "soxr"])

# Manifests point at the read-only mount, so this writes only JSON.
run([sys.executable, "prepare_data.py", "--config", CONFIG])

CONFIG_OUT = "/kaggle/working/zipformer_tiny_smoke" if SMOKE else "/kaggle/working/zipformer_tiny"

n_gpu = subprocess.run(
    ["nvidia-smi", "--list-gpus"], capture_output=True, text=True
).stdout.strip().count("\n") + 1
print(f"\nVisible GPUs: {n_gpu}", flush=True)

# torchrun spawns one process per GPU; train.py picks up RANK/LOCAL_RANK/WORLD_SIZE.
run([
    "torchrun", f"--nproc_per_node={n_gpu}", "--master_port=29500",
    "train.py", "--config", CONFIG,
])

# /kaggle/working is what Kaggle persists as the kernel's output.
run(["cp", "-r", CONFIG_OUT, "/kaggle/working/output_model"])
print("\nDone. Checkpoints under /kaggle/working/output_model", flush=True)
