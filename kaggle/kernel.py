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
# Not under /kaggle/working: everything there is returned as kernel output, and
# the cloned repo would be shipped back alongside the checkpoints.
WORKDIR = "/tmp/zipformer-training-pipeline"
# Flip to False for the full training run.
SMOKE = False
CONFIG = "config.kaggle.smoke.yaml" if SMOKE else "config.kaggle.yaml"


def run(cmd, **kwargs):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


if not os.path.isdir(WORKDIR):
    run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, WORKDIR])
os.chdir(WORKDIR)

# soxr does the 32kHz -> 16kHz resampling; everything else is preinstalled.
run([sys.executable, "-m", "pip", "install", "-q", "soxr"])

# W&B is how checkpoints and metrics leave the machine while it is still
# training - a batch kernel's /kaggle/working is not readable until the run
# ends. The key comes from a Kaggle secret, never from the committed config.
try:
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    for label in ("WANDB_API_KEY", "wandb_api_key", "WANDB", "wandb"):
        try:
            os.environ["WANDB_API_KEY"] = secrets.get_secret(label)
            print(f"wandb: using secret {label!r}")
            break
        except Exception:
            continue
    else:
        print("wandb: no secret found, continuing without it")
except Exception as exc:
    print(f"wandb: secrets unavailable ({exc}), continuing without it")

# Only turn wandb on when a key actually arrived, so a missing secret degrades
# to a normal run instead of failing at wandb.init().
os.environ["ZIPFORMER_WANDB"] = "1" if os.environ.get("WANDB_API_KEY") else "0"

# The mount directory is not always the dataset slug, and a dataset may nest
# the corpus under its own folder, so locate validated.tsv rather than assume
# a path - a wrong guess only shows up minutes into the run.
def find_corpus_root(root="/kaggle/input"):
    print(f"\nContents of {root}:", flush=True)
    for base, dirs, files in os.walk(root):
        depth = base[len(root):].count(os.sep)
        if depth <= 2:
            shown = [f for f in files if f.endswith(".tsv")][:6]
            print(f"  {base}  ({len(dirs)} dirs, {len(files)} files) {shown}", flush=True)
        if "validated.tsv" in files:
            return base
        # clips/ holds hundreds of thousands of entries; never descend into it.
        dirs[:] = [d for d in dirs if d != "clips"]
    return None


corpus_dir = find_corpus_root()
if corpus_dir is None:
    raise SystemExit("No validated.tsv anywhere under /kaggle/input - is the dataset attached?")
print(f"\nFound corpus at: {corpus_dir}", flush=True)

# Rewrite the config's corpus_dir to whatever was actually found.
import yaml

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)
for source in cfg["data"]["sources"]:
    if source.get("type") == "commonvoice":
        source["corpus_dir"] = corpus_dir
RESOLVED_CONFIG = "config.kaggle.resolved.yaml"
with open(RESOLVED_CONFIG, "w") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True)

# Manifests point at the read-only mount, so this writes only JSON.
run([sys.executable, "prepare_data.py", "--config", RESOLVED_CONFIG])

CONFIG_OUT = "/kaggle/working/zipformer_tiny_smoke" if SMOKE else "/kaggle/working/zipformer_tiny"

n_gpu = subprocess.run(
    ["nvidia-smi", "--list-gpus"], capture_output=True, text=True
).stdout.strip().count("\n") + 1
print(f"\nVisible GPUs: {n_gpu}", flush=True)

# torchrun spawns one process per GPU; train.py picks up RANK/LOCAL_RANK/WORLD_SIZE.
run([
    "torchrun", f"--nproc_per_node={n_gpu}", "--master_port=29500",
    "train.py", "--config", RESOLVED_CONFIG,
])

# /kaggle/working is what Kaggle persists as the kernel's output.
run(["cp", "-r", CONFIG_OUT, "/kaggle/working/output_model"])
print("\nDone. Checkpoints under /kaggle/working/output_model", flush=True)
