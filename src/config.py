"""Loads a section of config.yaml into a Namespace."""
import re
from argparse import Namespace
from pathlib import Path

import yaml

# PyYAML's default SafeLoader fails to recognize exponential notation without
# a decimal point (e.g. "1e-5") as a float and silently leaves it as a string.
# Patch the fuller float resolver into a *subclass* so values like config.yaml's
# `lr: 1e-5` load as floats without changing yaml.safe_load for the whole process.
class _FLOAT_LOADER(yaml.SafeLoader):
    pass


_FLOAT_LOADER.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:
         [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.VERBOSE,
    ),
    list("-+0123456789."),
)


def load_section(config_path: str, section: str, required: list = ()) -> Namespace:
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=_FLOAT_LOADER)
    section_cfg = cfg.get(section) or {}

    missing = [key for key in required if section_cfg.get(key) is None]
    if missing:
        raise SystemExit(
            f"Missing required config values: {', '.join(missing)} "
            f"(set them in {config_path} under '{section}:')"
        )

    return Namespace(**section_cfg)


def require_existing_paths(config_path: str, **paths) -> None:
    """Fails fast with the offending config key when a configured input path is
    missing - config.yaml holds absolute paths into a sibling repo, so a config
    copied between machines otherwise dies much later with a bare FileNotFoundError."""
    missing = [f"{key}: {value}" for key, value in paths.items() if not Path(value).exists()]
    if missing:
        raise SystemExit(
            "Configured path(s) do not exist:\n  "
            + "\n  ".join(missing)
            + f"\nFix them in {config_path} (or run prepare_data.py to build the manifests)."
        )


def load_sections(config_path: str, *sections: str) -> Namespace:
    """Merges several top-level sections into one flat Namespace - used by
    train.py/eval.py, which read across data/model/train (or data/model/eval)."""
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=_FLOAT_LOADER)
    merged = {}
    for section in sections:
        merged.update(cfg.get(section) or {})
    return Namespace(**merged)
