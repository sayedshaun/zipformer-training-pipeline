"""Loads a section of config.yaml into a Namespace."""
import re
from argparse import Namespace

import yaml

# PyYAML's default SafeLoader fails to recognize exponential notation without
# a decimal point (e.g. "1e-5") as a float and silently leaves it as a string.
# Patch in the fuller float resolver so values like config.yaml's `lr: 1e-5`
# still load as floats.
_FLOAT_LOADER = yaml.SafeLoader
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


def load_sections(config_path: str, *sections: str) -> Namespace:
    """Merges several top-level sections into one flat Namespace - used by
    train.py/eval.py, which read across data/model/train (or data/model/eval)."""
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=_FLOAT_LOADER)
    merged = {}
    for section in sections:
        merged.update(cfg.get(section) or {})
    return Namespace(**merged)
