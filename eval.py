"""Evaluate a trained Zipformer checkpoint on a manifest: greedy decode
+ WER/CER against the reference transcripts.

Reads paths and model hyperparameters from config.yaml (manifests/model/eval
sections) rather than flags - edit that file, or pass --config to point at
another one.

Usage:
    python eval.py --config config.yaml
"""

import argparse
import json
from itertools import groupby
from pathlib import Path

import torch

from src.config import load_sections, require_existing_paths
from src.dataset import build_dataloader
from src.model import ZipformerFromScratch
from tokenizer import BPETokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser


def greedy_ctc_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank_id: int, tokenizer):
    """log_probs: (B, T, C). Returns a list of decoded strings, one per batch item."""
    predicted_ids = log_probs.argmax(dim=-1)  # (B, T)
    hypotheses = []
    for ids, length in zip(predicted_ids, lengths):
        ids = ids[: length.item()].tolist()
        collapsed = [token for token, _ in groupby(ids)]
        collapsed = [token for token in collapsed if token != blank_id]
        hypotheses.append(tokenizer.decode(collapsed))
    return hypotheses


def edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein distance between two token sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        curr = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def corpus_error_rate(references: list, hypotheses: list, level: str) -> float:
    """level: 'word' or 'char'. Aggregates edit distance / ref length over the whole corpus."""
    total_errors, total_ref_len = 0, 0
    for ref, hyp in zip(references, hypotheses):
        ref_units = ref.split() if level == "word" else list(ref)
        hyp_units = hyp.split() if level == "word" else list(hyp)
        total_errors += edit_distance(ref_units, hyp_units)
        total_ref_len += len(ref_units)
    return total_errors / max(total_ref_len, 1)


@torch.no_grad()
def main():
    cli_args = build_arg_parser().parse_args()
    manifests_args = load_sections(cli_args.config, "manifests")
    args = load_sections(cli_args.config, "model", "eval")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    output_dir = Path(manifests_args.output_dir)
    model_path = args.model_path or output_dir / "best.pt"
    vocab_path = output_dir / "tokenizer.model"

    require_existing_paths(
        cli_args.config, manifest=args.manifest, model_path=model_path, tokenizer=vocab_path,
    )

    tokenizer = BPETokenizer.load(str(vocab_path))
    loader = build_dataloader(
        args.manifest, tokenizer, args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    with open(args.manifest) as f:
        references = [json.loads(line)["text"] for line in f if line.strip()]

    model = ZipformerFromScratch(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        attn_head_dim=args.attn_head_dim,
        nla_hidden_dim=args.nla_hidden_dim,
        conv_kernel_size=args.conv_kernel_size,
        ff_expansion_factor=args.ff_expansion_factor,
        bypass_min_scale=args.bypass_min_scale,
        use_rnnt=False,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    # The RNNT branch is built only when training with `loss: rnnt`, and eval
    # decodes via CTC either way - so drop those keys and then load *strictly*,
    # so a config that no longer matches the checkpoint (wrong d_model,
    # n_layers, vocab size, ...) fails loudly instead of silently evaluating a
    # partly-randomly-initialised model.
    state_dict = {
        key: value
        for key, value in ckpt["model_state_dict"].items()
        if not key.startswith(("prediction_network.", "joint_network."))
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    hypotheses = []
    for waveforms, waveform_lengths, _, _ in loader:
        waveforms = waveforms.to(device)
        waveform_lengths = waveform_lengths.to(device)
        log_probs, encoded_lengths = model.forward_ctc(waveforms, waveform_lengths)
        hypotheses.extend(
            greedy_ctc_decode(log_probs, encoded_lengths, model.ctc_head.blank_id, tokenizer)
        )

    wer = corpus_error_rate(references, hypotheses, level="word")
    cer = corpus_error_rate(references, hypotheses, level="char")
    print(f"Utterances: {len(hypotheses)}")
    print(f"WER: {wer * 100:.2f}%")
    print(f"CER: {cer * 100:.2f}%")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for ref, hyp in zip(references, hypotheses):
                f.write(json.dumps({"reference": ref, "hypothesis": hyp}) + "\n")
        print(f"Wrote per-utterance hypotheses -> {out_path}")


if __name__ == "__main__":
    main()
