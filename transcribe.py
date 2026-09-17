"""Transcribe a single audio file with a trained Zipformer checkpoint.

eval.py scores a whole manifest; this is for pointing a checkpoint at one file
- a recording you have lying around, not a corpus utterance. Any format
soundfile can open works, at any sample rate or channel count.

Long recordings are decoded in chunks: attention is O(T^2) in frames, and a
model trained on ~5-10s utterances has never seen a minutes-long one, so
feeding it whole degrades the output well before it runs out of memory.

Usage:
    python transcribe.py --model zipformer_tiny/best.pt --audio clip.mp3
    python transcribe.py --model best.pt --audio clip.mp3 --reference truth.txt
"""

import argparse
from pathlib import Path

import soundfile as sf
import soxr
import torch

from eval import corpus_error_rate, greedy_ctc_decode
from src.config import load_sections
from src.model import ZipformerFromScratch
from tokenizer import BPETokenizer

SAMPLE_RATE = 16000


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", help="Checkpoint path (default: manifests.output_dir/best.pt)")
    parser.add_argument("--tokenizer", help="Tokenizer path (default: manifests.output_dir/tokenizer.model)")
    parser.add_argument("--audio", required=True, help="Audio file to transcribe")
    parser.add_argument("--reference", help="File holding the true transcript; enables WER/CER")
    parser.add_argument("--chunk-seconds", type=float, default=15.0)
    parser.add_argument("--overlap-seconds", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    return parser


def load_audio(path: str) -> torch.Tensor:
    waveform, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)  # downmix; the model is monaural
    if sr != SAMPLE_RATE:
        waveform = soxr.resample(waveform, sr, SAMPLE_RATE)
    return torch.from_numpy(waveform), len(waveform) / SAMPLE_RATE


def chunk_waveform(waveform: torch.Tensor, chunk_seconds: float, overlap_seconds: float):
    chunk = int(chunk_seconds * SAMPLE_RATE)
    step = max(int((chunk_seconds - overlap_seconds) * SAMPLE_RATE), 1)
    if waveform.numel() <= chunk:
        return [waveform]
    return [waveform[start : start + chunk] for start in range(0, waveform.numel(), step)]


def main():
    args = build_arg_parser().parse_args()
    manifests_args = load_sections(args.config, "manifests")
    model_args = load_sections(args.config, "model")

    output_dir = Path(manifests_args.output_dir)
    model_path = args.model or output_dir / "best.pt"
    tokenizer_path = args.tokenizer or output_dir / "tokenizer.model"
    for path in (model_path, tokenizer_path, args.audio):
        if not Path(path).exists():
            raise SystemExit(f"Not found: {path}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = BPETokenizer.load(str(tokenizer_path))

    model = ZipformerFromScratch(
        vocab_size=tokenizer.vocab_size,
        d_model=model_args.d_model,
        n_layers=model_args.n_layers,
        n_heads=model_args.n_heads,
        attn_head_dim=model_args.attn_head_dim,
        nla_hidden_dim=model_args.nla_hidden_dim,
        conv_kernel_size=model_args.conv_kernel_size,
        ff_expansion_factor=model_args.ff_expansion_factor,
        dropout=model_args.dropout,
        bypass_min_scale=model_args.bypass_min_scale,
        use_rnnt=False,
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # The RNNT branch is absent when the checkpoint was trained with --loss ctc.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"warning: {len(missing)} missing keys, first few: {missing[:3]}")
    model.to(device).eval()
    print(f"Loaded {model_path} (epoch {checkpoint.get('epoch', '?')}, step {checkpoint.get('step', '?')})")

    waveform, duration = load_audio(args.audio)
    chunks = chunk_waveform(waveform, args.chunk_seconds, args.overlap_seconds)
    print(f"{args.audio}: {duration:.1f}s -> {len(chunks)} chunk(s)")

    pieces = []
    with torch.no_grad():
        for i, chunk in enumerate(chunks, 1):
            batch = chunk.unsqueeze(0).to(device)
            lengths = torch.tensor([chunk.numel()], device=device)
            log_probs, encoded_lengths = model.forward_ctc(batch, lengths)
            text = greedy_ctc_decode(
                log_probs, encoded_lengths, model.ctc_head.blank_id, tokenizer
            )[0]
            print(f"  [{i}/{len(chunks)}] {text}")
            pieces.append(text)

    hypothesis = " ".join(p for p in pieces if p).strip()
    print("\n--- transcript ---")
    print(hypothesis or "(empty - the model emitted only blanks)")

    if args.reference:
        reference = Path(args.reference).read_text(encoding="utf-8").strip()
        wer = corpus_error_rate([reference], [hypothesis], level="word")
        cer = corpus_error_rate([reference], [hypothesis], level="char")
        print(f"\nWER: {wer * 100:.2f}%\nCER: {cer * 100:.2f}%")


if __name__ == "__main__":
    main()
