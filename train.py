"""Training loop for the from-scratch Zipformer (src/model.py) on manifests
produced by prepare_data.py, using tokenizer.py and src/dataset.py.

Supports CTC (default, recommended per ARCHITECTURE.md build order) and, once
CTC training converges, RNNT via --loss rnnt.

Reads hyperparameters and run paths from config.yaml (manifests/model/train
sections) rather than flags - edit that file, or pass --config to point at
another one.

Usage:
    python train.py --config config.yaml
"""

import argparse
import json
import random
import time
from itertools import groupby
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.config import load_sections, require_existing_paths
from src.dataset import build_dataloader
from src.model import ZipformerFromScratch
from tokenizer import BPETokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    return parser


def build_model(args, vocab_size: int) -> ZipformerFromScratch:
    return ZipformerFromScratch(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        attn_head_dim=args.attn_head_dim,
        nla_hidden_dim=args.nla_hidden_dim,
        conv_kernel_size=args.conv_kernel_size,
        ff_expansion_factor=args.ff_expansion_factor,
        dropout=args.dropout,
        bypass_min_scale=args.bypass_min_scale,
        use_rnnt=(args.loss == "rnnt"),
    )


def warmup_decay_scale(step: int, warmup_steps: int) -> float:
    """Multiplier on the peak LR: linear ramp to 1.0 at warmup_steps, then
    inverse-sqrt decay. Peaks at exactly 1.0 so `args.lr` is the peak LR."""
    step = max(step, 1)
    return min(step / warmup_steps, (warmup_steps / step) ** 0.5)


AMP_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def compute_loss(model, batch, loss_type: str, device, amp_dtype=None):
    waveforms, waveform_lengths, targets, target_lengths = batch
    waveforms = waveforms.to(device)
    waveform_lengths = waveform_lengths.to(device)
    targets = targets.to(device)
    target_lengths = target_lengths.to(device)

    # Loss functions (ctc_loss/rnnt_loss) are numerically unstable in fp16/bf16,
    # so only the model forward pass runs under autocast; loss is computed in fp32.
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
    )

    if loss_type == "ctc":
        with autocast_ctx:
            log_probs, encoded_lengths = model.forward_ctc(waveforms, waveform_lengths)
        log_probs = log_probs.float().transpose(0, 1)  # CTCLoss wants (T, B, C)
        loss = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            encoded_lengths,
            target_lengths,
            blank=model.ctc_head.blank_id,
            zero_infinity=True,
        )
        return loss

    import torchaudio

    with autocast_ctx:
        joint_log_probs, encoded_lengths = model.forward_rnnt(
            waveforms, waveform_lengths, targets
        )
    loss = torchaudio.functional.rnnt_loss(
        joint_log_probs.float(),
        targets.int(),
        encoded_lengths.int(),
        target_lengths.int(),
        blank=model.prediction_network.blank_id,
        fused_log_softmax=False,
    )
    return loss


@torch.no_grad()
def log_sample_transcription(model, batch, tokenizer, device):
    """Greedy-decodes one random utterance from `batch` and prints it next to
    its reference transcript, so drift/collapse is visible without a full eval run."""
    waveforms, waveform_lengths, targets, target_lengths = batch
    idx = random.randrange(waveforms.size(0))

    model.eval()
    waveform = waveforms[idx : idx + 1].to(device)
    waveform_length = waveform_lengths[idx : idx + 1].to(device)
    log_probs, encoded_lengths = model.forward_ctc(waveform, waveform_length)
    model.train()

    predicted_ids = log_probs[0, : encoded_lengths[0].item()].argmax(dim=-1).tolist()
    collapsed = [token for token, _ in groupby(predicted_ids)]
    collapsed = [token for token in collapsed if token != model.ctc_head.blank_id]
    pred_text = tokenizer.decode(collapsed)

    true_ids = targets[idx][: target_lengths[idx].item()].tolist()
    true_text = tokenizer.decode(true_ids)

    tqdm.write(f"  [sample] true: {true_text}\n  [sample] pred: {pred_text}\n  {'-' * 60}")


@torch.no_grad()
def evaluate(model, val_loader, loss_type: str, device, amp_dtype=None) -> float:
    model.eval()
    total_loss, total_batches = 0.0, 0
    pbar = tqdm(val_loader, desc="val", leave=False)
    for batch in pbar:
        loss = compute_loss(model, batch, loss_type, device, amp_dtype)
        total_loss += loss.item()
        total_batches += 1
        pbar.set_postfix(loss=f"{total_loss / total_batches:.4f}")
    model.train()
    return total_loss / max(total_batches, 1)


def save_checkpoint(
    path: Path, model, optimizer, scheduler, scaler, epoch: int, step: int, best_val_loss: float,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def main():
    cli_args = build_arg_parser().parse_args()
    args = load_sections(cli_args.config, "manifests", "model", "train")
    wandb_args = load_sections(cli_args.config, "wandb")

    require_existing_paths(
        cli_args.config,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    run = None
    if wandb_args.wandb_enabled:
        import wandb

        if wandb_args.wandb_api_key:
            wandb.login(key=wandb_args.wandb_api_key)

        run = wandb.init(
            project=wandb_args.wandb_project,
            entity=wandb_args.wandb_entity,
            name=wandb_args.wandb_run_name,
            config={**vars(args), "device": str(device)},
        )

    vocab_path = output_dir / "tokenizer.model"
    if vocab_path.exists():
        tokenizer = BPETokenizer.load(str(vocab_path))
    else:
        tokenizer = BPETokenizer.build_from_manifests(
            [args.train_manifest, args.val_manifest],
            vocab_size=args.tokenizer_vocab_size,
            model_type=args.tokenizer_model_type,
        )
        tokenizer.save(str(vocab_path))
    print(f"Vocab size: {tokenizer.vocab_size} ({vocab_path})")

    train_loader = build_dataloader(
        args.train_manifest, tokenizer, args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = build_dataloader(
        args.val_manifest, tokenizer, args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    model = build_model(args, tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_decay_scale(step, args.warmup_steps)
    )

    amp_dtype = AMP_DTYPES[args.precision]
    scaler = torch.amp.GradScaler(device.type, enabled=(amp_dtype == torch.float16))
    print(f"Precision: {args.precision}")

    start_epoch, step, best_val_loss = 0, 0, float("inf")
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        step = ckpt["step"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from {args.resume_from} at epoch {start_epoch}, step {step}")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            step += 1

            optimizer.zero_grad()
            loss = compute_loss(model, batch, args.loss, device, amp_dtype)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}", step=step)

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                if run is not None:
                    run.log({"train/loss": avg_loss, "train/lr": lr, "epoch": epoch}, step=step)
                running_loss = 0.0

            if args.loss == "ctc" and step % args.transcribe_interval == 0:
                log_sample_transcription(model, batch, tokenizer, device)

        val_loss = evaluate(model, val_loader, args.loss, device, amp_dtype)
        elapsed = time.time() - epoch_start
        print(f"epoch {epoch} done in {elapsed:.1f}s -- val_loss {val_loss:.4f}")
        if run is not None:
            run.log({"val/loss": val_loss, "epoch": epoch}, step=step)

        save_checkpoint(
            output_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, step, best_val_loss,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                output_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, step, best_val_loss,
            )
            print(f"New best val_loss {best_val_loss:.4f}, saved {output_dir / 'best.pt'}")

    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(
            {"epochs": args.epochs, "final_step": step, "best_val_loss": best_val_loss}, f, indent=2,
        )

    print(f"All run artifacts -> {output_dir}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
