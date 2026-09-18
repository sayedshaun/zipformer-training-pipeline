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
import os
import random
import time
from itertools import groupby
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.config import load_sections, require_existing_paths
from src.dataset import build_dataloader
from src.model import ZipformerFromScratch
from tokenizer import BPETokenizer


def setup_distributed():
    """Reads the rank/world-size torchrun injects into the environment.

    Returns (is_distributed, rank, local_rank, world_size). Running the script
    directly (no torchrun) leaves these unset and falls back to single-GPU.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # nccl is the only backend worth using for GPU collectives; gloo is the
    # fallback for a CPU-only smoke test of the distributed path.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


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


def unwrap(model):
    """The underlying ZipformerFromScratch, whether or not DDP wrapped it."""
    return model.module if hasattr(model, "module") else model


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
            # Call the module, not .forward_ctc - see ZipformerFromScratch.forward.
            log_probs, encoded_lengths = model(waveforms, waveform_lengths)
        log_probs = log_probs.float().transpose(0, 1)  # CTCLoss wants (T, B, C)
        loss = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            encoded_lengths,
            target_lengths,
            blank=unwrap(model).ctc_head.blank_id,
            zero_infinity=True,
        )
        return loss

    import torchaudio

    with autocast_ctx:
        joint_log_probs, encoded_lengths = model(waveforms, waveform_lengths, targets)
    loss = torchaudio.functional.rnnt_loss(
        joint_log_probs.float(),
        targets.int(),
        encoded_lengths.int(),
        target_lengths.int(),
        blank=unwrap(model).prediction_network.blank_id,
        fused_log_softmax=False,
    )
    return loss


@torch.no_grad()
def log_sample_transcription(model, batch, tokenizer, device):
    """Greedy-decodes one random utterance from `batch` and prints it next to
    its reference transcript, so drift/collapse is visible without a full eval run."""
    waveforms, waveform_lengths, targets, target_lengths = batch
    idx = random.randrange(waveforms.size(0))

    inner = unwrap(model)
    model.eval()
    waveform = waveforms[idx : idx + 1].to(device)
    waveform_length = waveform_lengths[idx : idx + 1].to(device)
    log_probs, encoded_lengths = inner.forward_ctc(waveform, waveform_length)
    model.train()

    predicted_ids = log_probs[0, : encoded_lengths[0].item()].argmax(dim=-1).tolist()
    collapsed = [token for token, _ in groupby(predicted_ids)]
    collapsed = [token for token in collapsed if token != inner.ctc_head.blank_id]
    pred_text = tokenizer.decode(collapsed)

    true_ids = targets[idx][: target_lengths[idx].item()].tolist()
    true_text = tokenizer.decode(true_ids)

    tqdm.write(f"  [sample] true: {true_text}\n  [sample] pred: {pred_text}\n  {'-' * 60}")


@torch.no_grad()
def evaluate(model, val_loader, loss_type: str, device, amp_dtype=None, is_main: bool = True) -> float:
    model.eval()
    total_loss, total_batches = 0.0, 0
    pbar = tqdm(val_loader, desc="val", leave=False, disable=not is_main)
    for batch in pbar:
        loss = compute_loss(model, batch, loss_type, device, amp_dtype)
        total_loss += loss.item()
        total_batches += 1
        pbar.set_postfix(loss=f"{total_loss / total_batches:.4f}")
    model.train()

    # Each rank saw a different shard of the val set. Reduce the summed loss and
    # batch count (not the per-rank means, which would mis-weight a short shard)
    # so every rank computes the same number and agrees on the best checkpoint.
    if dist.is_available() and dist.is_initialized():
        totals = torch.tensor([total_loss, float(total_batches)], device=device, dtype=torch.float64)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, total_batches = totals[0].item(), int(totals[1].item())

    return total_loss / max(total_batches, 1)


def save_checkpoint(
    path: Path, model, optimizer, scheduler, scaler, epoch: int, step: int, best_val_loss: float,
):
    # Unwrap DDP so the checkpoint has plain keys, not "module."-prefixed ones,
    # and stays loadable by eval.py / a single-GPU resume.
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(
        {
            "model_state_dict": state_dict,
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

    distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0

    require_existing_paths(
        cli_args.config,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
    )

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    if distributed and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if is_main:
        print(f"World size: {world_size} | device: {device}")

    # ZIPFORMER_WANDB lets the Kaggle kernel enable wandb only when its secret
    # actually resolved, without writing a key into the committed config.
    wandb_enabled = wandb_args.wandb_enabled
    env_wandb = os.environ.get("ZIPFORMER_WANDB")
    if env_wandb is not None:
        wandb_enabled = env_wandb == "1"

    run = None
    if is_main and wandb_enabled:
        import wandb

        if wandb_args.wandb_api_key:
            wandb.login(key=wandb_args.wandb_api_key)

        run = wandb.init(
            project=wandb_args.wandb_project,
            entity=wandb_args.wandb_entity,
            name=wandb_args.wandb_run_name,
            config={**vars(args), "device": str(device)},
        )

    # Only rank 0 trains the tokenizer; the others wait on the barrier and load
    # the file it wrote. Letting every rank train its own would race on the same
    # path and risk ranks disagreeing about token ids.
    vocab_path = output_dir / "tokenizer.model"
    if is_main and not vocab_path.exists():
        tokenizer = BPETokenizer.build_from_manifests(
            [args.train_manifest, args.val_manifest],
            vocab_size=args.tokenizer_vocab_size,
            model_type=args.tokenizer_model_type,
        )
        tokenizer.save(str(vocab_path))
    if distributed:
        dist.barrier()
    tokenizer = BPETokenizer.load(str(vocab_path))
    if is_main:
        print(f"Vocab size: {tokenizer.vocab_size} ({vocab_path})")

    train_loader = build_dataloader(
        args.train_manifest, tokenizer, args.batch_size, shuffle=True,
        num_workers=args.num_workers, distributed=distributed,
    )
    val_loader = build_dataloader(
        args.val_manifest, tokenizer, args.batch_size, shuffle=False,
        num_workers=args.num_workers, distributed=distributed,
    )

    model = build_model(args, tokenizer.vocab_size).to(device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
        )
    # Parameters must come from the DDP wrapper so the optimizer updates the
    # same tensors the reducer writes gradients into.
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
        skipped_steps, total_steps = 0, 0
        # Without set_epoch every epoch replays the same shuffle and the same
        # rank -> utterance assignment.
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=not is_main)
        for batch_idx, batch in enumerate(pbar):
            step += 1

            optimizer.zero_grad()
            loss = compute_loss(model, batch, args.loss, device, amp_dtype)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # A non-finite gradient makes scaler.step() silently skip the update
            # while scheduler.step() still advances, which looks exactly like a
            # plateau. Without this count there is no way to tell a model that
            # has converged from one that is barely being updated.
            if not torch.isfinite(grad_norm):
                skipped_steps += 1
            total_steps += 1

            running_loss += loss.item()
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}", step=step)

            if is_main and (batch_idx + 1) % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                skip_frac = skipped_steps / max(total_steps, 1)
                if run is not None:
                    run.log({"train/loss": avg_loss, "train/lr": lr,
                             "train/skipped_step_frac": skip_frac, "epoch": epoch}, step=step)
                if skip_frac > 0.05:
                    tqdm.write(
                        f"  [warn] {skip_frac:.1%} of steps skipped on non-finite grads "
                        f"(scale {scaler.get_scale():.0f}) - fp16 overflow"
                    )
                running_loss = 0.0

            # Epoch boundaries are hours apart on a full corpus, so without this
            # a crash mid-epoch loses every step since the last one. Writes
            # last.pt only - "best" is meaningless before a val pass.
            save_interval = getattr(args, "save_interval_steps", 0)
            if is_main and save_interval and step % save_interval == 0:
                save_checkpoint(
                    output_dir / "last.pt", model, optimizer, scheduler, scaler,
                    epoch, step, best_val_loss,
                )
                tqdm.write(f"  [checkpoint] step {step} -> {output_dir / 'last.pt'}")

            if is_main and args.loss == "ctc" and step % args.transcribe_interval == 0:
                log_sample_transcription(model, batch, tokenizer, device)

        val_loss = evaluate(model, val_loader, args.loss, device, amp_dtype, is_main)
        elapsed = time.time() - epoch_start
        if is_main:
            skip_frac = skipped_steps / max(total_steps, 1)
            print(
                f"epoch {epoch} done in {elapsed:.1f}s -- val_loss {val_loss:.4f} "
                f"-- skipped {skipped_steps}/{total_steps} steps ({skip_frac:.1%})"
            )
            if run is not None:
                run.log({"val/loss": val_loss, "epoch": epoch}, step=step)

        # val_loss is all-reduced, so every rank takes this branch identically -
        # but only rank 0 writes, or the ranks would clobber each other's file.
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        if is_main:
            save_checkpoint(
                output_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, step, best_val_loss,
            )
            if is_best:
                best_path = output_dir / "best.pt"
                save_checkpoint(
                    best_path, model, optimizer, scheduler, scaler, epoch, step, best_val_loss,
                )
                print(f"New best val_loss {best_val_loss:.4f}, saved {best_path}")
                # Upload it so the weights are retrievable while the run is still
                # going - the whole point on Kaggle, where /kaggle/working stays
                # sealed until the kernel exits.
                if run is not None:
                    import wandb

                    artifact = wandb.Artifact(f"{run.id}-checkpoint", type="model")
                    artifact.add_file(str(best_path))
                    run.log_artifact(artifact, aliases=["best", f"epoch-{epoch}"])

    if is_main:
        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(
                {"epochs": args.epochs, "final_step": step, "best_val_loss": best_val_loss}, f, indent=2,
            )
        print(f"All run artifacts -> {output_dir}")
        if run is not None:
            run.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
