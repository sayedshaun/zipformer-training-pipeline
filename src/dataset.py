"""Pure-PyTorch dataset for the from-scratch Zipformer in src/model.py: reads
the {audio_filepath, text, duration} manifests prepare_data.py produces.
Tokenizer is injected (see tokenizer.py) - anything with
encode()/decode()/vocab_size works.
"""

import json
import random

import soundfile as sf
import soxr
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

# 100 fps log-mel (hop 160 at 16kHz), then three stride-2 convs in ConvSubsampling.
ENCODER_FRAMES_PER_SECOND = 12.5


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: str, tokenizer, sample_rate: int = 16000):
        self.entries = []
        with open(manifest_path) as f:
            for line in f:
                # Manifests are concatenated from per-source parts; tolerate a
                # stray blank line rather than dying on json.loads("").
                if line.strip():
                    self.entries.append(json.loads(line))
        if not self.entries:
            raise ValueError(f"{manifest_path} contains no utterances")
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        waveform, sr = sf.read(entry["audio_filepath"], dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if sr != self.sample_rate:
            # Common Voice ships 32kHz MP3 and the corpus is mounted read-only,
            # so there is nowhere to write a resampled copy - resample per item
            # instead of pre-converting the way src/audio.py does.
            # soxr rather than torchaudio: torchaudio is frozen at 2.11 and has no
            # build for newer torch, so depending on it here would pin the repo's
            # torch version for what is a few lines of signal processing.
            waveform = soxr.resample(waveform, sr, self.sample_rate)

        waveform = torch.from_numpy(waveform)
        target = torch.tensor(self.tokenizer.encode(entry["text"]), dtype=torch.long)
        return waveform, target


class LengthBudgetBatchSampler(Sampler):
    """Batches similar-length utterances, capped by padded cost instead of count.

    A fixed batch size is sized for the worst batch: RNNT's joint tensor is
    batch x frames x tokens x vocab, so one batch of 30-40s clips needs many
    times the memory of a typical one, and every batch pays for its longest
    member's padding. Here each batch is closed once adding the next
    (length-sorted) utterance would exceed max_batch_seconds of padded audio
    (bounds encoder activations) or max_joint_cells padded frames x tokens
    (bounds the joint). One utterance over budget still forms its own batch.
    """

    def __init__(self, durations, token_counts, max_batch_seconds, max_joint_cells=None,
                 max_batch_size=None, shuffle=True, seed=0):
        self.durations = durations
        self.token_counts = token_counts
        self.max_batch_seconds = max_batch_seconds
        self.max_joint_cells = max_joint_cells
        self.max_batch_size = max_batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.set_epoch(0)

    def set_epoch(self, epoch: int):
        rng = random.Random(self.seed + epoch)
        # Jitter the sort key so batch membership changes between epochs while
        # batches still hold near-identical lengths.
        jitter = 0.5 if self.shuffle else 0.0
        order = sorted(range(len(self.durations)),
                       key=lambda i: self.durations[i] + rng.uniform(0, jitter))

        batches, batch, max_dur, max_tok = [], [], 0.0, 0
        for i in order:
            dur = max(max_dur, self.durations[i])
            tok = max(max_tok, self.token_counts[i])
            size = len(batch) + 1
            cells = size * (int(dur * ENCODER_FRAMES_PER_SECOND) + 2) * (tok + 1)
            over = batch and (
                size * dur > self.max_batch_seconds
                or (self.max_joint_cells and cells > self.max_joint_cells)
                or (self.max_batch_size and size > self.max_batch_size)
            )
            if over:
                batches.append(batch)
                batch, dur, tok = [], self.durations[i], self.token_counts[i]
            batch.append(i)
            max_dur, max_tok = dur, tok
        if batch:
            batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collate_fn(batch):
    waveforms, targets = zip(*batch)
    waveform_lengths = torch.tensor([w.numel() for w in waveforms], dtype=torch.long)
    target_lengths = torch.tensor([t.numel() for t in targets], dtype=torch.long)

    padded_waveforms = pad_sequence(waveforms, batch_first=True)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    return padded_waveforms, waveform_lengths, padded_targets, target_lengths


def build_dataloader(
    manifest_path: str,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    sample_rate: int = 16000,
    distributed: bool = False,
    pin_memory: bool = True,
    max_batch_seconds: float = None,
    max_joint_cells: int = None,
) -> DataLoader:
    """`distributed` shards the manifest across ranks with a DistributedSampler.
    The caller owns the returned loader's `.sampler` and must call
    `sampler.set_epoch(epoch)` each epoch, or every rank reshuffles identically
    and each epoch sees the same rank->utterance assignment.

    `pin_memory` allocates CUDA page-locked host memory regardless of which
    device the caller passes tensors to, so it should be False for a CPU run -
    otherwise it competes with whatever else is using the GPU and can OOM even
    though the run itself never touches CUDA."""
    dataset = ManifestDataset(manifest_path, tokenizer, sample_rate)

    if max_batch_seconds:
        # With max_batch_seconds set, batch_size is only an upper bound on count.
        if distributed:
            raise ValueError("max_batch_seconds batching is not implemented for distributed training")
        batch_sampler = LengthBudgetBatchSampler(
            durations=[float(e["duration"]) for e in dataset.entries],
            token_counts=[len(tokenizer.encode(e["text"])) for e in dataset.entries],
            max_batch_seconds=max_batch_seconds,
            max_joint_cells=max_joint_cells,
            max_batch_size=batch_size,
            shuffle=shuffle,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        # Sampler and shuffle are mutually exclusive; the sampler does the shuffling.
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
