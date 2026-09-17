"""Pure-PyTorch dataset for the from-scratch Zipformer in src/model.py: reads
the {audio_filepath, text, duration} manifests prepare_data.py produces.
Tokenizer is injected (see tokenizer.py) - anything with
encode()/decode()/vocab_size works.
"""

import json

import soundfile as sf
import soxr
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler


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
            # Common Voice ships 32kHz MP3 and the corpus is mounted read-only on
            # Kaggle, so there is nowhere to write a resampled copy - resample per
            # item instead of pre-converting the way src/audio.py does.
            # soxr rather than torchaudio: torchaudio is frozen at 2.11 and has no
            # build for newer torch, so depending on it here would pin the repo's
            # torch version for what is a few lines of signal processing.
            waveform = soxr.resample(waveform, sr, self.sample_rate)

        waveform = torch.from_numpy(waveform)
        target = torch.tensor(self.tokenizer.encode(entry["text"]), dtype=torch.long)
        return waveform, target


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
) -> DataLoader:
    """`distributed` shards the manifest across ranks with a DistributedSampler.
    The caller owns the returned loader's `.sampler` and must call
    `sampler.set_epoch(epoch)` each epoch, or every rank reshuffles identically
    and each epoch sees the same rank->utterance assignment."""
    dataset = ManifestDataset(manifest_path, tokenizer, sample_rate)

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
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
