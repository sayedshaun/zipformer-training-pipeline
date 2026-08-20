"""Pure-PyTorch dataset for the from-scratch FastConformer: reads the same
{audio_filepath, text, duration} manifests prepare_data.py produces. Tokenizer
is injected (see tokenizer.py) - anything with encode()/decode()/vocab_size
works.
"""

import json

import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


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
        if sr != self.sample_rate:
            raise ValueError(
                f"{entry['audio_filepath']}: expected {self.sample_rate}Hz, got {sr}Hz"
            )
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

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
) -> DataLoader:
    dataset = ManifestDataset(manifest_path, tokenizer, sample_rate)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
