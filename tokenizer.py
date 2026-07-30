"""SentencePiece BPE tokenizer trained on the {audio_filepath, text, duration}
manifests produced by prepare_data.py (Mozilla Common Voice transcripts).

Piece ids are dense in [0, vocab_size) with no reserved bos/eos - model.py
appends its own CTC/RNNT blank at id `vocab_size`, same convention the old
character-level tokenizer used.
"""

import json
import tempfile
from pathlib import Path

import sentencepiece as spm


class BPETokenizer:
    def __init__(self, sp: spm.SentencePieceProcessor):
        self.sp = sp

    @classmethod
    def build_from_manifests(
        cls, manifest_paths, vocab_size: int = 1024, model_type: str = "bpe"
    ) -> "BPETokenizer":
        with tempfile.TemporaryDirectory() as tmp_dir:
            text_path = Path(tmp_dir) / "corpus.txt"
            with open(text_path, "w", encoding="utf-8") as out:
                for manifest_path in manifest_paths:
                    with open(manifest_path) as f:
                        for line in f:
                            out.write(json.loads(line)["text"] + "\n")

            model_prefix = str(Path(tmp_dir) / "tokenizer")
            spm.SentencePieceTrainer.train(
                input=str(text_path),
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=1.0,  # Bengali script has many rare conjuncts
                unk_id=0,
                bos_id=-1,
                eos_id=-1,
                pad_id=-1,
            )
            sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
        return cls(sp)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        sp = spm.SentencePieceProcessor(model_file=path)
        return cls(sp)

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.sp.serialized_model_proto())

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list) -> str:
        return self.sp.decode([i for i in ids if i >= 0])
