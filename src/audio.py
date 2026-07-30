"""Shared audio-clip conversion helper, used by both the mcv and openslr sources."""
import soundfile as sf
from pydub import AudioSegment

TARGET_SAMPLE_RATE = 16000


def convert_clip(args) -> float:
    src_path, dst_path = args
    if not dst_path.exists():
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)
        audio.export(dst_path, format="wav")
    with sf.SoundFile(dst_path) as f:
        return len(f) / f.samplerate
