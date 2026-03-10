import librosa
from extract_base import run_feature_extraction


def extract_chroma_feature(y, sr, cfg):
    return librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=cfg.get("n_fft", 1024),
        hop_length=cfg.get("hop_length", 512),
        n_chroma=cfg.get("n_chroma", 12),
    )


if __name__ == "__main__":
    run_feature_extraction("chroma", extract_chroma_feature)