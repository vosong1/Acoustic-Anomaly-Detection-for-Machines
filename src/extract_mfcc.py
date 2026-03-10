import librosa
from extract_base import run_feature_extraction


def extract_mfcc_feature(y, sr, cfg):
    return librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=cfg.get("n_mfcc", 20),
        n_fft=cfg.get("n_fft", 1024),
        hop_length=cfg.get("hop_length", 512),
        win_length=cfg.get("win_length", 1024),
    )


if __name__ == "__main__":
    run_feature_extraction("mfcc", extract_mfcc_feature)