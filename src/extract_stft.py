import librosa
import numpy as np
from extract_base import run_feature_extraction


def extract_stft_feature(y, sr, cfg):
    stft = librosa.stft(
        y=y,
        n_fft=cfg.get("n_fft", 1024),
        hop_length=cfg.get("hop_length", 512),
        win_length=cfg.get("win_length", 1024),
    )
    return np.abs(stft)
    

if __name__ == "__main__":
    run_feature_extraction("stft", extract_stft_feature)