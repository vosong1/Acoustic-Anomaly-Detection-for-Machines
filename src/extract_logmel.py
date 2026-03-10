import librosa
import numpy as np
from extract_base import run_feature_extraction


def extract_logmel_feature(y, sr, cfg):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg.get("n_fft", 1024),
        hop_length=cfg.get("hop_length", 512),
        win_length=cfg.get("win_length", 1024),
        n_mels=cfg.get("n_mels", 64),
    )
    return librosa.power_to_db(mel, ref=np.max)


if __name__ == "__main__":
    run_feature_extraction("logmel", extract_logmel_feature)