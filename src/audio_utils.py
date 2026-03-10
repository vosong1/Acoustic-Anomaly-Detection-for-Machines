from pathlib import Path
import librosa
import numpy as np


def load_audio(file_path: str | Path, sr: int = 16000, mono: bool = True, duration: float | None = None):
    y, sr = librosa.load(file_path, sr=sr, mono=mono, duration=duration)
    return y, sr


def feature_to_1d(feature: np.ndarray) -> np.ndarray:
    """
    Convert 2D feature (freq x time) -> 1D vector by averaging across time axis.
    """
    if feature.ndim == 2:
        return np.mean(feature, axis=1)
    return feature