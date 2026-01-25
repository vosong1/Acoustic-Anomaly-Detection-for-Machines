import numpy as np
import soundfile as sf

def load_audio(file_path, sr_target=None):
    y, sr = sf.read(file_path)
    
    if y.ndim == 2:
        y = y.mean(axis=1)

    if sr_target and sr != sr_target:
        import scipy.signal
        n = int(len(y) * sr_target / sr)
        y = scipy.signal.resample(y, n)
        sr = sr_target

    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr
