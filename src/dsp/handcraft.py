import numpy as np

def frame_signal(y, frame_size, hop_size):
    n_frames = 1 + int((len(y) - frame_size) / hop_size)
    frames = np.zeros((n_frames, frame_size))
    for i in range(n_frames):
        frames[i] = y[i * hop_size : i * hop_size + frame_size]
    
    frames *= np.hamming(frame_size)
    return frames
def magnitude_spectrum(frames, n_fft):
    mag_spec = np.abs(np.fft.rfft(frames, n=n_fft))
    power_spec = mag_spec**2
    return power_spec

def compute_rms(frames):
    return np.sqrt(np.mean(frames**2, axis=1))

def compute_zcr(frames):
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    return zcr

def compute_spectral_centroid(power_spec, sr, n_fft):
    freqs = np.linspace(0, sr / 2, power_spec.shape[1])
    centroid = np.sum(power_spec * freqs, axis=1) / (np.sum(power_spec, axis=1) + 1e-10)
    return centroid

def compute_spectral_rolloff(power_spec, sr, n_fft, roll_percent=0.85):
    freqs = np.linspace(0, sr / 2, power_spec.shape[1])
    cumulative_power = np.cumsum(power_spec, axis=1)
    total_power = cumulative_power[:, -1:]
    threshold = roll_percent * total_power
    rolloff_idx = np.argmax(cumulative_power >= threshold, axis=1)
    return freqs[rolloff_idx]

def compute_spectral_flatness(power_spec):
    ps = power_spec + 1e-10
    geometric_mean = np.exp(np.mean(np.log(ps), axis=1))
    arithmetic_mean = np.mean(ps, axis=1)
    
    flatness = geometric_mean / arithmetic_mean
    return flatness

def extract_all_ml_features(y, sr, frame_size=400, hop_size=160, n_fft=512):
    frames = frame_signal(y, frame_size, hop_size)
    power_spec = magnitude_spectrum(frames, n_fft)
    
    rms = compute_rms(frames)
    zcr = compute_zcr(frames)
    centroid = compute_spectral_centroid(power_spec, sr, n_fft)
    rolloff = compute_spectral_rolloff(power_spec, sr, n_fft)
    flatness = compute_spectral_flatness(power_spec)
    
    features = []
    for feat in [rms, zcr, centroid, rolloff, flatness]:
        features.append(np.mean(feat))
        features.append(np.std(feat))
        
    return np.array(features)