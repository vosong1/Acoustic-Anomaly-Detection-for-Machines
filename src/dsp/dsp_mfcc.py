import numpy as np
from scipy.fftpack import dct

def frame_signal(y, frame_size, hop_size):
    if len(y) < frame_size:
        pad_width = frame_size - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')
    
    pad_len = (frame_size - len(y) % hop_size) % hop_size
    y = np.pad(y, (0, pad_len), mode='constant')
    
    n_frames = 1 + (len(y) - frame_size) // hop_size
    frames = np.zeros((n_frames, frame_size))
    
    for i in range(n_frames):
        start = i * hop_size
        frames[i] = y[start:start + frame_size]
        
    window = np.hamming(frame_size)
    return frames * window

def power_spectrum(frames, n_fft):
    """Tính phổ năng lượng"""
    fft_val = np.fft.rfft(frames, n=n_fft)
    mag_spec = np.abs(fft_val)
    return (1.0 / n_fft) * (mag_spec ** 2)

def mel_filterbank(sr, n_fft, n_mels=128):
    """Tạo bộ lọc Mel thủ công"""
    def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel): return 700 * (10**(mel / 2595) - 1)

    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbank = np.zeros((n_mels, int(n_fft / 2 + 1)))
    
    for i in range(1, n_mels + 1):
        left, center, right = bins[i-1], bins[i], bins[i+1]
        for j in range(left, center):
            fbank[i-1, j] = (j - left) / (center - left)
        for j in range(center, right):
            fbank[i-1, j] = (right - j) / (right - center)
    return fbank

def compute_mfcc(y, sr, n_mfcc=20, frame_size=1024, hop_size=512, n_fft=1024):
    frames = frame_signal(y, frame_size, hop_size)
    pow_frames = power_spectrum(frames, n_fft)
    
    mel_fb = mel_filterbank(sr, n_fft, n_mels=128) 
    mel_energy = np.dot(pow_frames, mel_fb.T)

    mel_energy = np.where(mel_energy == 0, 1e-10, mel_energy)
    log_mel = 10 * np.log10(mel_energy)
    
    mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    return mfcc


def compute_handcrafted_features(y, sr, frame_size=1024, hop_size=512, n_fft=1024):
    frames = frame_signal(y, frame_size, hop_size)
    pow_spec = power_spectrum(frames, n_fft)
    
    rms = np.sqrt(np.mean(frames**2, axis=1))
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)

    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    if pow_spec.shape[1] < len(freqs): 
        freqs = freqs[:pow_spec.shape[1]]

    centroid = np.sum(pow_spec * freqs, axis=1) / (np.sum(pow_spec, axis=1) + 1e-10)
    
    g_mean = np.exp(np.mean(np.log(pow_spec + 1e-10), axis=1))
    a_mean = np.mean(pow_spec, axis=1) + 1e-10
    flatness = g_mean / a_mean

    flux = np.sqrt(np.sum(np.diff(pow_spec, axis=0)**2, axis=1))
    flux = np.pad(flux, (1, 0), mode='constant')
    
    max_amp = np.max(np.abs(frames), axis=1)

    features = []
    feature_list = [rms, zcr, centroid, flatness, flux, max_amp]
    
    for feat in feature_list:
        features.append(np.mean(feat)) 
        features.append(np.std(feat))  
        features.append(np.max(feat)) 
        
    return np.array(features)