import numpy as np
from scipy.fftpack import dct

def frame_signal(y, frame_size, hop_size):
    pad_len = (frame_size - len(y) % hop_size) % hop_size
    y = np.pad(y, (0, pad_len))
    n_frames = 1 + (len(y) - frame_size) // hop_size
    frames = np.zeros((n_frames, frame_size))
    for i in range(n_frames):
        start = i * hop_size
        frames[i] = y[start:start + frame_size]
    return frames


def magnitude_spectrum(frames, n_fft):
    window = np.hamming(frames.shape[1])
    frames = frames * window
    fft = np.fft.rfft(frames, n=n_fft)
    return np.abs(fft) ** 2


def mel_filterbank(sr, n_fft, n_mels=20):
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbanks = np.zeros((n_mels, int(n_fft / 2 + 1)))

    for i in range(1, n_mels + 1):
        left, center, right = bins[i-1], bins[i], bins[i+1]
        for j in range(left, center):
            fbanks[i-1, j] = (j - left) / (center - left)
        for j in range(center, right):
            fbanks[i-1, j] = (right - j) / (right - center)

    return fbanks


def compute_mfcc(y, sr, n_mfcc=20, frame_size=400, hop_size=160, n_fft=512):
    frames = frame_signal(y, frame_size, hop_size)
    power_spec = magnitude_spectrum(frames, n_fft)
    mel_fb = mel_filterbank(sr, n_fft, n_mels=n_mfcc)
    mel_energy = np.dot(power_spec, mel_fb.T)
    mel_energy = np.where(mel_energy == 0, 1e-10, mel_energy)

    log_mel = np.log(mel_energy)
    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    return mfcc
