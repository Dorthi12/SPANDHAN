import numpy as np

def extract_audio_features(signal, sampling_rate=44100):
    """Extract key time and frequency domain features from audio signal."""
    mean_amp = float(np.mean(np.abs(signal)))
    std_amp = float(np.std(signal))
    zero_crossings = int(np.sum(np.diff(np.sign(signal)) != 0))
    fft_vals = np.abs(np.fft.rfft(signal))
    spectral_centroid = float(np.sum(np.arange(len(fft_vals)) * fft_vals) / (np.sum(fft_vals) + 1e-10))
    
    return np.array([mean_amp, std_amp, zero_crossings, spectral_centroid])
