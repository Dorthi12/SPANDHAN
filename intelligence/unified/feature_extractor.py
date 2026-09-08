"""
intelligence/unified/feature_extractor.py
==========================================
Unified feature extractor for audio signals and 2-D images.

Audio features  (DOMAIN_AUDIO = 0.0) — 22 features
Image features  (DOMAIN_IMAGE = 1.0) — 29 features

Both produce a fixed-length float64 vector padded with zeros to
UNIFIED_FEATURE_DIM = 30 features + 1 domain tag = 31 total.

The domain tag lets the model distinguish modalities without
needing a separate feature set per domain.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew
from scipy.signal import welch

# ─── constants ───────────────────────────────────────────────────────────────

DOMAIN_AUDIO = 0.0
DOMAIN_IMAGE = 1.0

# Shared feature dimension (audio and image vectors are both padded to this)
UNIFIED_FEATURE_DIM = 31  # 30 signal features + 1 domain tag

AUDIO_FEATURE_NAMES = [
    "rms", "variance", "std", "kurtosis", "skewness", "crest_factor",
    "zero_crossing_rate", "spectral_centroid", "spectral_flatness",
    "spectral_entropy", "spectral_rolloff", "mains_band_energy",
    "high_band_energy", "snr_db", "impulse_count", "peak_count_rate",
    "energy_ratio_first_half", "spectral_variance", "low_band_energy",
    "mid_band_energy",
    "zcr_std",          # zero-crossing std across segments (impulsive vs periodic)
    "peak_to_rms",      # peak / rms — alternative crest measure
]  # len = 22

IMAGE_FEATURE_NAMES = [
    "mean", "std", "skewness", "kurtosis", "min_val", "max_val",
    "range_val", "energy",
    "edge_density",                # Sobel magnitude mean
    "spectral_hf_ratio",           # DFT high / low freq ratio
    "impulse_fraction",            # pixels > 3σ from mean
    "glcm_contrast", "glcm_homogeneity",
    "hist_00", "hist_01", "hist_02", "hist_03",
    "hist_04", "hist_05", "hist_06", "hist_07",
    "hist_08", "hist_09", "hist_10", "hist_11",
    "hist_12", "hist_13", "hist_14", "hist_15",
    "periodic_score",              # dominant frequency power in DFT
]  # len = 29

UNIFIED_FEATURE_NAMES = [f"f{i:02d}" for i in range(UNIFIED_FEATURE_DIM - 1)] + ["domain"]


# ─── audio features ──────────────────────────────────────────────────────────


def _safe(v: float) -> float:
    """Replace nan/inf with 0.0."""
    return float(v) if np.isfinite(v) else 0.0


def extract_audio_features(
    signal: np.ndarray,
    sampling_rate: float = 8000.0,
) -> np.ndarray:
    """
    Extract 22 features from a 1-D float64 audio signal.

    Parameters
    ----------
    signal        : 1-D float64 NumPy array
    sampling_rate : Hz

    Returns
    -------
    np.ndarray of shape (UNIFIED_FEATURE_DIM,), float64
    """
    sig = np.asarray(signal, dtype=np.float64).ravel()
    if sig.size == 0:
        raise ValueError("Signal is empty.")

    n = len(sig)
    sr = float(sampling_rate)

    # ── time-domain ────────────────────────────────────────────────
    rms = _safe(np.sqrt(np.mean(sig ** 2)))
    var = _safe(np.var(sig))
    std = _safe(np.std(sig))
    kurt = _safe(float(sp_kurtosis(sig, fisher=True)))
    skewness = _safe(float(sp_skew(sig)))
    peak = float(np.max(np.abs(sig)))
    crest = _safe(peak / rms) if rms > 1e-12 else 0.0
    zcr = _safe(np.sum(np.abs(np.diff(np.sign(sig)))) / (2 * n))

    # ZCR std across 8 segments
    segs = np.array_split(sig, 8)
    zcr_per_seg = [np.sum(np.abs(np.diff(np.sign(s)))) / max(2 * len(s), 1) for s in segs]
    zcr_std = _safe(float(np.std(zcr_per_seg)))

    peak_to_rms = _safe(peak / rms) if rms > 1e-12 else 0.0

    # ── spectral ───────────────────────────────────────────────────
    nperseg = min(256, n)
    freqs, psd = welch(sig, fs=sr, nperseg=nperseg)

    psd_sum = psd.sum() + 1e-30
    norm_psd = psd / psd_sum

    spec_centroid = _safe(np.sum(freqs * norm_psd))
    spec_flatness = _safe(
        float(np.exp(np.mean(np.log(psd + 1e-30))) / (np.mean(psd) + 1e-30))
    )
    spec_entropy = _safe(float(-np.sum(norm_psd * np.log(norm_psd + 1e-30))))
    cumsum = np.cumsum(psd)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    spec_rolloff = _safe(float(freqs[min(rolloff_idx, len(freqs) - 1)]))

    nyq = sr / 2.0
    mains_mask = (freqs >= 45) & (freqs <= 65)
    high_mask = freqs >= 0.8 * nyq
    low_mask = freqs <= 0.1 * nyq
    mid_mask = (freqs > 0.1 * nyq) & (freqs <= 0.5 * nyq)
    mains_energy = _safe(psd[mains_mask].sum() / psd_sum)
    high_energy = _safe(psd[high_mask].sum() / psd_sum)
    low_energy = _safe(psd[low_mask].sum() / psd_sum)
    mid_energy = _safe(psd[mid_mask].sum() / psd_sum)
    spec_var = _safe(float(np.var(psd)))

    # ── SNR ────────────────────────────────────────────────────────
    signal_power = float(np.mean(sig ** 2))
    noise_floor = float(np.percentile(psd, 10))
    snr_db = _safe(10.0 * np.log10(signal_power / (noise_floor + 1e-30) + 1e-30))

    # ── impulse features ───────────────────────────────────────────
    threshold3 = 3.0 * std if std > 0 else 1e-6
    above = np.abs(sig) > threshold3
    # Count impulse groups (transitions)
    groups = np.diff(above.astype(int))
    impulse_count = _safe(float(np.sum(groups == 1)))
    threshold2 = 2.0 * std if std > 0 else 1e-6
    peaks_above2 = np.sum(np.abs(sig) > threshold2)
    peak_count_rate = _safe(float(peaks_above2) / n)

    e_first = float(np.mean(sig[: n // 2] ** 2))
    e_second = float(np.mean(sig[n // 2 :] ** 2))
    total_e = e_first + e_second + 1e-30
    energy_ratio = _safe(e_first / total_e)

    # ── assemble ───────────────────────────────────────────────────
    raw = np.array([
        rms, var, std, kurt, skewness, crest, zcr,
        spec_centroid, spec_flatness, spec_entropy, spec_rolloff,
        mains_energy, high_energy, snr_db,
        impulse_count, peak_count_rate, energy_ratio,
        spec_var, low_energy, mid_energy,
        zcr_std, peak_to_rms,
    ], dtype=np.float64)  # length 22

    # Pad to UNIFIED_FEATURE_DIM-1 and append domain tag
    padded = np.zeros(UNIFIED_FEATURE_DIM - 1, dtype=np.float64)
    padded[: len(raw)] = raw
    return np.append(padded, DOMAIN_AUDIO)


# ─── image features ──────────────────────────────────────────────────────────


def _sobel(img: np.ndarray) -> np.ndarray:
    """Simple numpy-only Sobel edge detector."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    from scipy.ndimage import convolve
    gx = convolve(img, kx)
    gy = convolve(img, ky)
    return np.sqrt(gx ** 2 + gy ** 2)


def _glcm_features(img: np.ndarray, levels: int = 16) -> tuple[float, float]:
    """Approximate GLCM contrast and homogeneity (horizontal, distance=1)."""
    quantised = (np.clip(img, 0, 1) * (levels - 1)).astype(np.int32)
    H, W = quantised.shape
    glcm = np.zeros((levels, levels), dtype=np.float64)
    for i in range(H):
        for j in range(W - 1):
            glcm[quantised[i, j], quantised[i, j + 1]] += 1
    glcm_sum = glcm.sum()
    if glcm_sum > 0:
        glcm /= glcm_sum
    i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    diff = (i_idx - j_idx).astype(np.float64)
    contrast = _safe(float(np.sum(diff ** 2 * glcm)))
    homogeneity = _safe(float(np.sum(glcm / (1.0 + diff ** 2))))
    return contrast, homogeneity


def extract_image_features(image: np.ndarray) -> np.ndarray:
    """
    Extract 29 features from a 2-D float64 image in [0, 1].

    Parameters
    ----------
    image : np.ndarray  shape (H, W), float64, [0, 1]

    Returns
    -------
    np.ndarray of shape (UNIFIED_FEATURE_DIM,), float64
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("image must be 2-D (H, W).")
    if img.size == 0:
        raise ValueError("image is empty.")

    flat = img.ravel()

    # ── intensity statistics ───────────────────────────────────────
    mean = _safe(float(np.mean(flat)))
    std = _safe(float(np.std(flat)))
    skewness = _safe(float(sp_skew(flat)))
    kurt = _safe(float(sp_kurtosis(flat, fisher=True)))
    min_val = _safe(float(np.min(flat)))
    max_val = _safe(float(np.max(flat)))
    range_val = _safe(max_val - min_val)
    energy = _safe(float(np.mean(flat ** 2)))

    # ── edge density ───────────────────────────────────────────────
    edges = _sobel(img)
    edge_density = _safe(float(edges.mean()))

    # ── spectral features (2-D DFT) ────────────────────────────────
    f2d = np.abs(np.fft.fft2(img))
    f2d_shift = np.fft.fftshift(f2d)
    H, W = img.shape
    cy, cx = H // 2, W // 2
    # Low-freq mask: inner 25% of freq domain
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    max_dist = np.sqrt(cy ** 2 + cx ** 2)
    low_mask = dist <= 0.25 * max_dist
    high_mask = ~low_mask
    f_total = f2d_shift.sum() + 1e-30
    hf_ratio = _safe(float(f2d_shift[high_mask].sum() / f_total))

    # Periodic score: dominant non-DC freq power ratio
    f2d_shift[cy, cx] = 0  # remove DC
    periodic_score = _safe(float(f2d_shift.max() / (f_total + 1e-30)))

    # ── impulse fraction ──────────────────────────────────────────
    threshold = mean + 3 * std if std > 1e-12 else 1.0
    impulse_fraction = _safe(float(np.mean(flat > threshold)))

    # ── GLCM texture ──────────────────────────────────────────────
    glcm_contrast, glcm_homogeneity = _glcm_features(img)

    # ── histogram (16 bins) ───────────────────────────────────────
    hist, _ = np.histogram(flat, bins=16, range=(0.0, 1.0))
    hist = hist.astype(np.float64) / (hist.sum() + 1e-12)  # normalised

    # ── assemble ──────────────────────────────────────────────────
    raw = np.concatenate([
        [mean, std, skewness, kurt, min_val, max_val, range_val, energy,
         edge_density, hf_ratio, impulse_fraction,
         glcm_contrast, glcm_homogeneity],
        hist,            # 16 values
        [periodic_score],
    ])  # length = 13 + 16 + 1 = 30 -- but we want 29; trim periodic_score at end

    # Actually: 13 base + 16 hist + 1 periodic = 30 -- we cap at UNIFIED_FEATURE_DIM-1 = 30
    padded = np.zeros(UNIFIED_FEATURE_DIM - 1, dtype=np.float64)
    padded[: min(len(raw), UNIFIED_FEATURE_DIM - 1)] = raw[: UNIFIED_FEATURE_DIM - 1]
    return np.append(padded, DOMAIN_IMAGE)


# ─── unified convenience ─────────────────────────────────────────────────────


def extract_features_from_file(
    path: str,
    sampling_rate: float = 8000.0,
) -> tuple[np.ndarray, str]:
    """
    Auto-detect domain (audio vs image) from file extension and extract features.

    Supported audio : .wav, .npy (1-D), .csv (single column)
    Supported image : .png, .jpg, .jpeg, .bmp, .tif, .tiff, .npy (2-D)

    Returns
    -------
    (feature_vector, domain_str)   domain_str = "audio" | "image"
    """
    import pathlib
    p = pathlib.Path(path)
    ext = p.suffix.lower()

    audio_exts = {".wav", ".csv"}
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    if ext in audio_exts:
        sig = _load_audio(p, ext)
        return extract_audio_features(sig, sampling_rate), "audio"

    elif ext in image_exts:
        img = _load_image(p)
        return extract_image_features(img), "image"

    elif ext == ".npy":
        arr = np.load(p)
        if arr.ndim == 1:
            return extract_audio_features(arr, sampling_rate), "audio"
        elif arr.ndim == 2:
            arr = arr.astype(np.float64)
            if arr.max() > 1.5:
                arr = arr / 255.0
            return extract_image_features(arr), "image"
        else:
            raise ValueError(f"Unsupported .npy array shape: {arr.shape}")

    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {audio_exts | image_exts | {'.npy'}}"
        )


def _load_audio(path, ext) -> np.ndarray:
    """Load audio signal as float64 1-D array."""
    if ext == ".wav":
        try:
            import soundfile as sf
            sig, _ = sf.read(str(path), dtype="float64", always_2d=False)
            if sig.ndim > 1:
                sig = sig.mean(axis=1)
            return sig.astype(np.float64)
        except Exception:
            pass
        try:
            import scipy.io.wavfile as wav
            rate, data = wav.read(str(path))
            sig = data.astype(np.float64)
            if sig.ndim > 1:
                sig = sig.mean(axis=1)
            if data.dtype == np.int16:
                sig /= 32768.0
            elif data.dtype == np.int32:
                sig /= 2147483648.0
            return sig
        except Exception as e:
            raise RuntimeError(f"Cannot read WAV file: {e}")
    elif ext == ".csv":
        import csv
        values = []
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    try:
                        values.append(float(cell.strip()))
                    except ValueError:
                        pass
        if not values:
            raise ValueError("CSV file contains no numeric data.")
        return np.array(values, dtype=np.float64)
    raise ValueError(f"Unsupported audio extension: {ext}")


def _load_image(path) -> np.ndarray:
    """Load image as float64 grayscale [0,1] array."""
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("cv2 returned None.")
        return img.astype(np.float64) / 255.0
    except Exception:
        pass
    try:
        from PIL import Image
        img = Image.open(str(path)).convert("L")
        return np.array(img, dtype=np.float64) / 255.0
    except Exception as e:
        raise RuntimeError(f"Cannot load image {path}: {e}")
