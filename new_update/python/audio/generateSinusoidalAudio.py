#!/usr/bin/env python3
"""
SPANDHAN Signal Processing Project
Synthetic Sinusoidal Audio Dataset Generator

Generates a research-grade synthetic audio dataset containing 1,000 sinusoidal WAV samples
with randomized physical and DSP parameters, detailed metadata CSV, dataset validation,
and preview plots.

Author: SPANDHAN Audio Signal Processing Team
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import spectrogram, get_window
import matplotlib.pyplot as plt


def generate_sinusoidal_signal(sample_id: int, seed: int = None) -> tuple[np.ndarray, int, dict]:
    """
    Generates a single synthetic sinusoidal audio signal.

    Mathematical formulation:
        x(t) = A(t) * sin(2*pi*f*t + phi) + n(t)

    Parameters:
        sample_id (int): Unique integer ID for the sample (1..1000)
        seed (int, optional): Random seed for reproducible sample synthesis

    Returns:
        signal (np.ndarray): 1D float32 array normalized to [-1.0, 1.0] max bound
        fs (int): Sampling rate in Hz
        metadata (dict): Parameter metadata dictionary for CSV export
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # 1. SAMPLING FREQUENCY
    valid_fs = [8000, 16000, 22050, 44100, 48000]
    fs = int(rng.choice(valid_fs))

    # 2. DURATION & TIME VECTOR
    duration_sec = float(rng.uniform(0.5, 3.0))
    num_samples = int(np.round(duration_sec * fs))
    t = np.arange(num_samples, dtype=np.float64) / fs

    # 3. FREQUENCY SELECTION
    # Maintain Nyquist margin f < 0.4 * fs, max frequency 3500 Hz
    f_min = 20.0
    f_max = min(3500.0, 0.4 * fs)

    # Continuous log-uniform frequency distribution
    log_f_min = np.log10(f_min)
    log_f_max = np.log10(f_max)
    f = float(10 ** rng.uniform(log_f_min, log_f_max))

    # 4. AMPLITUDE & PHASE
    amp = float(rng.uniform(0.2, 1.0))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    phi_deg = float(np.degrees(phi))

    # 5. SINUSOIDAL WAVEFORM SYNTHESIS
    x_clean = amp * np.sin(2.0 * np.pi * f * t + phi)

    # 6. OPTIONAL MILD AMPLITUDE ENVELOPE (~10% of samples)
    # Slow AM modulation with small depth (5%-25%) so sinusoid dominance is preserved
    if rng.random() < 0.10:
        mod_depth = float(rng.uniform(0.05, 0.25))
        f_mod = float(rng.uniform(0.5, 2.0))
        phi_mod = float(rng.uniform(0.0, 2.0 * np.pi))
        envelope = 1.0 - mod_depth * (0.5 + 0.5 * np.sin(2.0 * np.pi * f_mod * t + phi_mod))
        x_clean = x_clean * envelope

    # 7. ADDITIVE GAUSSIAN WHITE NOISE (SNR)
    snr_options = [np.inf, 40.0, 30.0, 20.0, 10.0]
    snr_idx = int(rng.integers(0, len(snr_options)))
    snr_db = float(snr_options[snr_idx])

    if np.isinf(snr_db):
        signal_raw = x_clean
        noise_level = "clean"
    else:
        p_signal = np.mean(x_clean ** 2)
        p_noise = p_signal / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(loc=0.0, scale=np.sqrt(p_noise), size=num_samples)
        signal_raw = x_clean + noise
        noise_level = f"{int(snr_db)}dB"

    # 8. SIGNAL INTEGRITY & NORMALIZATION
    if np.any(np.isnan(signal_raw)) or np.any(np.isinf(signal_raw)):
        raise ValueError(f"Sample {sample_id} generated invalid NaN or Inf values.")

    max_val = float(np.max(np.abs(signal_raw)))
    if max_val > 1.0:
        signal = signal_raw / max_val
        normalized = 1
    else:
        signal = signal_raw
        normalized = 0

    signal = signal.astype(np.float32)
    num_cycles = float(f * duration_sec)

    metadata = {
        "file_name": f"sinusoidal_{sample_id:04d}.wav",
        "class": "sinusoidal",
        "sample_id": sample_id,
        "sampling_rate": fs,
        "duration_sec": round(duration_sec, 4),
        "num_samples": num_samples,
        "frequency_hz": round(f, 2),
        "amplitude": round(amp, 4),
        "phase_rad": round(phi, 4),
        "phase_deg": round(phi_deg, 2),
        "snr_db": snr_db,
        "noise_level": noise_level,
        "num_cycles": round(num_cycles, 2),
        "normalized": normalized,
        "seed": seed if seed is not None else 0,
    }

    return signal, fs, metadata


def validate_sinusoidal_dataset(dataset_dir: str, metadata_path: str) -> tuple[bool, dict]:
    """
    Validates dataset completeness, signal integrity, and DSP spectral properties.
    """
    report = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "metadata_rows": 0,
        "errors": [],
        "warnings": [],
    }

    if not os.path.exists(metadata_path):
        report["errors"].append(f"Metadata file missing: {metadata_path}")
        return False, report

    df = pd.read_csv(metadata_path)
    report["metadata_rows"] = len(df)

    if len(df) != 1000:
        report["errors"].append(f"Metadata row count is {len(df)}, expected 1000.")

    valid_fs_set = {8000, 16000, 22050, 44100, 48000}

    wav_files = [f for f in os.listdir(dataset_dir) if f.endswith(".wav")]
    report["total_files"] = len(wav_files)

    if len(wav_files) != 1000:
        report["errors"].append(f"Found {len(wav_files)} WAV files, expected 1000.")

    for i in range(1, 1001):
        file_name = f"sinusoidal_{i:04d}.wav"
        file_path = os.path.join(dataset_dir, file_name)

        if not os.path.exists(file_path):
            report["invalid_files"] += 1
            report["errors"].append(f"Missing WAV file: {file_name}")
            continue

        try:
            fs, data = wavfile.read(file_path)
        except Exception as e:
            report["invalid_files"] += 1
            report["errors"].append(f"Corrupt WAV file {file_name}: {e}")
            continue

        # Convert to float32 normalized [-1, 1] if int format
        if data.dtype == np.int16:
            x = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            x = data.astype(np.float32) / 2147483648.0
        else:
            x = data.astype(np.float32)

        # Check NaN / Inf
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            report["invalid_files"] += 1
            report["errors"].append(f"{file_name} contains NaN/Inf values.")
            continue

        # Check clipping
        max_abs = np.max(np.abs(x))
        if max_abs > 1.0001:
            report["invalid_files"] += 1
            report["errors"].append(f"{file_name} clipped with max amplitude {max_abs:.4f}.")
            continue

        # Check sampling rate
        if fs not in valid_fs_set:
            report["invalid_files"] += 1
            report["errors"].append(f"{file_name} unexpected sampling rate {fs} Hz.")
            continue

        # Metadata match
        row_match = df[df["sample_id"] == i]
        if row_match.empty:
            report["invalid_files"] += 1
            report["errors"].append(f"Sample ID {i} missing from metadata.")
            continue

        row = row_match.iloc[0]
        stored_f = float(row["frequency_hz"])

        # Nyquist check
        if stored_f >= 0.5 * fs:
            report["invalid_files"] += 1
            report["errors"].append(f"{file_name} frequency {stored_f} Hz exceeds Nyquist limit.")
            continue

        # DSP Peak FFT check
        N = len(x)
        Nfft = max(2048, int(2 ** np.ceil(np.log2(N))))
        window = get_window("hann", N)
        X_fft = np.abs(np.fft.rfft(x * window, n=Nfft))
        freq_axis = np.fft.rfftfreq(Nfft, d=1.0 / fs)

        max_bin = np.argmax(X_fft)
        est_f = freq_axis[max_bin]

        bin_res = fs / Nfft
        abs_err = abs(est_f - stored_f)

        if abs_err > 5 * bin_res and abs_err / max(1.0, stored_f) > 0.10 and row["snr_db"] > 10:
            report["warnings"].append(
                f"{file_name} FFT peak f_est={est_f:.1f}Hz vs stored={stored_f:.1f}Hz"
            )

        report["valid_files"] += 1

    is_valid = (report["invalid_files"] == 0) and (len(report["errors"]) == 0)
    return is_valid, report


def visualize_sinusoidal_dataset(dataset_dir: str, metadata_path: str, save_path: str) -> None:
    """
    Generates preview plot displaying time-domain waveforms, zoomed cycles, FFT spectra, and spectrograms.
    """
    df = pd.read_csv(metadata_path)

    # Select 4 representative samples
    clean_mask = (df["snr_db"] == np.inf) | (df["noise_level"] == "clean")
    clean_df = df[clean_mask].sort_values("frequency_hz")

    if len(clean_df) >= 2:
        s1 = int(clean_df.iloc[0]["sample_id"])  # Lowest freq clean
        s2 = int(clean_df.iloc[-1]["sample_id"]) # Highest freq clean
    else:
        s1, s2 = 1, 10

    snr20_df = df[(df["snr_db"] == 20) | (df["noise_level"] == "20dB")]
    s3 = int(snr20_df.iloc[0]["sample_id"]) if not snr20_df.empty else 50

    snr10_df = df[(df["snr_db"] == 10) | (df["noise_level"] == "10dB")]
    s4 = int(snr10_df.iloc[0]["sample_id"]) if not snr10_df.empty else 100

    sample_ids = [s1, s2, s3, s4]

    fig, axes = plt.subplots(4, 4, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        "SPANDHAN Synthetic Sinusoidal Dataset - Representative Waveform & Spectral Analysis",
        fontsize=14,
        fontweight="bold",
    )

    for row_idx, sample_id in enumerate(sample_ids):
        row = df[df["sample_id"] == sample_id].iloc[0]
        file_name = str(row["file_name"])
        file_path = os.path.join(dataset_dir, file_name)

        fs, data = wavfile.read(file_path)
        if data.dtype == np.int16:
            x = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            x = data.astype(np.float32) / 2147483648.0
        else:
            x = data.astype(np.float32)

        f_gen = float(row["frequency_hz"])
        snr_str = str(row["noise_level"])
        dur = float(row["duration_sec"])
        N = len(x)
        time_vec = np.arange(N) / fs

        # 1. Full Waveform
        ax0 = axes[row_idx, 0]
        ax0.plot(time_vec, x, color="#0072BD", linewidth=0.8)
        ax0.set_xlabel("Time (s)", fontsize=8)
        ax0.set_ylabel("Amplitude", fontsize=8)
        ax0.set_title(f"Sample #{sample_id}: Full Waveform (f={f_gen:.1f}Hz, SNR={snr_str})", fontsize=9, fontweight="bold")
        ax0.set_ylim(-1.1, 1.1)
        ax0.grid(True, linestyle=":", alpha=0.6)

        # 2. Zoomed Waveform
        ax1 = axes[row_idx, 1]
        cycles_to_show = 8
        t_zoom_end = min(dur, cycles_to_show / max(20.0, f_gen))
        zoom_samples = max(20, int(np.round(t_zoom_end * fs)))
        ax1.plot(time_vec[:zoom_samples], x[:zoom_samples], color="#D95319", linewidth=1.2)
        ax1.set_xlabel("Time (s)", fontsize=8)
        ax1.set_ylabel("Amplitude", fontsize=8)
        ax1.set_title(f"Zoomed Sinusoid (First {cycles_to_show} Cycles)", fontsize=9, fontweight="bold")
        ax1.set_ylim(-1.1, 1.1)
        ax1.grid(True, linestyle=":", alpha=0.6)

        # 3. FFT Magnitude Spectrum
        ax2 = axes[row_idx, 2]
        Nfft = max(4096, int(2 ** np.ceil(np.log2(N))))
        window = get_window("hann", N)
        X_fft = np.abs(np.fft.rfft(x * window, n=Nfft)) / N
        freq_axis = np.fft.rfftfreq(Nfft, d=1.0 / fs)
        mag_db = 20.0 * np.log10(X_fft + 1e-6)

        ax2.plot(freq_axis, mag_db, color="#7E2F8E", linewidth=1.0)
        ax2.axvline(f_gen, color="red", linestyle="--", linewidth=1.2, label=f"f={f_gen:.1f}Hz")
        ax2.set_xlabel("Frequency (Hz)", fontsize=8)
        ax2.set_ylabel("Magnitude (dB)", fontsize=8)
        ax2.set_xlim(0, min(fs / 2, max(4000.0, f_gen * 1.5)))
        ax2.set_title("FFT Spectrum & Peak", fontsize=9, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # 4. Spectrogram
        ax3 = axes[row_idx, 3]
        nperseg = max(128, int(np.round(0.04 * fs)))
        noverlap = int(0.75 * nperseg)
        f_spec, t_spec, Sxx = spectrogram(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap)
        spec_db = 10.0 * np.log10(Sxx + 1e-6)

        im = ax3.pcolormesh(t_spec, f_spec, spec_db, shading="gouraud", cmap="jet")
        ax3.set_xlabel("Time (s)", fontsize=8)
        ax3.set_ylabel("Freq (Hz)", fontsize=8)
        ax3.set_ylim(0, min(fs / 2, max(4000.0, f_gen * 1.5)))
        ax3.set_title("Spectrogram (Stable Frequency)", fontsize=9, fontweight="bold")
        fig.colorbar(im, ax=ax3, format="%+2.0fdB")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Visualization figure saved to: {save_path}")


def generate_sinusoidal_dataset(
    output_dir: str = "datasets/audio/sinusoidal",
    metadata_path: str = "datasets/audio/sinusoidal_metadata.csv",
    preview_path: str = "datasets/audio/sinusoidal_dataset_preview.png",
    num_samples: int = 1000,
    master_seed: int = 42,
) -> None:
    """
    Master function to generate 1,000 synthetic sinusoidal audio signals, export WAV files,
    create metadata CSV, run validation suite, and render preview plots.
    """
    print("========================================")
    print("SPANDHAN SINUSOIDAL DATASET GENERATOR")
    print("========================================\n")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)

    metadata_list = []

    print(f"Generating {num_samples} synthetic sinusoidal audio samples...")

    for i in range(1, num_samples + 1):
        sample_seed = master_seed * 10000 + i
        signal, fs, meta = generate_sinusoidal_signal(i, sample_seed)

        # Save WAV file as 16-bit PCM for universal compatibility
        wav_path = os.path.join(output_dir, meta["file_name"])
        scaled_signal = np.int16(signal * 32767.0)
        wavfile.write(wav_path, fs, scaled_signal)

        metadata_list.append(meta)

        if i % 200 == 0 or i == num_samples:
            print(f"  Progress: {i} / {num_samples} samples generated...")

    # Create Metadata DataFrame & CSV Export
    df_meta = pd.DataFrame(metadata_list)
    df_meta.to_csv(metadata_path, index=False)
    print(f"\nMetadata CSV exported to: {metadata_path}\n")

    # Run Validation
    print("Running automated dataset integrity validation...")
    is_valid, val_report = validate_sinusoidal_dataset(output_dir, metadata_path)

    # Run Visualization
    print("Generating dataset preview visualization plot...")
    try:
        visualize_sinusoidal_dataset(output_dir, metadata_path, preview_path)
    except Exception as e:
        print(f"[WARNING] Visualization generation error: {e}")

    # Compute Terminal Report Stats
    freqs = df_meta["frequency_hz"].values
    amps = df_meta["amplitude"].values
    fs_vals = df_meta["sampling_rate"].values
    snr_vals = df_meta["snr_db"].values

    cnt_8k = int(np.sum(fs_vals == 8000))
    cnt_16k = int(np.sum(fs_vals == 16000))
    cnt_22k = int(np.sum(fs_vals == 22050))
    cnt_44k = int(np.sum(fs_vals == 44100))
    cnt_48k = int(np.sum(fs_vals == 48000))

    cnt_clean = int(np.sum(np.isinf(snr_vals)))
    cnt_40db = int(np.sum(snr_vals == 40))
    cnt_30db = int(np.sum(snr_vals == 30))
    cnt_20db = int(np.sum(snr_vals == 20))
    cnt_10db = int(np.sum(snr_vals == 10))

    status_str = "SUCCESS" if is_valid else "FAILED (Validation Errors Detected)"

    print("\n========================================")
    print("SPANDHAN SINUSOIDAL DATASET")
    print("========================================\n")
    print(f"Total samples       : {val_report['total_files']}")
    print(f"Valid WAV files     : {val_report['valid_files']}")
    print(f"Invalid files       : {val_report['invalid_files']}\n")

    print("Frequency range:")
    print(f"Minimum             : {np.min(freqs):.2f} Hz")
    print(f"Maximum             : {np.max(freqs):.2f} Hz")
    print(f"Mean                : {np.mean(freqs):.2f} Hz\n")

    print("Sampling rates:")
    print(f"8000 Hz             : {cnt_8k}")
    print(f"16000 Hz            : {cnt_16k}")
    print(f"22050 Hz            : {cnt_22k}")
    print(f"44100 Hz            : {cnt_44k}")
    print(f"48000 Hz            : {cnt_48k}\n")

    print("Amplitude range:")
    print(f"Minimum             : {np.min(amps):.4f}")
    print(f"Maximum             : {np.max(amps):.4f}\n")

    print("SNR distribution:")
    print(f"Clean               : {cnt_clean}")
    print(f"40 dB               : {cnt_40db}")
    print(f"30 dB               : {cnt_30db}")
    print(f"20 dB               : {cnt_20db}")
    print(f"10 dB               : {cnt_10db}\n")

    print(f"Dataset generation  : {status_str}")
    print("========================================\n")


if __name__ == "__main__":
    generate_sinusoidal_dataset()
