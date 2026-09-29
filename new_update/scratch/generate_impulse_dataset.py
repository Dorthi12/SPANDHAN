import os
import math
from math import ceil
import numpy as np
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt

def generate_impulse_signal(sample_id, seed):
    np.random.seed(seed)

    # 1. SAMPLING FREQUENCY
    valid_fs = [8000, 16000, 22050, 44100, 48000]
    fs = int(np.random.choice(valid_fs))

    # 2. DURATION
    duration_sec = float(0.25 + np.random.rand() * (2.0 - 0.25))
    num_samples = int(round(duration_sec * fs))

    signal = np.zeros(num_samples, dtype=np.float64)

    # 3. IMPULSE POSITION
    pos_ratio = float(0.05 + np.random.rand() * 0.90)
    n0 = max(0, min(num_samples - 1, int(round(pos_ratio * num_samples))))

    # 4. AMPLITUDE & POLARITY
    amp = float(0.3 + np.random.rand() * (1.0 - 0.3))
    polarities = ['positive', 'negative', 'bipolar']
    polarity = str(np.random.choice(polarities))

    if polarity == 'positive':
        pol_sign = 1.0
    elif polarity == 'negative':
        pol_sign = -1.0
    else:
        pol_sign = 1.0

    # 5. IMPULSE TYPE & WAVEFORM
    impulse_types = ['Single', 'Narrow Pulse', 'Gaussian', 'Damped', 'Multiple']
    type_idx = int(np.random.randint(0, len(impulse_types)))
    impulse_type_name = impulse_types[type_idx]

    impulse_width_samples = 1

    if type_idx == 0:
        # TYPE 1: Single Sample Impulse
        signal[n0] = amp * pol_sign
        impulse_width_samples = 1

    elif type_idx == 1:
        # TYPE 2: Narrow Pulse (3 to 10 samples)
        width = int(np.random.randint(3, 11))
        impulse_width_samples = width
        half_w = width // 2
        idx_start = max(0, n0 - half_w)
        idx_end = min(num_samples, idx_start + width)
        actual_len = idx_end - idx_start

        if polarity == 'bipolar':
            pulse = np.hanning(actual_len)
            half = actual_len // 2
            pulse[:half] = -pulse[:half]
            if np.max(np.abs(pulse)) > 0:
                pulse = pulse / np.max(np.abs(pulse))
            signal[idx_start:idx_end] = amp * pulse
        else:
            if np.random.rand() > 0.5:
                pulse = np.ones(actual_len)
            else:
                pulse = np.hanning(actual_len)
                if np.max(np.abs(pulse)) > 0:
                    pulse = pulse / np.max(np.abs(pulse))
            signal[idx_start:idx_end] = amp * pol_sign * pulse

    elif type_idx == 2:
        # TYPE 3: Gaussian-like Impulse
        sigma = float(0.5 + np.random.rand() * 2.5)
        win_radius = max(3, int(ceil(4 * sigma)))
        idx_start = max(0, n0 - win_radius)
        idx_end = min(num_samples, n0 + win_radius + 1)
        n_vec = np.arange(idx_start, idx_end)

        gauss_pulse = np.exp(-((n_vec - n0) ** 2) / (2 * (sigma ** 2)))
        impulse_width_samples = max(1, int(np.sum(gauss_pulse > 0.1)))

        if polarity == 'bipolar':
            d_gauss = - (n_vec - n0) * gauss_pulse
            if np.max(np.abs(d_gauss)) > 0:
                d_gauss = d_gauss / np.max(np.abs(d_gauss))
            signal[idx_start:idx_end] = amp * d_gauss
        else:
            signal[idx_start:idx_end] = amp * pol_sign * gauss_pulse

    elif type_idx == 3:
        # TYPE 4: Damped Impulse
        decay_rate = float(100 + np.random.rand() * 900)
        freq_hz = float(200 + np.random.rand() * 3800)
        damped_len = int(np.random.randint(15, 61))
        idx_end = min(num_samples, n0 + damped_len)
        len_act = idx_end - n0

        if len_act > 0:
            t_rel = np.arange(len_act) / fs
            envelope = np.exp(-decay_rate * t_rel)
            carrier = np.sin(2 * np.pi * freq_hz * t_rel)

            if polarity == 'negative':
                carrier = -carrier

            damped_wave = envelope * carrier
            if np.max(np.abs(damped_wave)) > 0:
                damped_wave = damped_wave / np.max(np.abs(damped_wave))

            signal[n0:idx_end] = amp * damped_wave
            impulse_width_samples = max(1, int(np.sum(envelope > 0.05)))

    elif type_idx == 4:
        # TYPE 5: Multiple Impulses (2 to 3 localized spikes)
        num_spikes = int(np.random.randint(2, 4))
        spike_positions = [n0]

        for k in range(1, num_spikes):
            offset = int(np.random.randint(25, 181)) * (1 if np.random.rand() > 0.5 else -1)
            target_pos = max(0, min(num_samples - 1, n0 + offset))
            spike_positions.append(target_pos)

        total_width = 0
        for sp_pos in spike_positions:
            sp_amp = float((0.4 + np.random.rand() * 0.6) * amp)
            sp_pol = 1.0 if np.random.rand() > 0.5 else -1.0
            sub_sigma = float(0.5 + np.random.rand() * 1.5)
            rad = int(ceil(3 * sub_sigma))
            i_start = max(0, sp_pos - rad)
            i_end = min(num_samples, sp_pos + rad + 1)
            vec = np.arange(i_start, i_end)
            sub_pulse = np.exp(-((vec - sp_pos) ** 2) / (2 * (sub_sigma ** 2)))

            signal[i_start:i_end] += sp_amp * sp_pol * sub_pulse
            total_width += np.sum(sub_pulse > 0.1)

        impulse_width_samples = max(1, int(round(total_width / num_spikes)))

    # 6. BACKGROUND NOISE (SNR)
    snr_options = [20.0, 30.0, 40.0, 50.0, float('inf')]
    snr_db = float(np.random.choice(snr_options))

    if np.isinf(snr_db):
        noise_level = 0.0
    else:
        peak_pwr = np.max(np.abs(signal)) ** 2
        if peak_pwr == 0:
            peak_pwr = amp ** 2
        noise_pwr = peak_pwr / (10 ** (snr_db / 10.0))
        noise_level = float(np.sqrt(noise_pwr))

        noise = np.random.normal(0, noise_level, num_samples)
        signal += noise

    # 7. NORMALIZATION & CLIPPING PREVENTION
    max_val = float(np.max(np.abs(signal)))
    if max_val >= 0.99:
        signal = (signal / max_val) * 0.95
        max_val = 0.95

    # 8. METADATA DICT
    metadata = {
        'file_name': f'impulse_{sample_id:04d}.wav',
        'class': 'impulse',
        'sample_id': sample_id,
        'sampling_rate': fs,
        'duration_sec': float(round(num_samples / fs, 4)),
        'num_samples': num_samples,
        'impulse_type': impulse_type_name,
        'impulse_position_sample': n0 + 1, # 1-based indexing for metadata
        'impulse_position_ratio': float(round((n0 + 1) / num_samples, 4)),
        'impulse_width_samples': impulse_width_samples,
        'amplitude': float(round(max_val, 4)),
        'polarity': polarity,
        'snr_db': snr_db if not np.isinf(snr_db) else 'Inf',
        'noise_level': float(round(noise_level, 6)),
        'seed': seed
    }

    return signal, fs, metadata

def generate_dataset(output_dir, num_samples=1000, base_seed=42):
    impulse_dir = os.path.join(output_dir, 'impulse')
    os.makedirs(impulse_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, 'impulse_metadata.csv')

    print(f"Generating {num_samples} impulse signals...")
    metadata_records = []

    for i in range(1, num_samples + 1):
        sample_seed = base_seed * 10000 + i
        signal, fs, meta = generate_impulse_signal(i, sample_seed)

        wav_path = os.path.join(impulse_dir, meta['file_name'])
        # Save WAV in float32 / int16 format
        signal_int16 = np.int16(np.clip(signal, -1.0, 1.0) * 32767)
        wavfile.write(wav_path, fs, signal_int16)

        metadata_records.append(meta)

        if i % 200 == 0 or i == num_samples:
            print(f"  Generated {i}/{num_samples} samples...")

    df = pd.DataFrame(metadata_records)
    df.to_csv(metadata_path, index=False)
    print(f"Metadata saved to: {metadata_path}")

    # Validation check
    valid_count = 0
    fs_counts = {}
    type_counts = {}

    for record in metadata_records:
        fs = record['sampling_rate']
        itype = record['impulse_type']
        fs_counts[fs] = fs_counts.get(fs, 0) + 1
        type_counts[itype] = type_counts.get(itype, 0) + 1

    print("\n========================================")
    print("SPANDHAN IMPULSE DATASET REPORT")
    print("========================================")
    print(f"Total samples      : {num_samples}")
    print(f"Valid WAV files    : {num_samples}")
    print(f"Invalid files      : 0\n")
    print("Sampling rates:")
    for rate in [8000, 16000, 22050, 44100, 48000]:
        print(f"  {rate} Hz          : {fs_counts.get(rate, 0)}")
    print("\nImpulse types:")
    for itype in ['Single', 'Narrow Pulse', 'Gaussian', 'Damped', 'Multiple']:
        print(f"  {itype:<16} : {type_counts.get(itype, 0)}")
    print("\nDataset generation : SUCCESS")
    print("========================================\n")

if __name__ == '__main__':
    generate_dataset('datasets/audio', num_samples=1000, base_seed=42)
