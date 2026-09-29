"""
SPANDHAN — DSP Controller & Canonical Result Engine
====================================================
Coordinates DSP result ingestion, canonical verification fixtures,
and synchronization with the MATLAB DSP pipeline contracts.

Provides:
- Canonical sample datasets for all 10 verification test cases:
    5 Audio: Chirp, Sinusoidal, White Noise, Step, Impulse
    5 Image: Impulse, Sinusoidal, White Noise, Step, Chirp
- Reading and loading MATLAB .mat files
- Safe state mutation through DSPStateManager
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from ui.dsp.dsp_state import DSPStateManager, get_dsp_state
from ui.dsp.dsp_result_adapter import (
    AudioDSPResult,
    ImageDSPResult,
    AudioFFTData,
    AudioSTFTData,
    AudioWaveletData,
    AudioFIRData,
    AudioIIRData,
    AudioConvolutionData,
    AudioDeconvolutionData,
    ImageFFT2Data,
    ImageWavelet2DData,
    ImageFilterData,
    ImageConvolutionData,
    ImageDeconvolutionData,
)


class DSPController:
    """
    Controller managing DSP interactions, sample generation matching MATLAB contracts,
    and file import.
    """

    def __init__(self, state: Optional[DSPStateManager] = None):
        self.state = state or get_dsp_state()

    # -----------------------------------------------------------------------
    # Canonical Sample Generators (Exact MATLAB Contract Matches)
    # -----------------------------------------------------------------------

    def load_canonical_audio_sample(self, signal_class: str, switch_modality: bool = False) -> AudioDSPResult:
        """
        Builds a canonical AudioDSPResult reflecting the exact MATLAB analyzeAudio() output.
        """
        c = signal_class.lower().strip().replace(" ", "_")
        fs = 16000.0
        n_samples = 32000
        duration = n_samples / fs
        time = np.arange(n_samples) / fs

        if c == "chirp":
            # Linear frequency sweep from 500 Hz to 4500 Hz
            f0, f1 = 500.0, 4500.0
            freq_t = f0 + (f1 - f0) * (time / duration)
            phase = 2.0 * np.pi * (f0 * time + 0.5 * (f1 - f0) * (time ** 2) / duration)
            sig = np.sin(phase)

            # 1. FFT
            nfft = 4096
            n_pos = nfft // 2 + 1
            f_axis = np.linspace(0, fs / 2, n_pos)
            mag = np.exp(-((f_axis - 2500) ** 2) / (2 * (1000 ** 2))) * 0.45 + 0.05
            mag_db = 20.0 * np.log10(np.maximum(mag, 1e-6))

            fft_data = AudioFFTData(
                frequency=f_axis,
                magnitude=mag,
                magnitude_db=mag_db,
                nfft=nfft,
                sampling_frequency=fs,
                peak_frequency=2500.0,
                peak_magnitude=0.50,
                spectral_centroid=2500.0,
                spectral_bandwidth=1150.0,
                bandwidth_3db=1800.0,
                spectral_flatness=0.15,
            )

            # 2. STFT Spectrogram
            n_frames = 62
            n_stft_bins = 1025
            t_stft = np.linspace(0, duration, n_frames)
            f_stft = np.linspace(0, fs / 2, n_stft_bins)

            s_db = np.full((n_stft_bins, n_frames), -75.0)
            dom_freqs = np.zeros(n_frames)

            for col, tc in enumerate(t_stft):
                f_inst = f0 + (f1 - f0) * (tc / duration)
                dom_freqs[col] = f_inst
                bin_idx = int(np.clip(f_inst / (fs / 2) * n_stft_bins, 0, n_stft_bins - 1))
                low_b = max(0, bin_idx - 15)
                high_b = min(n_stft_bins, bin_idx + 15)
                s_db[low_b:high_b, col] = np.linspace(-40, 0, high_b - low_b)

            stft_data = AudioSTFTData(
                frequency=f_stft,
                time=t_stft,
                magnitude_db=s_db,
                dominant_frequency=dom_freqs,
                total_energy=8.45e4,
                time_resolution=0.064,
                frequency_resolution=7.81,
                window_length=1024,
                overlap=512,
                hop_size=512,
                nfft=2048,
                sampling_frequency=fs,
            )

            # 3. Wavelet
            wavelet_data = AudioWaveletData(
                wavelet_name="db4",
                decomposition_level=5,
                approximation=np.sin(time[:1000] * 20),
                details=[np.random.normal(0, 0.05, 1000) for _ in range(5)],
                detail_energy=np.array([120.5, 340.2, 510.8, 280.4, 90.1]),
                dominant_level=3,
                wavelet_entropy=1.52,
                reconstruction_rmse=1.2e-6,
            )

            res = AudioDSPResult(
                signal_class="Chirp",
                status="Completed",
                sampling_frequency=fs,
                signal_length=n_samples,
                duration=duration,
                completed_analyses=["FFT", "STFT", "Wavelet"],
                failed_analyses=[],
                execution_log=[
                    "FFT completed.",
                    "STFT completed.",
                    "Wavelet analysis completed.",
                    "Convolution skipped: no impulse response supplied.",
                    "Deconvolution skipped: no impulse response supplied.",
                ],
                input_file="datasets/audio/chirp/chirp_0001.wav",
                raw_signal=sig,
                analysis_signal=sig,
                confidence=0.9968,
                probabilities={"chirp": 0.9968, "sinusoidal": 0.0018, "white_noise": 0.0006, "step": 0.0005, "impulse": 0.0003},
                fft=fft_data,
                stft=stft_data,
                wavelet=wavelet_data,
            )

        elif c == "sinusoidal":
            # Pure sinusoidal 1000 Hz + subtle 3000 Hz harmonic
            f_main = 1000.0
            sig = np.sin(2.0 * np.pi * f_main * time) + 0.15 * np.sin(2.0 * np.pi * 3000.0 * time)

            nfft = 4096
            n_pos = nfft // 2 + 1
            f_axis = np.linspace(0, fs / 2, n_pos)
            mag = np.exp(-((f_axis - f_main) ** 2) / (2 * (15 ** 2))) * 1.0 + 0.15 * np.exp(-((f_axis - 3000) ** 2) / (2 * (25 ** 2)))
            mag_db = 20.0 * np.log10(np.maximum(mag, 1e-6))

            fft_data = AudioFFTData(
                frequency=f_axis,
                magnitude=mag,
                magnitude_db=mag_db,
                peak_frequency=1000.0,
                peak_magnitude=1.002,
                spectral_centroid=1260.0,
                spectral_bandwidth=620.0,
                bandwidth_3db=32.0,
                spectral_flatness=0.0024,
            )

            # FIR
            f_fir = np.linspace(0, fs / 2, 512)
            fir_mag = np.where(f_fir <= 3000.0, 0.0, -20.0 * np.log10(1 + (f_fir / 3000.0) ** 8))
            fir_imp = np.sinc(2 * 3000.0 * (np.arange(101) - 50) / fs) * np.hamming(101)
            fir_data = AudioFIRData(
                order=100,
                filter_type="low",
                cutoff_frequency=3000.0,
                window="hamming",
                frequency=f_fir,
                magnitude_db=fir_mag,
                impulse_response=fir_imp,
                estimated_3db_cutoff=2985.0,
            )

            # IIR
            f_iir = np.linspace(0, fs / 2, 512)
            iir_mag = -10.0 * np.log10(1.0 + (f_iir / 3000.0) ** 12)
            iir_imp = np.exp(-np.linspace(0, 5, 100)) * np.cos(np.linspace(0, 20, 100))
            iir_data = AudioIIRData(
                order=6,
                filter_family="butter",
                passband_frequency=3000.0,
                stopband_frequency=4000.0,
                frequency=f_iir,
                magnitude_db=iir_mag,
                impulse_response=iir_imp,
                stable=True,
                peak_frequency=0.0,
                estimated_3db_cutoff=3000.0,
            )

            res = AudioDSPResult(
                signal_class="Sinusoidal",
                status="Completed",
                sampling_frequency=fs,
                signal_length=n_samples,
                duration=duration,
                completed_analyses=["FFT", "FIR", "IIR"],
                failed_analyses=[],
                execution_log=[
                    "FFT completed.",
                    "FIR filtering completed.",
                    "IIR filtering completed.",
                ],
                input_file="datasets/audio/sinusoidal/sinusoidal_0001.wav",
                raw_signal=sig,
                analysis_signal=sig,
                confidence=0.9984,
                probabilities={"sinusoidal": 0.9984, "chirp": 0.0010, "step": 0.0003, "white_noise": 0.0002, "impulse": 0.0001},
                fft=fft_data,
                fir=fir_data,
                iir=iir_data,
            )

        elif c == "white_noise":
            np.random.seed(42)
            sig = np.random.normal(0, 0.35, n_samples)
            sig = np.clip(sig, -1.0, 1.0)

            nfft = 4096
            n_pos = nfft // 2 + 1
            f_axis = np.linspace(0, fs / 2, n_pos)
            mag = np.random.normal(0.015, 0.003, n_pos)
            mag_db = 20.0 * np.log10(np.maximum(mag, 1e-6))

            fft_data = AudioFFTData(
                frequency=f_axis,
                magnitude=mag,
                magnitude_db=mag_db,
                peak_frequency=3840.0,
                peak_magnitude=0.024,
                spectral_centroid=3980.0,
                spectral_bandwidth=2300.0,
                bandwidth_3db=7800.0,
                spectral_flatness=0.884,
            )

            f_fir = np.linspace(0, fs / 2, 512)
            fir_mag = np.where(f_fir <= 3000.0, 0.0, -45.0 * (f_fir - 3000.0) / 5000.0)
            fir_imp = np.sinc(2 * 3000.0 * (np.arange(101) - 50) / fs) * np.hamming(101)
            fir_data = AudioFIRData(
                order=100,
                filter_type="low",
                cutoff_frequency=3000.0,
                window="hamming",
                frequency=f_fir,
                magnitude_db=fir_mag,
                impulse_response=fir_imp,
            )

            f_iir = np.linspace(0, fs / 2, 512)
            iir_mag = -10.0 * np.log10(1.0 + (f_iir / 3000.0) ** 12)
            iir_imp = np.exp(-np.linspace(0, 5, 100)) * np.cos(np.linspace(0, 20, 100))
            iir_data = AudioIIRData(
                order=6,
                filter_family="butter",
                passband_frequency=3000.0,
                stopband_frequency=4000.0,
                frequency=f_iir,
                magnitude_db=iir_mag,
                impulse_response=iir_imp,
                stable=True,
            )

            res = AudioDSPResult(
                signal_class="White Noise",
                status="Completed",
                sampling_frequency=fs,
                signal_length=n_samples,
                duration=duration,
                completed_analyses=["FFT", "FIR", "IIR"],
                failed_analyses=[],
                execution_log=[
                    "FFT completed.",
                    "FIR filtering completed.",
                    "IIR filtering completed.",
                ],
                input_file="datasets/audio/white_noise/white_noise_0001.wav",
                raw_signal=sig,
                analysis_signal=sig,
                confidence=0.9942,
                probabilities={"white_noise": 0.9942, "step": 0.0028, "impulse": 0.0018, "chirp": 0.0008, "sinusoidal": 0.0004},
                fft=fft_data,
                fir=fir_data,
                iir=iir_data,
            )

        elif c == "step":
            # Step transition at sample 16000
            sig = np.zeros(n_samples)
            sig[16000:] = 1.0

            nfft = 4096
            n_pos = nfft // 2 + 1
            f_axis = np.linspace(0, fs / 2, n_pos)
            mag = 1.0 / (1.0 + f_axis / 20.0) * 0.8
            mag_db = 20.0 * np.log10(np.maximum(mag, 1e-6))

            fft_data = AudioFFTData(
                frequency=f_axis,
                magnitude=mag,
                magnitude_db=mag_db,
                peak_frequency=0.0,
                peak_magnitude=0.8,
                spectral_centroid=420.0,
                spectral_bandwidth=850.0,
                bandwidth_3db=120.0,
                spectral_flatness=0.045,
            )

            # Wavelet
            wavelet_data = AudioWaveletData(
                wavelet_name="db4",
                decomposition_level=5,
                approximation=np.ones(1000) * 0.5,
                details=[np.zeros(1000) for _ in range(5)],
                detail_energy=np.array([45.2, 85.1, 190.4, 320.6, 610.8]),
                dominant_level=5,
                wavelet_entropy=1.28,
            )

            f_fir = np.linspace(0, fs / 2, 512)
            fir_mag = -20.0 * np.log10(1 + (f_fir / 3000.0) ** 6)
            fir_imp = np.sinc(2 * 3000.0 * (np.arange(101) - 50) / fs) * np.hamming(101)
            fir_data = AudioFIRData(
                order=100,
                filter_type="low",
                cutoff_frequency=3000.0,
                frequency=f_fir,
                magnitude_db=fir_mag,
                impulse_response=fir_imp,
            )

            f_iir = np.linspace(0, fs / 2, 512)
            iir_mag = -10.0 * np.log10(1.0 + (f_iir / 3000.0) ** 12)
            iir_imp = np.exp(-np.linspace(0, 5, 100)) * np.cos(np.linspace(0, 20, 100))
            iir_data = AudioIIRData(
                order=6,
                filter_family="butter",
                passband_frequency=3000.0,
                stopband_frequency=4000.0,
                frequency=f_iir,
                magnitude_db=iir_mag,
                impulse_response=iir_imp,
                stable=True,
            )

            res = AudioDSPResult(
                signal_class="Step",
                status="Completed",
                sampling_frequency=fs,
                signal_length=n_samples,
                duration=duration,
                completed_analyses=["FFT", "Wavelet", "FIR", "IIR"],
                failed_analyses=[],
                execution_log=[
                    "FFT completed.",
                    "Wavelet analysis completed.",
                    "FIR filtering completed.",
                    "IIR filtering completed.",
                    "Convolution skipped: no impulse response supplied.",
                ],
                input_file="datasets/audio/step/step_0001.wav",
                raw_signal=sig,
                analysis_signal=sig,
                confidence=0.9972,
                probabilities={"step": 0.9972, "impulse": 0.0015, "white_noise": 0.0008, "sinusoidal": 0.0003, "chirp": 0.0002},
                fft=fft_data,
                wavelet=wavelet_data,
                fir=fir_data,
                iir=iir_data,
            )

        else:  # impulse
            sig = np.zeros(n_samples)
            sig[16000] = 1.0

            nfft = 4096
            n_pos = nfft // 2 + 1
            f_axis = np.linspace(0, fs / 2, n_pos)
            mag = np.full(n_pos, 0.0025)
            mag_db = 20.0 * np.log10(mag)

            fft_data = AudioFFTData(
                frequency=f_axis,
                magnitude=mag,
                magnitude_db=mag_db,
                peak_frequency=0.0,
                peak_magnitude=0.0025,
                spectral_centroid=4000.0,
                spectral_bandwidth=2309.0,
                bandwidth_3db=8000.0,
                spectral_flatness=0.998,
            )

            wavelet_data = AudioWaveletData(
                wavelet_name="db4",
                decomposition_level=5,
                approximation=np.zeros(1000),
                details=[np.zeros(1000) for _ in range(5)],
                detail_energy=np.array([890.2, 420.5, 180.1, 75.4, 25.1]),
                dominant_level=1,
                wavelet_entropy=0.94,
            )

            f_fir = np.linspace(0, fs / 2, 512)
            fir_mag = -20.0 * np.log10(1 + (f_fir / 3000.0) ** 6)
            fir_imp = np.sinc(2 * 3000.0 * (np.arange(101) - 50) / fs) * np.hamming(101)
            fir_data = AudioFIRData(
                order=100,
                filter_type="low",
                cutoff_frequency=3000.0,
                frequency=f_fir,
                magnitude_db=fir_mag,
                impulse_response=fir_imp,
            )

            res = AudioDSPResult(
                signal_class="Impulse",
                status="Completed",
                sampling_frequency=fs,
                signal_length=n_samples,
                duration=duration,
                completed_analyses=["FFT", "Wavelet", "FIR"],
                failed_analyses=[],
                execution_log=[
                    "FFT completed.",
                    "Wavelet analysis completed.",
                    "FIR filtering completed.",
                    "Convolution skipped: no impulse response supplied.",
                    "Deconvolution skipped: no impulse response supplied.",
                ],
                input_file="datasets/audio/impulse/impulse_0001.wav",
                raw_signal=sig,
                analysis_signal=sig,
                confidence=0.9991,
                probabilities={"impulse": 0.9991, "step": 0.0005, "white_noise": 0.0002, "chirp": 0.0001, "sinusoidal": 0.0001},
                fft=fft_data,
                wavelet=wavelet_data,
                fir=fir_data,
            )

        self.state.set_audio_result(res)
        if switch_modality:
            self.state.set_active_modality("audio")
        return res

    def load_canonical_image_sample(self, signal_class: str, switch_modality: bool = False) -> ImageDSPResult:
        """
        Builds a canonical ImageDSPResult reflecting the exact MATLAB analyzeImage() output.
        """
        c = signal_class.lower().strip().replace(" ", "_")
        sz = 128
        y, x = np.mgrid[0:sz, 0:sz]

        if c == "sinusoidal":
            # 2D sinusoidal grating with frequency 8 cycles across image
            img = 0.5 + 0.45 * np.cos(2.0 * np.pi * 8.0 * x / sz)

            # 2D FFT: prominent peaks at ±(8/128, 0)
            fft2_mag = np.full((sz, sz), -70.0)
            cx, cy = sz // 2, sz // 2
            fft2_mag[cy, cx - 8] = 0.0
            fft2_mag[cy, cx + 8] = 0.0
            fft2_mag[cy - 2:cy + 3, cx - 10:cx - 6] = -25.0
            fft2_mag[cy - 2:cy + 3, cx + 6:cx + 10] = -25.0

            fft2_data = ImageFFT2Data(
                gray_image=img,
                magnitude_db=fft2_mag,
                dominant_frequency_x=8.0 / sz,
                dominant_frequency_y=0.0,
                dominant_spatial_frequency=8.0 / sz,
                spectral_centroid_x=8.0 / sz,
                spectral_centroid_y=0.0,
                spectral_bandwidth=0.015,
                total_energy=4.2e4,
            )

            # Filter
            flt_img = 0.5 + 0.38 * np.cos(2.0 * np.pi * 8.0 * x / sz)
            flt_data = ImageFilterData(
                input_image=img,
                filtered_image=flt_img,
                difference_image=flt_img - img,
                filter_type="gaussian",
                kernel_size=(5, 5),
                sigma=1.0,
                input_mean=float(np.mean(img)),
                output_mean=float(np.mean(flt_img)),
                input_std=float(np.std(img)),
                output_std=float(np.std(flt_img)),
                difference_energy=0.82,
                computation_time=0.0034,
            )

            res = ImageDSPResult(
                signal_class="Sinusoidal",
                status="Completed",
                image_size=(sz, sz),
                completed_analyses=["2-D FFT", "Image Filter"],
                failed_analyses=[],
                execution_log=[
                    "2-D FFT completed.",
                    "Image filtering completed.",
                ],
                input_file="datasets/image/sinusoidal/sinusoidal_0001.png",
                raw_image=img,
                analysis_image=img,
                confidence=0.9976,
                probabilities={"sinusoidal": 0.9976, "chirp": 0.0014, "step": 0.0005, "white_noise": 0.0003, "impulse": 0.0002},
                fft2=fft2_data,
                filter=flt_data,
            )

        elif c == "chirp":
            # 2D zone-plate / circular chirp
            r2 = (x - sz / 2) ** 2 + (y - sz / 2) ** 2
            img = 0.5 + 0.45 * np.cos(2.0 * np.pi * 0.0035 * r2)

            fft2_mag = np.full((sz, sz), -55.0)
            cx, cy = sz // 2, sz // 2
            # Ring pattern in FFT
            r_fft = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            fft2_mag = np.where((r_fft >= 15) & (r_fft <= 35), -15.0 + 10 * np.sin(r_fft), -65.0)

            fft2_data = ImageFFT2Data(
                gray_image=img,
                magnitude_db=fft2_mag,
                dominant_spatial_frequency=0.185,
                spectral_centroid_x=0.0,
                spectral_centroid_y=0.0,
                spectral_bandwidth=0.082,
                total_energy=8.9e4,
            )

            wavelet2d_data = ImageWavelet2DData(
                wavelet_name="db4",
                decomposition_level=3,
                approximation=img[::4, ::4],
                horizontal_details=[np.zeros((16, 16)) for _ in range(3)],
                vertical_details=[np.zeros((16, 16)) for _ in range(3)],
                diagonal_details=[np.zeros((16, 16)) for _ in range(3)],
                wavelet_entropy=1.64,
                reconstruction_rmse=2.1e-6,
            )

            flt_img = 0.5 + 0.32 * np.cos(2.0 * np.pi * 0.0035 * r2)
            flt_data = ImageFilterData(
                input_image=img,
                filtered_image=flt_img,
                difference_image=flt_img - img,
                filter_type="gaussian",
                kernel_size=(5, 5),
                sigma=1.0,
                input_mean=float(np.mean(img)),
                output_mean=float(np.mean(flt_img)),
                input_std=float(np.std(img)),
                output_std=float(np.std(flt_img)),
                difference_energy=1.45,
                computation_time=0.0038,
            )

            res = ImageDSPResult(
                signal_class="Chirp",
                status="Completed",
                image_size=(sz, sz),
                completed_analyses=["2-D FFT", "2-D Wavelet", "Image Filter"],
                failed_analyses=[],
                execution_log=[
                    "2-D FFT completed.",
                    "2-D wavelet analysis completed.",
                    "Image filtering completed.",
                ],
                input_file="datasets/image/chirp/chirp_0001.png",
                raw_image=img,
                analysis_image=img,
                confidence=0.9982,
                probabilities={"chirp": 0.9982, "sinusoidal": 0.0011, "step": 0.0004, "white_noise": 0.0002, "impulse": 0.0001},
                fft2=fft2_data,
                wavelet2d=wavelet2d_data,
                filter=flt_data,
            )

        elif c == "step":
            # Sharp step edge along middle column
            img = np.where(x >= sz // 2, 0.9, 0.1)

            fft2_mag = np.full((sz, sz), -70.0)
            cx, cy = sz // 2, sz // 2
            # Sinc decay along X axis
            for u in range(sz):
                dist = abs(u - cx) + 1
                fft2_mag[cy, u] = max(-60.0, -10.0 - 15.0 * np.log10(dist))

            fft2_data = ImageFFT2Data(
                gray_image=img,
                magnitude_db=fft2_mag,
                dominant_spatial_frequency=0.045,
                spectral_centroid_x=0.05,
                spectral_centroid_y=0.0,
                total_energy=3.8e4,
            )

            wavelet2d_data = ImageWavelet2DData(
                wavelet_name="db4",
                decomposition_level=3,
                approximation=img[::4, ::4],
                horizontal_details=[np.zeros((16, 16)) for _ in range(3)],
                vertical_details=[np.zeros((16, 16)) for _ in range(3)],
                diagonal_details=[np.zeros((16, 16)) for _ in range(3)],
                wavelet_entropy=1.12,
            )

            flt_img = np.copy(img)
            flt_img[:, sz // 2 - 2:sz // 2 + 2] = 0.5
            flt_data = ImageFilterData(
                input_image=img,
                filtered_image=flt_img,
                difference_image=flt_img - img,
                filter_type="gaussian",
                kernel_size=(5, 5),
                sigma=1.0,
                input_mean=float(np.mean(img)),
                output_mean=float(np.mean(flt_img)),
                input_std=float(np.std(img)),
                output_std=float(np.std(flt_img)),
                difference_energy=0.95,
                computation_time=0.0035,
            )

            res = ImageDSPResult(
                signal_class="Step",
                status="Completed",
                image_size=(sz, sz),
                completed_analyses=["2-D FFT", "2-D Wavelet", "Image Filter"],
                failed_analyses=[],
                execution_log=[
                    "2-D FFT completed.",
                    "2-D wavelet analysis completed.",
                    "Image filtering completed.",
                ],
                input_file="datasets/image/step/step_0001.png",
                raw_image=img,
                analysis_image=img,
                confidence=0.9958,
                probabilities={"step": 0.9958, "impulse": 0.0021, "white_noise": 0.0011, "sinusoidal": 0.0006, "chirp": 0.0004},
                fft2=fft2_data,
                wavelet2d=wavelet2d_data,
                filter=flt_data,
            )

        elif c == "white_noise":
            np.random.seed(101)
            img = np.random.uniform(0.1, 0.9, (sz, sz))

            fft2_mag = np.random.normal(-35.0, 4.0, (sz, sz))
            fft2_data = ImageFFT2Data(
                gray_image=img,
                magnitude_db=fft2_mag,
                dominant_spatial_frequency=0.25,
                spectral_bandwidth=0.21,
                total_energy=5.5e4,
            )

            flt_img = np.full((sz, sz), 0.5) + np.random.normal(0, 0.08, (sz, sz))
            flt_data = ImageFilterData(
                input_image=img,
                filtered_image=flt_img,
                difference_image=flt_img - img,
                filter_type="gaussian",
                kernel_size=(5, 5),
                sigma=1.0,
                input_mean=float(np.mean(img)),
                output_mean=float(np.mean(flt_img)),
                input_std=float(np.std(img)),
                output_std=float(np.std(flt_img)),
                difference_energy=2.85,
                computation_time=0.0032,
            )

            res = ImageDSPResult(
                signal_class="White Noise",
                status="Completed",
                image_size=(sz, sz),
                completed_analyses=["2-D FFT", "Image Filter"],
                failed_analyses=[],
                execution_log=[
                    "2-D FFT completed.",
                    "Image filtering completed.",
                ],
                input_file="datasets/image/white_noise/white_noise_0001.png",
                raw_image=img,
                analysis_image=img,
                confidence=0.9934,
                probabilities={"white_noise": 0.9934, "step": 0.0031, "impulse": 0.0018, "sinusoidal": 0.0010, "chirp": 0.0007},
                fft2=fft2_data,
                filter=flt_data,
            )

        else:  # impulse
            img = np.zeros((sz, sz))
            img[sz // 2, sz // 2] = 1.0

            fft2_mag = np.full((sz, sz), -20.0)
            fft2_data = ImageFFT2Data(
                gray_image=img,
                magnitude_db=fft2_mag,
                dominant_spatial_frequency=0.0,
                total_energy=1.0,
            )

            wavelet2d_data = ImageWavelet2DData(
                wavelet_name="db4",
                decomposition_level=3,
                approximation=img[::4, ::4],
                horizontal_details=[np.zeros((16, 16)) for _ in range(3)],
                vertical_details=[np.zeros((16, 16)) for _ in range(3)],
                diagonal_details=[np.zeros((16, 16)) for _ in range(3)],
                wavelet_entropy=0.85,
            )

            flt_img = np.zeros((sz, sz))
            flt_img[sz // 2 - 1:sz // 2 + 2, sz // 2 - 1:sz // 2 + 2] = 0.11
            flt_data = ImageFilterData(
                input_image=img,
                filtered_image=flt_img,
                difference_image=flt_img - img,
                filter_type="gaussian",
                kernel_size=(5, 5),
                sigma=1.0,
                input_mean=float(np.mean(img)),
                output_mean=float(np.mean(flt_img)),
                difference_energy=0.78,
                computation_time=0.0031,
            )

            res = ImageDSPResult(
                signal_class="Impulse",
                status="Completed",
                image_size=(sz, sz),
                completed_analyses=["2-D FFT", "2-D Wavelet", "Image Filter"],
                failed_analyses=[],
                execution_log=[
                    "2-D FFT completed.",
                    "2-D wavelet analysis completed.",
                    "Image filtering completed.",
                ],
                input_file="datasets/image/impulse/impulse_0001.png",
                raw_image=img,
                analysis_image=img,
                confidence=0.9995,
                probabilities={"impulse": 0.9995, "step": 0.0003, "white_noise": 0.0001, "sinusoidal": 0.0001, "chirp": 0.0000},
                fft2=fft2_data,
                wavelet2d=wavelet2d_data,
                filter=flt_data,
            )

        self.state.set_image_result(res)
        if switch_modality:
            self.state.set_active_modality("image")
        return res
