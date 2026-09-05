"""
Test Dsp module.
"""

import numpy as np
import pytest

from dsp.fft import compute_fft
from dsp.psd import compute_psd
from dsp.stft import compute_stft
from dsp.correlation import (
    compute_cross_correlation,
    find_correlation_peak,
)
from dsp.hilbert import compute_hilbert
from dsp.filters import (
    apply_lowpass,
    apply_highpass,
    apply_bandpass,
    apply_bandstop,
)
def test_esprit_detects_two_tones():
    from dsp.esprit import estimate_esprit_frequencies

    fs = 1000

    t = np.arange(2000) / fs

    rng = np.random.default_rng(42)

    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.8 * np.sin(2 * np.pi * 120 * t)
        + 0.05 * rng.standard_normal(len(t))
    )

    result = estimate_esprit_frequencies(
        signal,
        sampling_rate=fs,
        model_order=30,
        num_sources=2,
    )

    frequencies = result["frequencies"]

    assert len(frequencies) == 2

    assert np.any(
        np.abs(frequencies - 50) < 2.0
    )

    assert np.any(
        np.abs(frequencies - 120) < 2.0
    )


def test_esprit_detects_close_frequencies():
    from dsp.esprit import estimate_esprit_frequencies

    fs = 1000

    t = np.arange(5000) / fs

    rng = np.random.default_rng(42)

    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.8 * np.sin(2 * np.pi * 51 * t)
        + 0.005 * rng.standard_normal(len(t))
    )

    result = estimate_esprit_frequencies(
        signal,
        sampling_rate=fs,
        model_order=80,
        num_sources=2,
    )

    frequencies = result["frequencies"]

    assert len(frequencies) == 2

    assert np.any(
        np.abs(frequencies - 50) < 1.0
    )

    assert np.any(
        np.abs(frequencies - 51) < 1.0
    )


def test_esprit_returns_subspace():
    from dsp.esprit import estimate_esprit_frequencies

    fs = 1000

    t = np.arange(1000) / fs

    signal = (
        np.sin(2 * np.pi * 50 * t)
        + np.sin(2 * np.pi * 100 * t)
    )

    result = estimate_esprit_frequencies(
        signal,
        sampling_rate=fs,
        model_order=20,
        num_sources=2,
    )

    assert "signal_subspace" in result
    assert "eigenvalues" in result

    assert result["signal_subspace"].shape[0] == 20
    assert result["signal_subspace"].shape[1] == 4


def test_esprit_invalid_parameters():
    from dsp.esprit import estimate_esprit_frequencies

    signal = np.ones(1000)

    with pytest.raises(ValueError):
        estimate_esprit_frequencies(
            signal,
            sampling_rate=1000,
            model_order=10,
            num_sources=10,
        )

    with pytest.raises(ValueError):
        estimate_esprit_frequencies(
            signal,
            sampling_rate=0,
            model_order=10,
            num_sources=2,
        )
        
def test_real_cepstrum_returns_correct_length():
    from dsp.cepstrum import compute_real_cepstrum

    fs = 1000
    signal = np.sin(2 * np.pi * 50 * np.arange(1000) / fs)

    quefrency, cepstrum = compute_real_cepstrum(signal, fs)

    assert len(quefrency) == len(signal)
    assert len(cepstrum) == len(signal)


def test_real_cepstrum_quefrency_spacing():
    from dsp.cepstrum import compute_real_cepstrum

    fs = 1000
    signal = np.sin(2 * np.pi * 50 * np.arange(1000) / fs)

    quefrency, _ = compute_real_cepstrum(signal, fs)

    assert np.isclose(quefrency[0], 0.0)
    assert np.isclose(quefrency[1], 1 / fs)


def test_real_cepstrum_is_finite():
    from dsp.cepstrum import compute_real_cepstrum

    signal = np.ones(1000)

    quefrency, cepstrum = compute_real_cepstrum(
        signal,
        sampling_rate=1000,
    )

    assert np.all(np.isfinite(quefrency))
    assert np.all(np.isfinite(cepstrum))


def test_real_cepstrum_invalid_sampling_rate():
    from dsp.cepstrum import compute_real_cepstrum

    signal = np.ones(1000)

    with pytest.raises(ValueError):
        compute_real_cepstrum(signal, sampling_rate=0)

def test_music_returns_spectrum():
    from dsp.music import estimate_music_frequencies

    fs = 1000

    t = np.arange(1000) / fs

    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.8 * np.sin(2 * np.pi * 120 * t)
    )

    result = estimate_music_frequencies(
        signal,
        sampling_rate=fs,
        model_order=20,
        num_sources=2,
    )

    assert "frequencies" in result
    assert "frequency_axis" in result
    assert "pseudospectrum" in result

    assert len(result["frequency_axis"]) == 4096
    assert len(result["pseudospectrum"]) == 4096


def test_music_detects_two_tones():
    from dsp.music import estimate_music_frequencies

    fs = 1000

    t = np.arange(2000) / fs

    rng = np.random.default_rng(42)

    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.8 * np.sin(2 * np.pi * 120 * t)
        + 0.05 * rng.standard_normal(len(t))
    )

    result = estimate_music_frequencies(
        signal,
        sampling_rate=fs,
        model_order=30,
        num_sources=2,
    )

    frequencies = result["frequencies"]

    assert len(frequencies) == 2

    assert np.any(np.abs(frequencies - 50) < 2.0)
    assert np.any(np.abs(frequencies - 120) < 2.0)


def test_music_detects_close_frequencies():
    from dsp.music import estimate_music_frequencies

    fs = 1000

    # Longer observation gives MUSIC more information
    # for resolving closely spaced frequencies.
    t = np.arange(5000) / fs

    rng = np.random.default_rng(42)

    signal = (
        np.sin(2 * np.pi * 50 * t)
        + 0.8 * np.sin(2 * np.pi * 51 * t)
        + 0.005 * rng.standard_normal(len(t))
    )

    result = estimate_music_frequencies(
        signal,
        sampling_rate=fs,
        model_order=80,
        num_sources=2,
        nfft=16384,
    )

    frequencies = result["frequencies"]

    assert len(frequencies) == 2

    assert np.any(
        np.abs(frequencies - 50) < 1.0
    )

    assert np.any(
        np.abs(frequencies - 51) < 1.0
    )

def test_music_invalid_parameters():
    from dsp.music import estimate_music_frequencies

    signal = np.ones(1000)

    with pytest.raises(ValueError):
        estimate_music_frequencies(
            signal,
            sampling_rate=1000,
            model_order=10,
            num_sources=10,
        )

    with pytest.raises(ValueError):
        estimate_music_frequencies(
            signal,
            sampling_rate=1000,
            model_order=10,
            num_sources=2,
            nfft=1,
        )

            
def test_discrete_wavelet_transform_returns_coefficients():
    from dsp.wavelet import discrete_wavelet_transform

    signal = np.sin(2 * np.pi * 10 * np.arange(1000) / 1000)

    result = discrete_wavelet_transform(signal)

    assert "coefficients" in result
    assert "approximation" in result
    assert "details" in result
    assert isinstance(result["coefficients"], list)
    assert len(result["coefficients"]) > 1


def test_wavelet_approximation_and_details_are_nonempty():
    from dsp.wavelet import discrete_wavelet_transform

    signal = np.sin(2 * np.pi * 20 * np.arange(1000) / 1000)

    result = discrete_wavelet_transform(
        signal,
        wavelet="db4",
        level=4,
    )

    assert len(result["approximation"]) > 0
    assert len(result["details"]) == 4

    for detail in result["details"]:
        assert len(detail) > 0


def test_wavelet_invalid_wavelet_raises_error():
    from dsp.wavelet import discrete_wavelet_transform

    signal = np.ones(1000)

    with pytest.raises(ValueError):
        discrete_wavelet_transform(
            signal,
            wavelet="not_a_real_wavelet",
        )


def test_wavelet_invalid_level_raises_error():
    from dsp.wavelet import discrete_wavelet_transform

    signal = np.ones(1000)

    with pytest.raises(ValueError):
        discrete_wavelet_transform(
            signal,
            wavelet="db4",
            level=100,
        )


def test_lowpass_filter():

    fs = 1000

    t = np.arange(2000) / fs

    low = np.sin(
        2 * np.pi * 20 * t
    )

    high = 0.5 * np.sin(
        2 * np.pi * 300 * t
    )

    signal_data = low + high

    filtered = apply_lowpass(
        signal_data,
        sampling_rate=fs,
        cutoff_frequency=100,
    )

    correlation = np.corrcoef(
        filtered[100:-100],
        low[100:-100],
    )[0, 1]

    assert correlation > 0.9


def test_highpass_filter():

    fs = 1000

    t = np.arange(2000) / fs

    low = np.sin(
        2 * np.pi * 20 * t
    )

    high = np.sin(
        2 * np.pi * 300 * t
    )

    signal_data = low + high

    filtered = apply_highpass(
        signal_data,
        sampling_rate=fs,
        cutoff_frequency=100,
    )

    correlation = np.corrcoef(
        filtered[100:-100],
        high[100:-100],
    )[0, 1]

    assert correlation > 0.9


def test_bandpass_filter():

    fs = 1000

    t = np.arange(2000) / fs

    target = np.sin(
        2 * np.pi * 100 * t
    )

    outside = 0.5 * np.sin(
        2 * np.pi * 300 * t
    )

    signal_data = target + outside

    filtered = apply_bandpass(
        signal_data,
        sampling_rate=fs,
        low_frequency=50,
        high_frequency=150,
    )

    correlation = np.corrcoef(
        filtered[100:-100],
        target[100:-100],
    )[0, 1]

    assert correlation > 0.9


def test_bandstop_filter():

    fs = 1000

    t = np.arange(2000) / fs

    wanted = np.sin(
        2 * np.pi * 20 * t
    )

    interference = 0.5 * np.sin(
        2 * np.pi * 200 * t
    )

    signal_data = wanted + interference

    filtered = apply_bandstop(
        signal_data,
        sampling_rate=fs,
        low_frequency=180,
        high_frequency=220,
    )

    correlation = np.corrcoef(
        filtered[100:-100],
        wanted[100:-100],
    )[0, 1]

    assert correlation > 0.9


def test_filter_rejects_invalid_band():

    signal_data = np.ones(1000)

    try:
        apply_bandpass(
            signal_data,
            sampling_rate=1000,
            low_frequency=200,
            high_frequency=100,
        )
        assert False
    except ValueError:
        assert True

def test_hilbert_envelope():

    sampling_rate = 1000

    t = np.arange(2000) / sampling_rate

    amplitude = 2.0

    signal_data = amplitude * np.sin(
        2 * np.pi * 50 * t
    )

    result = compute_hilbert(
        signal_data,
        sampling_rate,
    )

    envelope = result["envelope"]

    assert envelope.shape == signal_data.shape
    assert np.isfinite(envelope).all()

    # Ignore edge effects from the Hilbert transform.
    assert np.isclose(
        np.mean(envelope[100:-100]),
        amplitude,
        atol=0.05,
    )


def test_hilbert_detects_frequency():

    sampling_rate = 1000
    frequency = 50

    t = np.arange(2000) / sampling_rate

    signal_data = np.sin(
        2 * np.pi * frequency * t
    )

    result = compute_hilbert(
        signal_data,
        sampling_rate,
    )

    instantaneous_frequency = result[
        "instantaneous_frequency"
    ]

    middle = instantaneous_frequency[
        100:-100
    ]

    assert np.isclose(
        np.median(middle),
        frequency,
        atol=1.0,
    )


def test_hilbert_output_shapes():

    signal_data = np.random.default_rng(
        42
    ).normal(size=1000)

    result = compute_hilbert(
        signal_data,
        sampling_rate=1000,
    )

    assert result["analytic_signal"].shape == signal_data.shape
    assert result["envelope"].shape == signal_data.shape
    assert result["phase"].shape == signal_data.shape
    assert result[
        "instantaneous_frequency"
    ].shape == signal_data.shape


def test_hilbert_rejects_invalid_sampling_rate():

    signal_data = np.ones(100)

    try:
        compute_hilbert(
            signal_data,
            sampling_rate=0,
        )
        assert False
    except ValueError:
        assert True

def test_cross_correlation_identical_signals():

    signal_data = np.sin(
        2 * np.pi * 20 * np.arange(1000) / 1000
    )

    lags, correlation = compute_cross_correlation(
        signal_data,
        signal_data,
    )

    peak_lag, peak_value = find_correlation_peak(
        lags,
        correlation,
    )

    assert peak_lag == 0
    assert np.isclose(
        peak_value,
        1.0,
        atol=1e-10,
    )

def test_cross_correlation_detects_shift():

    base = np.zeros(1000)

    base[200:250] = np.hanning(50)

    shifted = np.zeros(1000)

    shifted[225:275] = np.hanning(50)

    lags, correlation = compute_cross_correlation(
        base,
        shifted,
    )

    peak_lag, _ = find_correlation_peak(
        lags,
        correlation,
    )

    assert abs(peak_lag) == 25

    
def test_cross_correlation_output_lengths():

    signal_a = np.ones(100)
    signal_b = np.ones(50)

    # Use non-constant signals because constant signals
    # have zero variance after mean removal.
    signal_a = np.sin(
        2 * np.pi * 10 * np.arange(100) / 100
    )

    signal_b = np.sin(
        2 * np.pi * 10 * np.arange(50) / 100
    )

    lags, correlation = compute_cross_correlation(
        signal_a,
        signal_b,
    )

    assert len(lags) == 149
    assert len(correlation) == 149


def test_cross_correlation_rejects_empty_signal():

    try:
        compute_cross_correlation(
            np.array([]),
            np.ones(10),
        )
        assert False
    except ValueError:
        assert True

def test_stft_detects_sine_frequency():

    sampling_rate = 1000
    frequency = 100

    t = np.arange(5000) / sampling_rate

    signal_data = np.sin(
        2 * np.pi * frequency * t
    )

    frequencies, times, magnitude = compute_stft(
        signal_data,
        sampling_rate=sampling_rate,
        nperseg=500,
        noverlap=250,
    )

    assert magnitude.shape == (
        len(frequencies),
        len(times),
    )

    peak_frequency_index = np.argmax(
        np.mean(magnitude, axis=1)
    )

    detected_frequency = frequencies[
        peak_frequency_index
    ]

    assert np.isclose(
        detected_frequency,
        frequency,
        atol=2.0,
    )


def test_stft_returns_valid_output():

    signal_data = np.random.default_rng(
        42
    ).normal(
        size=2000
    )

    frequencies, times, magnitude = compute_stft(
        signal_data,
        sampling_rate=1000,
        nperseg=256,
    )

    assert len(frequencies) > 0
    assert len(times) > 0

    assert magnitude.ndim == 2

    assert np.isfinite(frequencies).all()
    assert np.isfinite(times).all()
    assert np.isfinite(magnitude).all()


def test_stft_rejects_invalid_overlap():

    signal_data = np.ones(1000)

    try:
        compute_stft(
            signal_data,
            sampling_rate=1000,
            nperseg=256,
            noverlap=256,
        )
        assert False
    except ValueError:
        assert True


def test_stft_rejects_invalid_sampling_rate():

    signal_data = np.ones(1000)

    try:
        compute_stft(
            signal_data,
            sampling_rate=0,
        )
        assert False
    except ValueError:
        assert True

def test_psd_detects_sine_frequency():

    sampling_rate = 1000
    frequency = 100

    t = np.arange(2000) / sampling_rate

    signal_data = np.sin(
        2 * np.pi * frequency * t
    )

    frequencies, psd = compute_psd(
        signal_data,
        sampling_rate,
        nperseg=1000,
    )

    peak_index = np.argmax(psd)

    detected_frequency = frequencies[
        peak_index
    ]

    assert np.isclose(
        detected_frequency,
        frequency,
        atol=2.0,
    )


def test_psd_returns_valid_arrays():

    signal_data = np.random.default_rng(
        42
    ).normal(
        size=1000
    )

    frequencies, psd = compute_psd(
        signal_data,
        sampling_rate=1000,
    )

    assert len(frequencies) == len(psd)
    assert len(frequencies) > 0
    assert np.isfinite(frequencies).all()
    assert np.isfinite(psd).all()


def test_psd_is_nonnegative():

    signal_data = np.random.default_rng(
        42
    ).normal(
        size=1000
    )

    _, psd = compute_psd(
        signal_data,
        sampling_rate=1000,
    )

    assert np.all(psd >= 0)


def test_psd_rejects_invalid_sampling_rate():

    signal_data = np.ones(100)

    try:
        compute_psd(
            signal_data,
            sampling_rate=0,
        )
        assert False
    except ValueError:
        assert True

def test_fft_detects_sine_frequency():

    sampling_rate = 1000
    duration = 1.0
    frequency = 100

    t = np.arange(
        int(sampling_rate * duration)
    ) / sampling_rate

    signal = np.sin(
        2 * np.pi * frequency * t
    )

    frequencies, magnitude = compute_fft(
        signal,
        sampling_rate,
    )

    peak_index = np.argmax(magnitude)

    detected_frequency = frequencies[
        peak_index
    ]

    assert np.isclose(
        detected_frequency,
        frequency,
        atol=1.0,
    )


def test_fft_returns_one_sided_spectrum():

    signal = np.ones(1000)

    frequencies, magnitude = compute_fft(
        signal,
        sampling_rate=1000,
    )

    assert len(frequencies) == 501
    assert len(magnitude) == 501


def test_fft_frequency_range():

    signal = np.zeros(1000)

    frequencies, _ = compute_fft(
        signal,
        sampling_rate=1000,
    )

    assert np.isclose(
        frequencies[0],
        0.0,
    )

    assert np.isclose(
        frequencies[-1],
        500.0,
    )


def test_fft_rejects_invalid_sampling_rate():

    signal = np.ones(100)

    try:
        compute_fft(
            signal,
            sampling_rate=0,
        )
        assert False
    except ValueError:
        assert True