import numpy as np
from scipy.signal import find_peaks


def _validate_signal(signal_data):
    signal_data = np.asarray(signal_data, dtype=np.float64)

    if signal_data.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if signal_data.size == 0:
        raise ValueError("Signal is empty.")

    if not np.all(np.isfinite(signal_data)):
        raise ValueError("Signal contains NaN or infinite values.")

    return signal_data


def _validate_sampling_rate(sampling_rate):
    sampling_rate = float(sampling_rate)

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be finite and greater than zero."
        )

    return sampling_rate


def _validate_parameters(signal_length, model_order, num_sources):
    if not isinstance(model_order, int) or model_order < 2:
        raise ValueError(
            "Model order must be an integer >= 2."
        )

    if not isinstance(num_sources, int) or num_sources < 1:
        raise ValueError(
            "Number of sources must be a positive integer."
        )

    if num_sources >= model_order:
        raise ValueError(
            "Number of sources must be smaller than model order."
        )

    if model_order >= signal_length:
        raise ValueError(
            "Model order must be smaller than signal length."
        )

    if model_order > signal_length // 2:
        raise ValueError(
            "Model order should not exceed half the signal length."
        )


def _build_data_matrix(signal_data, model_order):
    """
    Construct a Hankel-style snapshot matrix.

    Shape:
        (model_order, snapshots)
    """

    snapshots = len(signal_data) - model_order + 1

    if snapshots < 2:
        raise ValueError(
            "Signal is too short for the requested model order."
        )

    data_matrix = np.empty(
        (model_order, snapshots),
        dtype=np.float64,
    )

    for row in range(model_order):
        data_matrix[row] = signal_data[
            row:row + snapshots
        ]

    return data_matrix


def _estimate_noise_subspace(
    data_matrix,
    num_sources,
):
    """
    Estimate the noise subspace using eigendecomposition.
    """

    covariance = (
        data_matrix @ data_matrix.T
    ) / data_matrix.shape[1]

    covariance = (
        covariance + covariance.T
    ) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    # Eigenvalues are ascending.
    #
    # For a real-valued sinusoid, positive and negative
    # frequency components appear as a pair. Therefore,
    # each real sinusoidal source contributes approximately
    # two dimensions to the covariance matrix.
    #
    # The caller specifies num_sources as the number of
    # physical sinusoidal sources.

    signal_dimension = 2 * num_sources

    if signal_dimension >= len(eigenvalues):
        raise ValueError(
            "Model order is too small for the requested "
            "number of sources."
        )

    noise_subspace = eigenvectors[
        :, :len(eigenvalues) - signal_dimension
    ]

    return eigenvalues, noise_subspace


def _music_spectrum(
    noise_subspace,
    sampling_rate,
    nfft,
    model_order,
):
    frequencies = np.linspace(
        0.0,
        sampling_rate / 2.0,
        nfft,
    )

    indices = np.arange(
        model_order
    )

    steering_matrix = np.exp(
        -2j
        * np.pi
        * np.outer(
            indices,
            frequencies,
        )
        / sampling_rate
    )

    projection = (
        noise_subspace.conj().T
        @ steering_matrix
    )

    denominator = np.sum(
        np.abs(projection) ** 2,
        axis=0,
    )

    denominator = np.maximum(
        denominator,
        np.finfo(float).eps,
    )

    pseudospectrum = 1.0 / denominator

    pseudospectrum /= np.max(
        pseudospectrum
    )

    return frequencies, pseudospectrum


def _find_music_peaks(
    frequencies,
    pseudospectrum,
    num_sources,
):
    """
    Extract the strongest separated MUSIC peaks.
    """

    if len(frequencies) < 2:
        return np.array([], dtype=np.float64)

    spacing = (
        frequencies[1] - frequencies[0]
    )

    minimum_separation_hz = 0.2

    distance = max(
        1,
        int(
            minimum_separation_hz / spacing
        ),
    )

    peaks, _ = find_peaks(
        pseudospectrum,
        distance=distance,
        prominence = 1e-6,
    )

    if len(peaks) == 0:
        return np.array([], dtype=np.float64)

    strengths = pseudospectrum[peaks]

    strongest = np.argsort(
        strengths
    )[-num_sources:]

    selected = peaks[strongest]

    return np.sort(
        frequencies[selected]
    )


def estimate_music_frequencies(
    signal_data,
    sampling_rate,
    model_order,
    num_sources,
    nfft=4096,
):
    """
    Estimate sinusoidal frequencies using MUSIC.
    """

    signal_data = _validate_signal(
        signal_data
    )

    sampling_rate = _validate_sampling_rate(
        sampling_rate
    )

    _validate_parameters(
        len(signal_data),
        model_order,
        num_sources,
    )

    if not isinstance(nfft, int) or nfft < 2:
        raise ValueError(
            "nfft must be an integer >= 2."
        )

    data_matrix = _build_data_matrix(
        signal_data,
        model_order,
    )

    eigenvalues, noise_subspace = (
        _estimate_noise_subspace(
            data_matrix,
            num_sources,
        )
    )

    frequencies, pseudospectrum = (
        _music_spectrum(
            noise_subspace,
            sampling_rate,
            nfft,
            model_order,
        )
    )

    estimated_frequencies = (
        _find_music_peaks(
            frequencies,
            pseudospectrum,
            num_sources,
        )
    )

    return {
        "frequencies": estimated_frequencies,
        "frequency_axis": frequencies,
        "pseudospectrum": pseudospectrum,
        "eigenvalues": eigenvalues,
        "noise_subspace": noise_subspace,
        "model_order": model_order,
        "num_sources": num_sources,
    }