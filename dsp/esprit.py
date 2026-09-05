"""
Esprit module.
"""

import numpy as np


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
    Build a forward-shifted snapshot matrix.

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
        dtype=np.complex128,
    )

    for row in range(model_order):
        data_matrix[row, :] = signal_data[
            row:row + snapshots
        ]

    return data_matrix


def _estimate_signal_subspace(data_matrix, num_sources):
    """
    Estimate the signal subspace from the covariance matrix.

    For a real-valued signal, each sinusoidal source contributes
    approximately two conjugate frequency components.
    """

    covariance = (
        data_matrix @ data_matrix.conj().T
    ) / data_matrix.shape[1]

    covariance = (
        covariance + covariance.conj().T
    ) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    signal_dimension = 2 * num_sources

    if signal_dimension >= len(eigenvalues):
        raise ValueError(
            "Model order is too small for the requested "
            "number of sources."
        )

    # Largest eigenvalues correspond to the signal subspace.
    signal_subspace = eigenvectors[
        :, -signal_dimension:
    ]

    return eigenvalues, signal_subspace


def _estimate_frequencies_from_subspace(
    signal_subspace,
    sampling_rate,
    num_sources,
):
    """
    Estimate frequencies from the rotational invariance
    of the signal subspace.
    """

    # Remove the last row and first row to form
    # the two overlapping subspaces.
    subspace_upper = signal_subspace[:-1, :]
    subspace_lower = signal_subspace[1:, :]

    # Least-squares solution to:
    #
    #     U_upper @ Psi ≈ U_lower
    #
    psi = np.linalg.pinv(
        subspace_upper
    ) @ subspace_lower

    eigenvalues = np.linalg.eigvals(psi)

    phases = np.angle(eigenvalues)

    frequencies = (
        phases
        * sampling_rate
        / (2.0 * np.pi)
    )

    frequencies = np.abs(frequencies)

    frequencies = frequencies[
        frequencies <= sampling_rate / 2.0
    ]

    frequencies = np.sort(frequencies)

    # Real-valued signals produce conjugate pairs.
    # Keep one estimate from each pair.
    unique_frequencies = []

    tolerance = max(
        0.5,
        sampling_rate / 10000.0,
    )

    for frequency in frequencies:
        if not unique_frequencies:
            unique_frequencies.append(
                frequency
            )
            continue

        if abs(
            frequency - unique_frequencies[-1]
        ) > tolerance:
            unique_frequencies.append(
                frequency
            )

    unique_frequencies = np.asarray(
        unique_frequencies,
        dtype=np.float64,
    )

    if len(unique_frequencies) > num_sources:
        # Select the lowest-frequency estimates.
        #
        # This is only a defensive fallback. The normal
        # conjugate-pair removal should already leave the
        # requested number of physical sources.
        unique_frequencies = (
            unique_frequencies[:num_sources]
        )

    return unique_frequencies


def estimate_esprit_frequencies(
    signal_data,
    sampling_rate,
    model_order,
    num_sources,
):
    """
    Estimate sinusoidal frequencies using ESPRIT.

    Parameters
    ----------
    signal_data : array-like
        One-dimensional sampled signal.

    sampling_rate : float
        Sampling frequency in Hz.

    model_order : int
        Dimension of the covariance model.

    num_sources : int
        Number of sinusoidal sources.

    Returns
    -------
    dict
        ESPRIT frequency estimates and intermediate results.
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

    data_matrix = _build_data_matrix(
        signal_data,
        model_order,
    )

    eigenvalues, signal_subspace = (
        _estimate_signal_subspace(
            data_matrix,
            num_sources,
        )
    )

    frequencies = (
        _estimate_frequencies_from_subspace(
            signal_subspace,
            sampling_rate,
            num_sources,
        )
    )

    return {
        "frequencies": frequencies,
        "eigenvalues": eigenvalues,
        "signal_subspace": signal_subspace,
        "model_order": model_order,
        "num_sources": num_sources,
    }