from dataclasses import dataclass
from typing import Any

import numpy as np

from generators.noise_generator import (
    add_colored_noise,
    add_gaussian_noise,
    add_impulse_noise,
    add_mixed_noise,
    add_periodic_noise,
)
from generators.waveform_generator import generate_sine

from intelligence.noise.features import extract_noise_features


NOISE_CLASSES = (
    "Clean",
    "Gaussian",
    "Impulse",
    "Periodic",
    "Colored",
    "Mixed",
)


@dataclass
class NoiseDataset:
    """
    Feature dataset for noise classification.

    X:
        Feature matrix with shape (n_samples, 14).

    y:
        Noise class labels.

    metadata:
        Generation metadata kept separate from X so hidden
        generation parameters cannot accidentally become ML features.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    metadata: list[dict[str, Any]]


def _validate_positive_integer(value, name):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{name} must be a positive integer."
        )


def _validate_sampling_rate(sampling_rate):
    sampling_rate = float(sampling_rate)

    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(
            "Sampling rate must be finite and greater than zero."
        )

    return sampling_rate


def _validate_noise_class(noise_class):
    if noise_class not in NOISE_CLASSES:
        raise ValueError(
            f"Unsupported noise class: {noise_class}"
        )


def _generate_reference_signal(
    length,
    sampling_rate,
    rng,
):
    """
    Generate an independent reference signal.

    The waveform generator returns (time, signal), so only
    the signal component is retained here.
    """

    duration = length / sampling_rate

    f1 = rng.uniform(
        20.0,
        min(200.0, sampling_rate / 4.0),
    )

    f2 = rng.uniform(
        250.0,
        min(400.0, sampling_rate / 2.5),
    )

    if f2 <= f1:
        f2 = min(
            sampling_rate / 3.0,
            f1 + 50.0,
        )

    amplitude_1 = rng.uniform(0.5, 1.0)
    amplitude_2 = rng.uniform(0.2, 0.6)

    _, signal_1 = generate_sine(
        sampling_rate=sampling_rate,
        frequency=f1,
        duration=duration,
        amplitude=amplitude_1,
    )

    _, signal_2 = generate_sine(
        sampling_rate=sampling_rate,
        frequency=f2,
        duration=duration,
        amplitude=amplitude_2,
        phase=rng.uniform(
            0.0,
            2.0 * np.pi,
        ),
    )

    return signal_1 + signal_2


def _add_noise(
    clean_signal,
    sampling_rate,
    noise_class,
    rng,
):
    """
    Add one noise type to an independently generated signal.
    """

    if noise_class == "Clean":
        return clean_signal.copy(), {}

    if noise_class == "Gaussian":
        snr_db = rng.uniform(5.0, 25.0)

        noisy_signal = add_gaussian_noise(
            clean_signal,
            snr_db=snr_db,
            random_state=rng,
        )

        return noisy_signal, {
            "snr_target_db": float(snr_db),
        }

    if noise_class == "Impulse":
        # add_impulse_noise has no snr_db parameter -- severity is
        # controlled entirely by `probability` (see test_noise.py).
        # The realized SNR is measured independently afterward in
        # build_noise_dataset(), same as every other class.
        probability = rng.uniform(0.001, 0.01)

        noisy_signal = add_impulse_noise(
            clean_signal,
            probability=probability,
            random_state=rng,
        )

        return noisy_signal, {
            "impulse_probability": float(probability),
        }

    if noise_class == "Periodic":
        frequency = rng.uniform(
            40.0,
            min(100.0, sampling_rate / 4.0),
        )

        amplitude = rng.uniform(
            0.05,
            0.4,
        )

        noisy_signal = add_periodic_noise(
            clean_signal,
            sampling_rate=sampling_rate,
            frequency=frequency,
            amplitude=amplitude,
        )

        return noisy_signal, {
            "periodic_frequency_hz": float(frequency),
            "periodic_amplitude": float(amplitude),
        }

    if noise_class == "Colored":
        # NOTE: add_colored_noise's real parameter (per test_noise.py)
        # is `strength`, not `snr_db` -- verify the expected scale
        # against generators/noise_generator.py. This range is a
        # placeholder matching test_noise.py's example (strength=0.1).
        strength = rng.uniform(0.05, 0.4)
        color = rng.choice(["pink", "brown"])

        noisy_signal = add_colored_noise(
            clean_signal,
            color=color,
            strength=strength,
            random_state=rng,
        )

        return noisy_signal, {
            "strength": float(strength),
            "color": str(color),
        }

    if noise_class == "Mixed":
        snr_db = rng.uniform(5.0, 25.0)
        periodic_frequency = rng.uniform(
            40.0,
            min(100.0, sampling_rate / 4.0),
        )

        noisy_signal = add_mixed_noise(
            clean_signal,
            sampling_rate=sampling_rate,
            snr_db=snr_db,
            periodic_frequency=periodic_frequency,
            random_state=rng,
        )

        return noisy_signal, {
            "snr_target_db": float(snr_db),
            "periodic_frequency_hz": float(
                periodic_frequency
            ),
        }

    raise ValueError(
        f"Unsupported noise class: {noise_class}"
    )


def build_noise_dataset(
    samples_per_class=20,
    length=2000,
    sampling_rate=1000,
    seed=42,
):
    """
    Build an independent synthetic noise classification dataset.

    Returns
    -------
    NoiseDataset
        Feature matrix, labels, feature names, and generation metadata.
    """

    _validate_positive_integer(
        samples_per_class,
        "samples_per_class",
    )

    _validate_positive_integer(
        length,
        "length",
    )

    sampling_rate = _validate_sampling_rate(
        sampling_rate
    )

    if seed is not None:
        if not isinstance(seed, int):
            raise ValueError(
                "seed must be an integer or None."
            )

    rng = np.random.default_rng(seed)

    feature_rows = []
    labels = []
    metadata = []

    for noise_class in NOISE_CLASSES:
        for sample_index in range(samples_per_class):
            clean_signal = _generate_reference_signal(
                length=length,
                sampling_rate=sampling_rate,
                rng=rng,
            )

            noisy_signal, generation_metadata = _add_noise(
                clean_signal=clean_signal,
                sampling_rate=sampling_rate,
                noise_class=noise_class,
                rng=rng,
            )

            # Estimate SNR independently from the generated
            # clean/noisy pair. Do not use target SNR metadata.
            if noise_class == "Clean":
                snr_db = np.inf
            else:
                noise = noisy_signal - clean_signal
                signal_power = np.mean(clean_signal ** 2)
                noise_power = np.mean(noise ** 2)

                if noise_power <= 0:
                    snr_db = np.inf
                else:
                    snr_db = 10.0 * np.log10(
                        signal_power / noise_power
                    )

            # extract_noise_features() intentionally rejects a
            # non-finite snr_db argument (see
            # test_noise_features.py::test_invalid_snr), so the
            # independently estimated value -- which is legitimately
            # +inf for Clean signals -- is attached to the feature
            # row after extraction instead of being passed into the
            # validated snr_db parameter.
            features = extract_noise_features(
                noisy_signal,
                sampling_rate,
            )
            features["snr_db"] = float(snr_db)

            feature_rows.append(
                [
                    features[name]
                    for name in features
                ]
            )

            labels.append(noise_class)

            metadata.append(
                {
                    "sample_index": sample_index,
                    "class": noise_class,
                    "sampling_rate": sampling_rate,
                    "length": length,
                    "generation": generation_metadata,
                }
            )

    X = np.asarray(
        feature_rows,
        dtype=np.float64,
    )

    y = np.asarray(
        labels,
        dtype=str,
    )

    feature_names = list(
        features.keys()
    )

    return NoiseDataset(
        X=X,
        y=y,
        feature_names=feature_names,
        metadata=metadata,
    )