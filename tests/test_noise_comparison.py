import numpy as np
import pytest

from intelligence.noise.comparison import compare_signal_quality


def test_clean_signal_has_infinite_snr():
    reference = np.sin(
        2 * np.pi * 50 * np.arange(1000) / 1000
    )

    result = compare_signal_quality(
        reference,
        reference,
        reference,
        1000,
    )

    assert np.isinf(result["before"]["snr_db"])
    assert np.isinf(result["after"]["snr_db"])
    assert np.isclose(result["before"]["rmse"], 0.0)
    assert np.isclose(result["after"]["rmse"], 0.0)


def test_denoising_improves_snr():
    rng = np.random.default_rng(42)

    reference = np.sin(
        2 * np.pi * 50 * np.arange(2000) / 1000
    )

    before = reference + 0.5 * rng.normal(
        size=len(reference)
    )

    after = reference + 0.1 * rng.normal(
        size=len(reference)
    )

    result = compare_signal_quality(
        reference,
        before,
        after,
        1000,
    )

    assert result["after"]["snr_db"] > result["before"]["snr_db"]
    assert result["improvement"]["snr_db"] > 0


def test_denoising_reduces_rmse():
    rng = np.random.default_rng(42)

    reference = np.sin(
        2 * np.pi * 50 * np.arange(1000) / 1000
    )

    before = reference + 0.5 * rng.normal(
        size=len(reference)
    )

    after = reference + 0.05 * rng.normal(
        size=len(reference)
    )

    result = compare_signal_quality(
        reference,
        before,
        after,
        1000,
    )

    assert result["after"]["rmse"] < result["before"]["rmse"]
    assert result["improvement"]["rmse_reduction"] > 0


def test_correlation_improves():
    rng = np.random.default_rng(42)

    reference = np.sin(
        2 * np.pi * 50 * np.arange(1000) / 1000
    )

    before = reference + rng.normal(
        0,
        0.5,
        len(reference),
    )

    after = reference + rng.normal(
        0,
        0.05,
        len(reference),
    )

    result = compare_signal_quality(
        reference,
        before,
        after,
        1000,
    )

    assert (
        result["after"]["correlation"]
        > result["before"]["correlation"]
    )


def test_energy_is_calculated():
    reference = np.ones(100)

    result = compare_signal_quality(
        reference,
        reference,
        reference,
        1000,
    )

    assert np.isclose(
        result["reference_energy"]
        if "reference_energy" in result
        else result["before"]["energy"],
        100.0,
    )


def test_signal_lengths_must_match():
    reference = np.ones(100)
    before = np.ones(100)
    after = np.ones(200)

    with pytest.raises(ValueError):
        compare_signal_quality(
            reference,
            before,
            after,
            1000,
        )


def test_invalid_sampling_rate():
    signal = np.ones(100)

    with pytest.raises(ValueError):
        compare_signal_quality(
            signal,
            signal,
            signal,
            0,
        )


def test_invalid_signal():
    reference = np.ones(100)
    before = np.ones(100)
    after = np.full(100, np.nan)

    with pytest.raises(ValueError):
        compare_signal_quality(
            reference,
            before,
            after,
            1000,
        )