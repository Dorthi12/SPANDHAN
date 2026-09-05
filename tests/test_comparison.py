import numpy as np
import pytest

from dsp.comparison import (
    FrequencyComparison,
    compare_frequency_estimates,
)


def test_frequency_comparison():
    result = compare_frequency_estimates(
        target_frequencies=[50.0, 51.0],
        estimated_frequencies={
            "MUSIC": [49.99, 50.998],
            "ESPRIT": [49.9996, 51.0006],
        },
        tolerance_hz=1.0,
    )

    assert isinstance(result, FrequencyComparison)

    assert np.allclose(
        result.target_frequencies,
        [50.0, 51.0],
    )

    assert result.resolved["MUSIC"] is True
    assert result.resolved["ESPRIT"] is True

    assert np.all(
        result.errors["MUSIC"] < 1.0
    )

    assert np.all(
        result.errors["ESPRIT"] < 1.0
    )


def test_frequency_comparison_detects_failure():
    result = compare_frequency_estimates(
        target_frequencies=[50.0, 51.0],
        estimated_frequencies={
            "MUSIC": [50.0, 60.0],
        },
        tolerance_hz=1.0,
    )

    assert result.resolved["MUSIC"] is False

    assert not np.isinf(
        result.errors["MUSIC"]
    ).any()

    assert result.errors["MUSIC"][0] == 0.0
    assert result.errors["MUSIC"][1] == 9.0


def test_frequency_comparison_wrong_number_of_estimates():
    result = compare_frequency_estimates(
        target_frequencies=[50.0, 51.0],
        estimated_frequencies={
            "MUSIC": [50.0],
        },
    )

    assert result.resolved["MUSIC"] is False

    assert np.all(
        np.isinf(result.errors["MUSIC"])
    )


def test_invalid_target_frequencies():
    with pytest.raises(ValueError):
        compare_frequency_estimates(
            target_frequencies=[],
            estimated_frequencies={"MUSIC": [50.0]},
        )


def test_invalid_tolerance():
    with pytest.raises(ValueError):
        compare_frequency_estimates(
            target_frequencies=[50.0],
            estimated_frequencies={"MUSIC": [50.0]},
            tolerance_hz=0,
        )