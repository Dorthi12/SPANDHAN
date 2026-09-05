import numpy as np

from dsp.results import DSPResult, FrequencyResult


def test_dsp_result_stores_data():
    frequencies = np.array([50.0, 100.0])
    magnitude = np.array([1.0, 0.5])

    result = DSPResult(
        method="FFT",
        data={
            "frequency_axis": frequencies,
            "magnitude": magnitude,
        },
        parameters={
            "sampling_rate": 1000,
        },
    )

    assert result.method == "FFT"
    assert np.array_equal(result.get("frequency_axis"), frequencies)
    assert np.array_equal(result.get("magnitude"), magnitude)
    assert result.parameters["sampling_rate"] == 1000


def test_dsp_result_missing_value():
    result = DSPResult(method="FFT")

    assert result.get("missing") is None
    assert result.get("missing", 123) == 123


def test_frequency_result():
    frequencies = [49.99, 51.00]

    result = FrequencyResult(
        method="MUSIC",
        frequencies=frequencies,
    )

    assert result.method == "MUSIC"
    assert isinstance(result.frequencies, np.ndarray)
    assert result.frequencies.dtype == np.float64
    assert np.allclose(result.frequencies, [49.99, 51.00])