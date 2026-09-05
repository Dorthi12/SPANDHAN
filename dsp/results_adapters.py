from dsp.results import DSPResult, FrequencyResult


def fft_to_result(
    frequencies,
    magnitude,
    sampling_rate,
):
    return DSPResult(
        method="FFT",
        data={
            "frequency_axis": frequencies,
            "magnitude": magnitude,
        },
        parameters={
            "sampling_rate": sampling_rate,
        },
    )


def music_to_result(result):
    return FrequencyResult(
        method="MUSIC",
        frequencies=result["frequencies"],
        data={
            "frequency_axis": result["frequency_axis"],
            "pseudospectrum": result["pseudospectrum"],
            "eigenvalues": result["eigenvalues"],
            "noise_subspace": result["noise_subspace"],
        },
        parameters={
            "model_order": result["model_order"],
            "num_sources": result["num_sources"],
        },
    )


def esprit_to_result(result):
    return FrequencyResult(
        method="ESPRIT",
        frequencies=result["frequencies"],
        data={
            "eigenvalues": result["eigenvalues"],
            "signal_subspace": result["signal_subspace"],
        },
        parameters={
            "model_order": result["model_order"],
            "num_sources": result["num_sources"],
        },
    )