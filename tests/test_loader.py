from pathlib import Path

from data_io.signal_loader import load_signal


def test_wav_loader():
    path = Path("datasets/synthetic/test_audio.wav")

    signal_data = load_signal(path)

    assert signal_data.signal is not None
    assert len(signal_data.signal) > 0
    assert signal_data.sampling_rate == 44100
    assert signal_data.filename == "test_audio.wav"
    assert signal_data.domain == "general"
    assert signal_data.duration > 0