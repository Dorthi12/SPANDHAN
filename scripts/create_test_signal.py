import numpy as np
import soundfile as sf
from pathlib import Path


def main():
    fs = 44100
    duration = 2.0

    t = np.arange(0, duration, 1 / fs)

    signal = (
        0.6 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 880 * t)
    )

    output = Path("datasets/synthetic/test_audio.wav")
    output.parent.mkdir(parents=True, exist_ok=True)

    sf.write(output, signal, fs)

    print(f"Created: {output}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Samples: {len(signal)}")


if __name__ == "__main__":
    main()