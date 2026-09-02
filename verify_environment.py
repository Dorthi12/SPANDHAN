import sys

import numpy as np
import scipy
import pandas as pd
import sklearn
import pywt
import matplotlib
import soundfile
import librosa
import PySide6
import cv2
import skimage
import joblib
import reportlab


def main():
    print("=" * 60)
    print("SPANDHAN - ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"Python       : {sys.version.split()[0]}")
    print(f"NumPy        : {np.__version__}")
    print(f"SciPy        : {scipy.__version__}")
    print(f"Pandas       : {pd.__version__}")
    print(f"Scikit-learn : {sklearn.__version__}")
    print(f"PyWavelets   : {pywt.__version__}")
    print(f"Matplotlib   : {matplotlib.__version__}")
    print(f"SoundFile    : {soundfile.__version__}")
    print(f"Librosa      : {librosa.__version__}")
    print(f"PySide6      : {PySide6.__version__}")
    print(f"OpenCV       : {cv2.__version__}")
    print(f"scikit-image : {skimage.__version__}")
    print(f"Joblib       : {joblib.__version__}")
    print(f"ReportLab    : {reportlab.Version}")

    print("\nEnvironment check PASSED.")


if __name__ == "__main__":
    main()