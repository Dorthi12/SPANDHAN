# SPANDHAN: Signal Processing & Machine Learning Framework

SPANDHAN is an integrated platform for Digital Signal Processing (DSP) and Machine Learning (ML) analysis of audio and image signals.

## Project Structure

- `matlab/`: MATLAB algorithms for DSP, preprocessing, and visualization.
- `python/`: Python modules for feature extraction, machine learning classification, and inference.
- `models/`: Pre-trained ML classifiers and scalers for audio and image signals.
- `datasets/`: Dataset categories (impulse, sinusoidal, white noise, step, chirp).
- `tests/`: Unit test suites for MATLAB and Python components.
- `data/`: Input data samples and output results/figures.
- `docs/`: Technical documentation and specifications.

## Setup & Usage

### Python Setup
```bash
cd python
pip install -r requirements.txt
python common/predict.py --input ../data/input/audio/sample.wav --type audio
```

### MATLAB Setup
Open MATLAB and add the `matlab/` directory to your search path, then run `main.m`.
