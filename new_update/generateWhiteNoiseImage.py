#!/usr/bin/env python3
"""
SPANDHAN Signal Processing Project
Synthetic 2D White Noise Image Dataset Generator

Generates a research-grade synthetic image dataset containing 1,000 8-bit grayscale PNG images (128x128)
with randomized 2D spatial white noise distributions, detailed metadata CSV,
automated 2D FFT & autocorrelation validation, and preview plots.
"""

from dataset_generation.image.white_noise.generate_white_noise_dataset import generate_white_noise_dataset

if __name__ == "__main__":
    generate_white_noise_dataset()
