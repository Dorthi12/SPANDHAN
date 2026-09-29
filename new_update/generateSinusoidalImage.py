#!/usr/bin/env python3
"""
SPANDHAN Signal Processing Project
Synthetic 2D Sinusoidal Grating Image Dataset Generator

Generates a research-grade synthetic image dataset containing 1,000 8-bit grayscale PNG images (128x128)
with randomized 2D spatial sinusoidal grating parameters, detailed metadata CSV,
automated 2D FFT validation, and preview plots.
"""

from dataset_generation.image.sinusoidal.generate_sinusoidal_dataset import generate_sinusoidal_dataset

if __name__ == "__main__":
    generate_sinusoidal_dataset(output_base_dir="datasets/image", num_samples=1000, seed=42)
