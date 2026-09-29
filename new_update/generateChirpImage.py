#!/usr/bin/env python3
"""
SPANDHAN Signal Processing Project
Synthetic 2D Chirp / Zone-Plate Image Dataset Generator

Generates a research-grade synthetic image dataset containing 1,000 8-bit grayscale PNG images (128x128)
with randomized 2D spatial chirp and Fresnel Zone Plate parameters, detailed metadata CSV,
automated 2D FFT validation, and preview plots.
"""

from dataset_generation.image.chirp.generate_chirp_dataset import generate_chirp_dataset

if __name__ == "__main__":
    generate_chirp_dataset()
