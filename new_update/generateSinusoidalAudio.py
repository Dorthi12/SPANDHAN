#!/usr/bin/env python3
"""
SPANDHAN Signal Processing Project
Synthetic Sinusoidal Audio Dataset Generator

Generates a research-grade synthetic audio dataset containing 1,000 sinusoidal WAV samples
with randomized physical and DSP parameters, detailed metadata CSV, dataset validation,
and preview plots.
"""

from python.audio.generateSinusoidalAudio import generate_sinusoidal_dataset

if __name__ == "__main__":
    generate_sinusoidal_dataset()
