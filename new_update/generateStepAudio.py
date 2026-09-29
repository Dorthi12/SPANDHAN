#!/usr/bin/env python3
"""
SPANDHAN Signal Processing Project
Synthetic Step Audio Dataset Generator

Generates a research-grade synthetic audio dataset containing 1,000 step-function WAV samples
with randomized physical and DSP parameters, detailed metadata CSV, dataset validation,
and preview plots.
"""

from python.audio.generateStepAudio import generate_step_dataset

if __name__ == "__main__":
    generate_step_dataset()
