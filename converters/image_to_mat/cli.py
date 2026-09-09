"""
converters/image_to_mat/cli.py
==============================
Command-line interface to convert image files to .mat format for Spandhan.

Usage:
------
python -m converters.image_to_mat.cli path/to/image.png -o output.mat --mode raster --fs 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from converters.image_to_mat.converter import (
    convert_image_to_mat,
    CONVERSION_MODES,
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert 2D image files into 1D MATLAB .mat signal files for Spandhan."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image file (.png, .jpg, .bmp, .tiff, .npy)",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output .mat file (defaults to output_mat/<name>_<mode>.mat)",
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=list(CONVERSION_MODES.keys()),
        default="raster",
        help="Signal extraction mode (raster, row_mean, col_mean, center_row, center_col)",
    )
    parser.add_argument(
        "--fs", "--sampling-rate",
        type=float,
        default=1000.0,
        help="Sampling rate in Hz (default: 1000.0)",
    )
    parser.add_argument(
        "--no-norm",
        action="store_true",
        help="Do not normalize image pixels to [0.0, 1.0]",
    )

    args = parser.parse_args()

    try:
        out_path, signal, info = convert_image_to_mat(
            image_path=args.image_path,
            output_mat_path=args.output,
            mode=args.mode,
            sampling_rate=args.fs,
            normalize=not args.no_norm,
        )
        print("=" * 60)
        print("SPANDHAN — IMAGE TO .MAT CONVERTER")
        print("=" * 60)
        print(f"Input Image   : {info['input_image']}")
        print(f"Image Shape   : {info['image_shape'][0]} x {info['image_shape'][1]}")
        print(f"Extraction    : {info['mode']}")
        print(f"Signal Length : {info['signal_length']:,} samples")
        print(f"Sampling Rate : {info['sampling_rate']:.1f} Hz")
        print(f"Output .MAT   : {info['output_mat']}")
        print("=" * 60)
        print("✓ Successfully generated .mat file ready for Spandhan!")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
