"""
tests/test_image_to_mat.py
==========================
Unit and integration tests for the Image to .MAT Converter module.
Verifies conversion modes, file writing, and compatibility with Spandhan's signal loader.
"""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from converters.image_to_mat.converter import (
    load_image_as_array,
    convert_image_to_signal,
    save_signal_to_mat,
    convert_image_to_mat,
    CONVERSION_MODES,
)
from data_io.signal_loader import load_signal


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary test PNG image."""
    img_data = (np.arange(100 * 80).reshape(100, 80) % 256).astype(np.uint8)
    img_file = tmp_path / "test_scan.png"
    Image.fromarray(img_data).save(img_file)
    return img_file


def test_load_image_as_array(sample_image_path):
    arr = load_image_as_array(sample_image_path, grayscale=True, normalize=True)
    assert arr.ndim == 2
    assert arr.shape == (100, 80)
    assert arr.dtype == np.float64
    assert 0.0 <= arr.min() <= arr.max() <= 1.0


def test_convert_modes():
    test_img = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ], dtype=np.float64)

    # 1. Raster
    sig_raster = convert_image_to_signal(test_img, mode="raster")
    assert sig_raster.shape == (9,)
    assert np.allclose(sig_raster, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    # 2. Row Mean
    sig_row = convert_image_to_signal(test_img, mode="row_mean")
    assert sig_row.shape == (3,)
    assert np.allclose(sig_row, [0.2, 0.5, 0.8])

    # 3. Col Mean
    sig_col = convert_image_to_signal(test_img, mode="col_mean")
    assert sig_col.shape == (3,)
    assert np.allclose(sig_col, [0.4, 0.5, 0.6])

    # 4. Center Row
    sig_center_r = convert_image_to_signal(test_img, mode="center_row")
    assert sig_center_r.shape == (3,)
    assert np.allclose(sig_center_r, [0.4, 0.5, 0.6])

    # 5. Center Col
    sig_center_c = convert_image_to_signal(test_img, mode="center_col")
    assert sig_center_c.shape == (3,)
    assert np.allclose(sig_center_c, [0.2, 0.5, 0.8])


def test_end_to_end_conversion_and_spandhan_loader(sample_image_path, tmp_path):
    for mode in CONVERSION_MODES.keys():
        out_mat = tmp_path / f"output_{mode}.mat"
        saved_path, signal, info = convert_image_to_mat(
            image_path=sample_image_path,
            output_mat_path=out_mat,
            mode=mode,
            sampling_rate=1000.0,
        )

        assert saved_path.exists()
        assert len(signal) > 0

        # CRITICAL TEST: Verify Spandhan's signal_loader loads this MAT file perfectly
        signal_data = load_signal(
            file_path=saved_path,
            sampling_rate=1000.0,
            domain="general",
        )

        assert signal_data.signal is not None
        assert signal_data.signal.ndim == 1
        assert len(signal_data.signal) == len(signal)
        assert signal_data.sampling_rate == 1000.0
        assert np.allclose(signal_data.signal, signal)
