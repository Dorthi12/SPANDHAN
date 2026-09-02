"""
Test Preprocessing module.
"""

import numpy as np

from preprocessing.dc_removal import remove_dc


def test_remove_dc():

    signal = np.array([
        3.0,
        4.0,
        5.0,
        6.0,
    ])

    processed = remove_dc(signal)

    assert np.isclose(np.mean(processed), 0.0)

    assert processed.shape == signal.shape

    assert np.isfinite(processed).all()


def test_remove_dc_preserves_shape():

    signal = np.linspace(
        1.0,
        10.0,
        1000,
    )

    processed = remove_dc(signal)

    assert processed.shape == signal.shape


def test_remove_dc_rejects_empty_signal():

    signal = np.array([])

    try:
        remove_dc(signal)
        assert False
    except ValueError:
        assert True