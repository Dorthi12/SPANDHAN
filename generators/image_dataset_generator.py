"""
generators/image_dataset_generator.py
=======================================
Synthetic image dataset generator for Spandhan — Milestone E.

Design principles
-----------------
* Mirrors audio_dataset_generator.py exactly in structure/API.
* All images are 2-D float64 arrays in [0, 1] — no channel axis.
* 8 image types: checkerboard, gradient, circles, stripes,
  sinusoidal, gaussian_blob, natural_texture, noise_pattern.
* Uses only numpy + scipy — NO cv2/skimage dependency at generation time.
* Every sample carries complete, typed metadata (JSON-serialisable).
* SyntheticImageDataset exposes generate() and save() — stable interface.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.ndimage import gaussian_filter

# ------------------------------------------------------------------
# Supported image types
# ------------------------------------------------------------------
IMAGE_TYPES = (
    "checkerboard",
    "gradient",
    "circles",
    "stripes",
    "sinusoidal",
    "gaussian_blob",
    "natural_texture",
    "noise_pattern",
)

DEFAULT_IMAGE_DATASET_DIR = Path("datasets/synthetic/image")


# ==================================================================
# Data structures
# ==================================================================


@dataclass
class ImageSampleMetadata:
    """
    Complete, typed metadata for one generated image sample.
    All fields must be JSON-serialisable.
    """

    sample_id: str
    domain: str                     # always "image"
    image_type: str
    height: int
    width: int
    seed: Optional[int]

    # Per-type parameters (not all populated for every type)
    tile_size: Optional[int] = None           # checkerboard
    gradient_direction: Optional[str] = None  # gradient: "horizontal"|"vertical"|"diagonal"
    num_circles: Optional[int] = None         # circles
    stripe_frequency: Optional[float] = None  # stripes (cycles/pixel)
    spatial_frequency: Optional[float] = None # sinusoidal (cycles/pixel)
    orientation_deg: Optional[float] = None   # sinusoidal
    num_blobs: Optional[int] = None           # gaussian_blob
    blob_sigma: Optional[float] = None        # gaussian_blob
    roughness: Optional[float] = None         # natural_texture


@dataclass
class ImageSample:
    """
    One generated image sample: raw 2-D float64 array in [0,1] + metadata.

    image  — 2-D float64 NumPy array, shape (H, W), values in [0, 1].
    meta   — ImageSampleMetadata instance.
    """

    image: np.ndarray       # shape (H, W), float64, [0, 1]
    meta: ImageSampleMetadata


@dataclass
class ImageDatasetConfig:
    """
    Fully-specified configuration for one image dataset generation run.
    All fields are plain Python scalars or lists so the config can be
    serialised to JSON without a custom encoder.
    """

    image_types: list[str] = field(
        default_factory=lambda: list(IMAGE_TYPES)
    )
    samples_per_type: int = 3
    height: int = 128
    width: int = 128

    # Checkerboard
    tile_size: int = 16

    # Gradient
    gradient_direction: str = "horizontal"   # "horizontal" | "vertical" | "diagonal"

    # Circles
    num_circles: int = 5

    # Stripes
    stripe_frequency: float = 0.05          # cycles/pixel

    # Sinusoidal
    spatial_frequency: float = 0.04         # cycles/pixel
    orientation_deg: float = 0.0

    # Gaussian blobs
    num_blobs: int = 3
    blob_sigma: float = 15.0

    # Natural texture
    roughness: float = 0.7                  # 0=smooth, 1=rough

    seed: Optional[int] = 42


# ==================================================================
# Low-level image generators (numpy/scipy only)
# ==================================================================


def _checkerboard(
    height: int, width: int, tile_size: int
) -> np.ndarray:
    """Alternating black/white tiles."""
    row = (np.arange(height) // tile_size) % 2
    col = (np.arange(width) // tile_size) % 2
    img = np.bitwise_xor(row[:, None].astype(np.uint8),
                         col[None, :].astype(np.uint8)).astype(np.float64)
    return img


def _gradient(
    height: int, width: int, direction: str
) -> np.ndarray:
    """Smooth luminance ramp."""
    if direction == "horizontal":
        img = np.tile(np.linspace(0, 1, width), (height, 1))
    elif direction == "vertical":
        img = np.tile(np.linspace(0, 1, height)[:, None], (1, width))
    else:  # diagonal
        h = np.linspace(0, 1, height)[:, None]
        w = np.linspace(0, 1, width)[None, :]
        img = (h + w) / 2.0
    return img.astype(np.float64)


def _circles(
    height: int, width: int, num_circles: int
) -> np.ndarray:
    """Concentric ring pattern."""
    cy, cx = height / 2.0, width / 2.0
    y, x = np.ogrid[:height, :width]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = np.sqrt(cy ** 2 + cx ** 2)
    # Map radius to [0, num_circles] and use sine to make rings
    img = 0.5 + 0.5 * np.sin(2 * np.pi * num_circles * r / max_r)
    return img.astype(np.float64)


def _stripes(
    height: int, width: int, frequency: float, vertical: bool = False
) -> np.ndarray:
    """Periodic stripe pattern."""
    if vertical:
        axis = np.arange(width)[None, :] / width
    else:
        axis = np.arange(height)[:, None] / height
    img = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * height * axis)
    return np.broadcast_to(img, (height, width)).copy().astype(np.float64)


def _sinusoidal(
    height: int, width: int, spatial_freq: float, orientation_deg: float
) -> np.ndarray:
    """2-D sinusoidal grating at given spatial frequency and orientation."""
    theta = np.deg2rad(orientation_deg)
    y, x = np.mgrid[:height, :width]
    # Normalise coordinates to [0, 1]
    xn = x / width
    yn = y / height
    phase = 2 * np.pi * spatial_freq * (xn * np.cos(theta) + yn * np.sin(theta))
    img = 0.5 + 0.5 * np.sin(phase * max(height, width))
    return img.astype(np.float64)


def _gaussian_blob(
    height: int, width: int,
    num_blobs: int, sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sum of Gaussian intensity blobs placed at random positions."""
    img = np.zeros((height, width), dtype=np.float64)
    for _ in range(num_blobs):
        cy = int(rng.integers(0, height))
        cx = int(rng.integers(0, width))
        blob = np.zeros_like(img)
        blob[cy, cx] = 1.0
        blob = gaussian_filter(blob, sigma=sigma)
        if blob.max() > 0:
            blob /= blob.max()
        img += blob
    if img.max() > 0:
        img /= img.max()
    return img


def _natural_texture(
    height: int, width: int, roughness: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Fractal noise via Diamond-Square algorithm (simplified via FFT).
    roughness in (0, 1): 1 = rough (pink noise-like), 0 = smooth.
    """
    # 1/f^beta noise where beta = 2*(1 - roughness) + 1
    beta = 2.0 * (1.0 - roughness) + 1.0
    rows = 1 << int(np.ceil(np.log2(height)))
    cols = 1 << int(np.ceil(np.log2(width)))
    noise = rng.standard_normal((rows, cols))
    f = np.fft.fft2(noise)
    # frequency grid
    fy = np.fft.fftfreq(rows)[:, None]
    fx = np.fft.fftfreq(cols)[None, :]
    freq = np.sqrt(fy ** 2 + fx ** 2)
    freq[0, 0] = 1.0   # avoid div-by-zero at DC
    f_filtered = f / (freq ** (beta / 2.0))
    f_filtered[0, 0] = 0.0
    img = np.real(np.fft.ifft2(f_filtered))
    img = img[:height, :width]
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = np.zeros((height, width), dtype=np.float64)
    return img.astype(np.float64)


def _noise_pattern(
    height: int, width: int, rng: np.random.Generator
) -> np.ndarray:
    """Structured random: Gaussian noise smoothed with random sigma."""
    sigma = float(rng.uniform(2.0, 10.0))
    raw = rng.standard_normal((height, width))
    img = gaussian_filter(raw, sigma=sigma)
    lo, hi = img.min(), img.max()
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return img.astype(np.float64)


# ==================================================================
# High-level generator
# ==================================================================


def _build_sample(
    image_type: str,
    config: ImageDatasetConfig,
    rng: np.random.Generator,
    sample_index: int,
) -> ImageSample:
    """Generate one ImageSample of the given type."""
    H, W = config.height, config.width
    seed_used = (config.seed if config.seed is not None else 0) + sample_index

    meta_kwargs: dict[str, Any] = {
        "sample_id": str(uuid.uuid4()),
        "domain": "image",
        "image_type": image_type,
        "height": H,
        "width": W,
        "seed": seed_used,
    }

    if image_type == "checkerboard":
        img = _checkerboard(H, W, config.tile_size)
        meta_kwargs["tile_size"] = config.tile_size

    elif image_type == "gradient":
        # Vary direction across samples
        directions = ["horizontal", "vertical", "diagonal"]
        direction = directions[sample_index % len(directions)]
        img = _gradient(H, W, direction)
        meta_kwargs["gradient_direction"] = direction

    elif image_type == "circles":
        img = _circles(H, W, config.num_circles)
        meta_kwargs["num_circles"] = config.num_circles

    elif image_type == "stripes":
        vertical = (sample_index % 2 == 1)
        img = _stripes(H, W, config.stripe_frequency, vertical=vertical)
        meta_kwargs["stripe_frequency"] = config.stripe_frequency

    elif image_type == "sinusoidal":
        angle = config.orientation_deg + sample_index * 30.0
        img = _sinusoidal(H, W, config.spatial_frequency, angle)
        meta_kwargs["spatial_frequency"] = config.spatial_frequency
        meta_kwargs["orientation_deg"] = angle

    elif image_type == "gaussian_blob":
        img = _gaussian_blob(H, W, config.num_blobs, config.blob_sigma, rng)
        meta_kwargs["num_blobs"] = config.num_blobs
        meta_kwargs["blob_sigma"] = config.blob_sigma

    elif image_type == "natural_texture":
        img = _natural_texture(H, W, config.roughness, rng)
        meta_kwargs["roughness"] = config.roughness

    elif image_type == "noise_pattern":
        img = _noise_pattern(H, W, rng)

    else:
        raise ValueError(
            f"Unknown image_type '{image_type}'. "
            f"Valid types: {IMAGE_TYPES}"
        )

    return ImageSample(image=img, meta=ImageSampleMetadata(**meta_kwargs))


class SyntheticImageDataset:
    """
    Generates synthetic image datasets from an ImageDatasetConfig.

    Usage
    -----
    ds = SyntheticImageDataset(config)
    samples = ds.generate()           # list[ImageSample]
    ds.save(samples, output_dir)      # writes PNG + manifest JSON
    """

    def __init__(self, config: Optional[ImageDatasetConfig] = None) -> None:
        self.config = config or ImageDatasetConfig()

    # ------------------------------------------------------------------
    def generate(self) -> list[ImageSample]:
        """
        Generate all samples defined by the config.

        Returns
        -------
        list[ImageSample]
            One ImageSample per (image_type × sample_index) combination.
        """
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        samples: list[ImageSample] = []

        for image_type in cfg.image_types:
            if image_type not in IMAGE_TYPES:
                raise ValueError(
                    f"Unknown image_type '{image_type}'. Valid: {IMAGE_TYPES}"
                )
            for i in range(cfg.samples_per_type):
                sample = _build_sample(image_type, cfg, rng, i)
                samples.append(sample)

        return samples

    # ------------------------------------------------------------------
    def save(
        self,
        samples: list[ImageSample],
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save samples to disk as NPY files + a JSON manifest.

        Each sample is saved as ``<sample_id>.npy``.
        A ``manifest.json`` is written with all metadata.

        Returns
        -------
        Path
            The directory samples were written to.
        """
        out = Path(output_dir or DEFAULT_IMAGE_DATASET_DIR)
        out.mkdir(parents=True, exist_ok=True)

        manifest = []
        for sample in samples:
            npy_path = out / f"{sample.meta.sample_id}.npy"
            np.save(npy_path, sample.image)
            entry = asdict(sample.meta)
            entry["npy_file"] = npy_path.name
            manifest.append(entry)

        manifest_path = out / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        return out

    # ------------------------------------------------------------------
    @staticmethod
    def load(output_dir: Path) -> list[ImageSample]:
        """Load samples previously saved with save()."""
        out = Path(output_dir)
        manifest_path = out / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json in {out}")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        samples: list[ImageSample] = []
        for entry in manifest:
            npy_file = entry.pop("npy_file")
            img = np.load(out / npy_file)
            meta = ImageSampleMetadata(**entry)
            samples.append(ImageSample(image=img, meta=meta))
        return samples
