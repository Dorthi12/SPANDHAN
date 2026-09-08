"""
scripts/create_reference_images.py
==================================
Creates sample reference test images for the user to upload and test in the ML page.
Saves to samples/
  - sample_clean.png
  - sample_gaussian_noise.png
  - sample_salt_pepper_noise.png
  - sample_periodic_noise.png
"""

import numpy as np
from PIL import Image
from pathlib import Path

out_dir = Path("samples")
out_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# Base synthetic image (circles + gradient pattern)
y, x = np.ogrid[:256, :256]
base = 0.5 + 0.3 * np.sin(x / 20.0) * np.cos(y / 20.0)
circle_mask = (x - 128)**2 + (y - 128)**2 < 60**2
base[circle_mask] = 0.8
base = np.clip(base, 0.0, 1.0)

# 1. Clean
clean_img = (base * 255).astype(np.uint8)
Image.fromarray(clean_img).save(out_dir / "sample_clean.png")

# 2. Gaussian noise
gauss_noise = base + rng.normal(0, 0.15, base.shape)
gauss_img = (np.clip(gauss_noise, 0.0, 1.0) * 255).astype(np.uint8)
Image.fromarray(gauss_img).save(out_dir / "sample_gaussian_noise.png")

# 3. Salt and Pepper noise
sp_noise = base.copy()
sp_mask_salt = rng.uniform(0, 1, base.shape) < 0.05
sp_mask_pepp = rng.uniform(0, 1, base.shape) < 0.05
sp_noise[sp_mask_salt] = 1.0
sp_noise[sp_mask_pepp] = 0.0
sp_img = (sp_noise * 255).astype(np.uint8)
Image.fromarray(sp_img).save(out_dir / "sample_salt_pepper_noise.png")

# 4. Periodic noise (horizontal stripes)
stripe_noise = base + 0.25 * np.sin(2 * np.pi * y / 8.0)
stripe_img = (np.clip(stripe_noise, 0.0, 1.0) * 255).astype(np.uint8)
Image.fromarray(stripe_img).save(out_dir / "sample_periodic_noise.png")

print("Created sample reference images in samples/:")
for p in out_dir.glob("*.png"):
    print(f"  - {p.resolve()}")
