"""
SPANDHAN — Training: Build Dataset
====================================
Walks all five class WAV folders, extracts 35 features per file,
saves X.npy / y.npy / filenames.npy to data/output/predictions/.

Run once before training:
    python -m ml.audio.training.build_dataset
"""

from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.common.config  import CLASS_DIRS, REPORTS_DIR, X_PATH, Y_PATH, FILENAMES_PATH
from ml.common.labels  import CLASS_NAMES, label_to_id
from ml.audio.features import extract_features


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    from scipy.io import wavfile
    sr, data = wavfile.read(str(path))
    if data.dtype.kind == "i":
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)
    if data.ndim == 2:          # stereo → mono
        data = data.mean(axis=1)
    return data, int(sr)


def build_dataset(verbose: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_rows:    list[np.ndarray] = []
    y_rows:    list[int]        = []
    filenames: list[str]        = []

    for cls in CLASS_NAMES:
        cls_dir = CLASS_DIRS.get(cls)
        if cls_dir is None or not cls_dir.exists():
            print(f"  [WARN] Missing folder for '{cls}': {cls_dir}")
            continue

        wavs = sorted(cls_dir.glob("*.wav"))
        if not wavs:
            print(f"  [WARN] No WAVs in {cls_dir}")
            continue

        if verbose:
            print(f"  [{cls:11s}] {len(wavs)} files …", flush=True)

        cid = label_to_id(cls)
        ok = err = 0
        for wav in wavs:
            try:
                audio, sr = _load_wav(wav)
                feat      = extract_features(audio, sr)
                X_rows.append(feat)
                y_rows.append(cid)
                filenames.append(str(wav))
                ok += 1
            except Exception as e:          # noqa: BLE001
                err += 1
                if verbose:
                    print(f"    [ERR] {wav.name}: {e}")
        if verbose:
            print(f"    >> {ok} ok, {err} errors")

    X = np.vstack(X_rows).astype(np.float64)
    y = np.array(y_rows, dtype=np.int64)
    return X, y, filenames


def save_dataset(X: np.ndarray, y: np.ndarray, filenames: list[str]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(X_PATH), X)
    np.save(str(Y_PATH), y)
    np.save(str(FILENAMES_PATH), np.array(filenames, dtype=object))
    print(f"\nSaved  X={X.shape}  y={y.shape}  to {REPORTS_DIR}")


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    print("SPANDHAN - Build Audio Dataset")
    print("=" * 60)
    X, y, fns = build_dataset()
    from ml.common.labels import ID_TO_CLASS
    print(f"\nTotal: {len(y)} samples")
    for cid, cname in ID_TO_CLASS.items():
        print(f"  {cname:12s}: {int(np.sum(y == cid))}")
    save_dataset(X, y, fns)
    print(f"\nFinished in {time.time()-t0:.1f}s")
