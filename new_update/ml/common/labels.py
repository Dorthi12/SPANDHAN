"""
SPANDHAN — Audio Signal Class Labels  (audio-only)
====================================================
Single source of truth for the 5 class names and their integer IDs.
Import from here in BOTH training and inference.
"""

CLASS_NAMES: list[str] = [
    "impulse",     # 0
    "sinusoidal",  # 1
    "white_noise", # 2
    "step",        # 3
    "chirp",       # 4
]

CLASS_TO_ID: dict[str, int] = {n: i for i, n in enumerate(CLASS_NAMES)}
ID_TO_CLASS: dict[int, str] = {i: n for i, n in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)


def label_to_id(label: str) -> int:
    """Convert class name → integer ID.  Raises ValueError on unknown label."""
    try:
        return CLASS_TO_ID[label.lower().strip()]
    except KeyError:
        raise ValueError(f"Unknown label '{label}'. Expected one of: {CLASS_NAMES}")


def id_to_label(class_id: int) -> str:
    """Convert integer ID → class name.  Raises ValueError on out-of-range ID."""
    try:
        return ID_TO_CLASS[int(class_id)]
    except KeyError:
        raise ValueError(
            f"Unknown class_id {class_id}. Expected integer in [0, {NUM_CLASSES - 1}]."
        )
