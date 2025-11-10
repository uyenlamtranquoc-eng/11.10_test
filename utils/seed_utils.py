import os
import numpy as np
import torch
import random


def set_seed(seed: int = 0, worker_id: int = 0) -> int:
    """Set global random seeds for reproducibility and return the actual seed.

    Mirrors training script behavior:
    - Derive worker-specific seed via `seed + worker_id * 10000`.
    - Seed Python, NumPy, Torch (and CUDA if available).
    - Configure CuDNN for determinism.
    - Set environment vars to reduce hidden randomness.
    """
    worker_seed = int(seed) + int(worker_id) * 10000

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(worker_seed)
    os.environ["SUMO_RANDOM"] = "false"

    return worker_seed


__all__ = ["set_seed"]