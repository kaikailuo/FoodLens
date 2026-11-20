from dataclasses import asdict
import platform
import random
import os
from pathlib import Path
import numpy as np
import torch
import torchvision


def random_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, scheduler, *, vocab, statistics, metadata, step,
                    scaler=None, progress=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": 2, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "model_config": asdict(model.config), "vocab": vocab.tokens,
        "statistics": asdict(statistics) if statistics is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "progress": progress or {}, "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(), "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "step": step, "metadata": metadata,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "torchvision": str(torchvision.__version__)}}, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, *, restore_rng=False, scaler=None):
    # Local checkpoints contain Python RNG tuples; only load trusted local artifacts.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and checkpoint["scheduler"]:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if restore_rng:
        random.setstate(checkpoint["rng_python"])
        torch.set_rng_state(checkpoint["rng_torch"])
        if checkpoint.get("rng_numpy") is not None:
            np.random.set_state(checkpoint["rng_numpy"])
        if torch.cuda.is_available() and checkpoint["rng_cuda"]:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
    return checkpoint
