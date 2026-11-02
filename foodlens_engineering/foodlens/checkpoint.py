from dataclasses import asdict
import platform
import random
import torch
import torchvision


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(path, model, optimizer, scheduler, *, vocab, statistics, metadata, step):
    torch.save({"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "model_config": asdict(model.config), "vocab": vocab.tokens, "statistics": asdict(statistics),
        "rng_python": random.getstate(), "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "step": step, "metadata": metadata,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "torchvision": str(torchvision.__version__)}}, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, *, restore_rng=False):
    # Local checkpoints contain Python RNG tuples; only load trusted local artifacts.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and checkpoint["scheduler"]:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if restore_rng:
        random.setstate(checkpoint["rng_python"])
        torch.set_rng_state(checkpoint["rng_torch"])
        if torch.cuda.is_available() and checkpoint["rng_cuda"]:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
    return checkpoint
