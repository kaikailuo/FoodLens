"""Single-process device/precision, metadata and durable JSON utilities."""
from contextlib import nullcontext
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import torch


def require(package, capability):
    try:
        return __import__(package)
    except ImportError as error:
        raise RuntimeError(f"{capability} requires optional dependency {package}; see requirements-*.txt") from error


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def append_log(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def runtime(device="cpu", precision="fp32", deterministic=True):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("only CPU/CUDA are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use device=cpu for engineering tests")
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32/bf16/fp16")
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("CPU verification uses fp32; mixed precision requires CUDA")
    if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("this CUDA device does not support BF16")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    return device, torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")


def autocast(device, precision):
    return nullcontext() if precision == "fp32" else torch.autocast(
        device_type=device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def environment():
    from . import __version__
    versions = {"foodlens": __version__, "python": platform.python_version(), "platform": platform.platform()}
    package = Path(__file__).parent
    source_digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        source_digest.update(path.relative_to(package).as_posix().encode())
        source_digest.update(path.read_bytes())
    versions["python_source_sha256"] = source_digest.hexdigest()
    for name in ("torch", "torchvision", "numpy", "pillow", "pycocoevalcap", "transformers", "peft", "accelerate", "scipy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        repository = package.parents[1]
        versions["git_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=repository).strip()
        diff = subprocess.check_output(["git", "diff", "--", "foodlens_engineering", "AGENTS.md"], cwd=repository)
        versions["tracked_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        versions["git_revision"] = None
    versions["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        versions["gpu"] = torch.cuda.get_device_name()
    return versions


def reject_secrets(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in {"api_key", "token", "password", "secret", "authorization"}:
                raise ValueError(f"credential field {key} is forbidden; use an environment variable")
            reject_secrets(child)
    elif isinstance(value, list):
        for child in value:
            reject_secrets(child)
