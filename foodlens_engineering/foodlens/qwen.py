"""Qwen3-VL local loading, strict answer masks and language-only PEFT adapters."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import torch
from PIL import Image
from torch.utils.data import Dataset
from .data import read_jsonl, safe_image_path
from .extensions import qwen_messages
from .runtime import reject_secrets, require
from .schema import validate_record, strict_loads


@dataclass
class QwenConfig:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    processor_revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
    processor_use_fast: bool = True
    local_files_only: bool = True
    device: str = "cuda"
    precision: str = "bf16"
    quantization: str = "none"
    min_pixels: int = 3136
    max_pixels: int = 200704
    max_length: int = 4096
    max_new_tokens: int = 512
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    gradient_checkpointing: bool = True
    seed: int = 42

    def __post_init__(self):
        reject_secrets(asdict(self))
        if not isinstance(self.model_id, str) or not self.model_id.strip() or type(self.seed) is not int or self.seed < 0:
            raise ValueError("Qwen model_id and nonnegative integer seed required")
        if type(self.local_files_only) is not bool or type(self.gradient_checkpointing) is not bool or type(self.processor_use_fast) is not bool:
            raise ValueError("Qwen local_files_only/gradient_checkpointing must be boolean")
        for name in ("revision", "processor_revision"):
            if not re.fullmatch(r"[0-9a-f]{40}", getattr(self, name)):
                raise ValueError(f"Qwen {name} must be an immutable 40-character commit hash")
        if self.quantization not in {"none", "nf4"} or self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("invalid Qwen precision/quantization")
        if any(type(x) is not int or x < 1 for x in (self.min_pixels, self.max_pixels, self.max_length, self.max_new_tokens, self.r, self.alpha)):
            raise ValueError("Qwen lengths/pixels/rank must be positive integers")
        if self.max_pixels < self.min_pixels or not 0 <= self.dropout < 1:
            raise ValueError("invalid Qwen pixel range/LoRA dropout")
        if self.quantization == "nf4" and not self.device.startswith("cuda"):
            raise ValueError("NF4 path requires CUDA")


def load_qwen_config(path):
    return QwenConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def template_hash(processor):
    template = processor.chat_template
    if not isinstance(template, str) or not template:
        raise ValueError("a single explicit Qwen chat template is required")
    return hashlib.sha256(template.encode()).hexdigest()


def processor_signature(processor):
    tokenizer = json.loads(processor.tokenizer.backend_tokenizer.to_str())
    for runtime_field in ("padding", "truncation"):
        tokenizer.pop(runtime_field, None)
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    return {"chat_template_sha256": template_hash(processor),
            "tokenizer_sha256": digest(json.dumps(tokenizer, sort_keys=True)),
            "image_processor_class": type(processor.image_processor).__name__,
            "image_processor_sha256": digest(processor.image_processor.to_json_string())}


def load_qwen(config):
    require("transformers", "local Qwen3-VL")
    require("accelerate", "local Qwen3-VL")
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, BitsAndBytesConfig
    from .runtime import runtime
    device, _ = runtime(config.device, config.precision)
    kwargs = {"revision": config.revision, "local_files_only": config.local_files_only,
              "dtype": {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[config.precision],
              "attn_implementation": "sdpa"}
    if config.quantization == "nf4":
        require("bitsandbytes", "optional QLoRA")
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=kwargs["dtype"])
        kwargs["device_map"] = {"": str(device)}
    try:
        processor = AutoProcessor.from_pretrained(config.model_id, revision=config.processor_revision,
            local_files_only=config.local_files_only, min_pixels=config.min_pixels, max_pixels=config.max_pixels,
            use_fast=config.processor_use_fast)
        model = Qwen3VLForConditionalGeneration.from_pretrained(config.model_id, **kwargs)
    except OSError as error:
        raise RuntimeError("Qwen/Processor files unavailable at pinned revision; provision the local HF cache or model directory first") from error
    if config.quantization == "none":
        model.to(device)
    template_hash(processor)
    return model, processor


LANGUAGE_ATTENTION = re.compile(r"^model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")


def lora_targets(model):
    targets = [name for name, module in model.named_modules() if LANGUAGE_ATTENTION.fullmatch(name)
               and isinstance(module, torch.nn.Linear)]
    # bitsandbytes Linear4bit inherits Linear; retain the same path checks.
    if not targets or len(targets) % 4:
        raise ValueError("Qwen language attention paths not recognized; refusing broad LoRA injection")
    layers = {name.rsplit(".", 1)[0] for name in targets}
    if any(sum(t.startswith(layer + ".") for t in targets) != 4 for layer in layers):
        raise ValueError("incomplete Q/K/V/O language attention target set")
    return sorted(targets)


def inject_lora(model, config):
    require("peft", "Qwen LoRA")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    targets = lora_targets(model)
    model.requires_grad_(False)
    if config.quantization == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
    # Regex matches full paths, unlike PEFT's suffix-list matching.
    exact = "^(?:" + "|".join(re.escape(path) for path in targets) + ")$"
    model = get_peft_model(model, LoraConfig(r=config.r, lora_alpha=config.alpha, lora_dropout=config.dropout,
        target_modules=exact, bias="none", task_type="CAUSAL_LM"))
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = [{"name": name, "shape": list(p.shape), "parameters": p.numel()}
                 for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(".language_model.layers." not in p["name"] or ".self_attn." not in p["name"] or ".lora_" not in p["name"] for p in trainable):
        raise RuntimeError("LoRA scope verification failed")
    return model, {"targets": targets, "trainable": trainable,
                   "trainable_parameters": sum(p["parameters"] for p in trainable)}


def answer_labels(input_ids, attention_mask, prompt_ids, *, end_id, forbidden_ids=()):
    """Exact prefix boundary after image expansion; includes assistant im_end only."""
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape or len(prompt_ids) != len(input_ids):
        raise ValueError("Qwen mask dimensions disagree")
    labels = torch.full_like(input_ids, -100)
    for i, prefix in enumerate(prompt_ids):
        positions = attention_mask[i].nonzero().flatten()
        active = input_ids[i, positions]
        prefix = torch.as_tensor(prefix, device=active.device, dtype=active.dtype).flatten()
        start = len(prefix)
        if start >= len(active) or not torch.equal(active[:start], prefix):
            raise ValueError("Qwen full conversation differs from expanded generation prefix")
        ends = active[start:].eq(end_id).nonzero().flatten()
        if len(ends) != 1:
            raise ValueError("expected exactly one assistant ending token")
        stop = start + int(ends[0]) + 1
        supervised = active[start:stop]
        if len(supervised) < 2 or any(bool(supervised.eq(token).any()) for token in forbidden_ids):
            raise ValueError("empty or multimodal/control-token-contaminated assistant answer")
        labels[i, positions[start:stop]] = supervised
    return labels


class QwenCollator:
    def __init__(self, processor, max_length=4096):
        self.processor, self.max_length = processor, max_length

    def __call__(self, samples):
        conversations, prefixes = [], []
        tokenizer = self.processor.tokenizer
        for sample in samples:
            if validate_record(sample["target"]):
                raise ValueError("invalid supervised Qwen record")
            content = json.dumps(sample["target"], ensure_ascii=False)
            if any(token in content for token in tokenizer.all_special_tokens):
                raise ValueError("target contains a tokenizer control token")
            image = sample["image"]
            if not isinstance(image, Image.Image) or image.mode != "RGB":
                raise ValueError("Qwen collator requires RGB PIL images")
            prompt = self.processor.apply_chat_template(qwen_messages(image), tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors="pt")
            prefixes.append(prompt["input_ids"][0])
            conversations.append(qwen_messages(image, training_target=sample["target"]))
        batch = self.processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=False,
            padding=True, truncation=False, return_dict=True, return_tensors="pt")
        if int(batch["attention_mask"].sum(1).max()) > self.max_length:
            raise ValueError("Qwen sequence exceeds max_length; reduce image pixels or increase limit; JSON is never truncated")
        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        forbidden = [tokenizer.convert_tokens_to_ids(token) for token in
                     ("<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>", "<|im_start|>")]
        if end_id is None or end_id == tokenizer.unk_token_id:
            raise ValueError("Qwen assistant end token unavailable")
        batch["labels"] = answer_labels(batch["input_ids"], batch["attention_mask"], prefixes,
                                         end_id=end_id, forbidden_ids=[x for x in forbidden if x is not None])
        return dict(batch)


class QwenDataset(Dataset):
    def __init__(self, annotation, data_root, *, split=None, inference=False):
        self.rows, self.root, self.inference = read_jsonl(annotation), data_root, inference
        key = "id" if inference else "dish_id"
        if len({row[key] for row in self.rows}) != len(self.rows):
            raise ValueError("duplicate Qwen dataset IDs")
        if inference:
            if any(set(row) != {"id", "image"} for row in self.rows):
                raise ValueError("Qwen inference accepts only id/image")
        elif any(row.get("split") != split or validate_record(row["target"]) for row in self.rows):
            raise ValueError("Qwen split/Schema mismatch")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(safe_image_path(self.root, row["image"])) as image:
            result = {"id": row["id"] if self.inference else row["dish_id"], "image": image.convert("RGB")}
        if not self.inference:
            result["target"] = row["target"]
        return result


@torch.inference_mode()
def generate_qwen(model, processor, rgb_images, config, *, instruction=None):
    if not rgb_images or any(not isinstance(image, Image.Image) or image.mode != "RGB" for image in rgb_images):
        raise ValueError("Qwen generation accepts RGB images only")
    conversations = [qwen_messages(image) for image in rgb_images]
    if instruction is not None:
        for messages in conversations:
            messages[0]["content"][1]["text"] = instruction
    model.eval()
    original_padding = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        inputs = processor.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True,
            padding=True, truncation=False, return_dict=True, return_tensors="pt")
    finally:
        processor.tokenizer.padding_side = original_padding
    if int(inputs["attention_mask"].sum(1).max()) > config.max_length:
        raise ValueError("Qwen image/prompt exceeds max_length")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=config.max_new_tokens, do_sample=False, num_beams=1,
                            use_cache=True, eos_token_id=processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                            pad_token_id=processor.tokenizer.pad_token_id)
    tokens = output[:, inputs["input_ids"].shape[1]:]
    texts = processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    return [{"raw": raw, "raw_token_ids": ids, "eos": end in ids, "truncated": end not in ids}
            for raw, ids in zip(texts, tokens.tolist(), strict=True)]


def parse_qwen_prediction(prediction):
    try:
        record = strict_loads(prediction["raw"])
        errors = validate_record(record)
        return {**prediction, "status": "schema_failed" if errors else "ok", "schema_errors": errors}
    except (ValueError, TypeError):
        return {**prediction, "status": "parse_failed"}
