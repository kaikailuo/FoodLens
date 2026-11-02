"""Executable A/B/B-global/Hybrid baselines; tiny backbone is smoke-only."""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights


@dataclass
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = False  # Explicit opt-in only; tests never download weights.
    hidden: int = 256
    embedding: int = 256
    heads: int = 4
    layers: int = 2
    feedforward: int = 1024
    dropout: float = 0.1
    frozen_backbone: bool = True


class VisualEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.frozen = config.frozen_backbone
        if config.backbone == "resnet18":
            network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if config.pretrained else None)
            self.features = nn.Sequential(*list(network.children())[:-2])
        elif config.backbone == "tiny":
            self.features = nn.Sequential(nn.Conv2d(3, 16, 3, stride=4, padding=1), nn.ReLU(),
                                          nn.AdaptiveAvgPool2d((7, 7)), nn.Conv2d(16, 512, 1))
        else:
            raise ValueError(config.backbone)
        if self.frozen:
            self.features.requires_grad_(False)
        self.projection = nn.Linear(512, config.hidden)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.features.eval()  # Keeps frozen BatchNorm running statistics fixed.
        return self

    def unfreeze_layer4(self):
        if len(self.features) != 8:
            raise ValueError("layer4 fine-tuning requires ResNet18")
        self.features[-1].requires_grad_(True)
        # Keep backbone eval mode deliberately: fine-tune weights with fixed BN.

    def forward(self, images):
        grid = self.features(images)
        v = grid.mean((-2, -1))
        z = self.projection(v).tanh()
        return grid, v, z


class CaptionA(nn.Module):
    def __init__(self, vocabulary_size, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.visual = VisualEncoder(self.config)
        self.embedding = nn.Embedding(vocabulary_size, self.config.embedding, padding_idx=0)
        self.gru = nn.GRU(self.config.embedding + self.config.hidden, self.config.hidden, batch_first=True)
        self.output = nn.Linear(self.config.hidden, vocabulary_size)

    def decode(self, inputs, z, initial=None):
        embedded = self.embedding(inputs)
        conditions = z.unsqueeze(1).expand(-1, inputs.shape[1], -1)
        states, hidden = self.gru(torch.cat((embedded, conditions), -1), z.unsqueeze(0) if initial is None else initial)
        return self.output(states), states, hidden

    def forward(self, images, inputs):
        _, _, z = self.visual(images)
        return self.decode(inputs, z)[0]


def sinusoidal(length, dimension, *, device, dtype):
    if dimension % 2:
        raise ValueError("sinusoidal dimension must be even")
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    scale = torch.exp(torch.arange(0, dimension, 2, device=device, dtype=dtype) * (-math.log(10000) / dimension))
    result = torch.zeros(length, dimension, device=device, dtype=dtype)
    result[:, 0::2], result[:, 1::2] = (position * scale).sin(), (position * scale).cos()
    return result


def grid_positions(height, width, dimension, *, device, dtype):
    if dimension % 4:
        raise ValueError("2D positional dimension must be divisible by 4")
    rows = sinusoidal(height, dimension // 2, device=device, dtype=dtype)
    columns = sinusoidal(width, dimension // 2, device=device, dtype=dtype)
    return torch.cat((rows[:, None].expand(height, width, -1), columns[None].expand(height, width, -1)), -1).reshape(-1, dimension)


def causal_mask(length, device=None):
    # True means blocked in nn.Transformer, unlike scaled_dot_product_attention.
    return torch.ones(length, length, dtype=torch.bool, device=device).triu(1)


class CaptionB(nn.Module):
    def __init__(self, vocabulary_size, config=None, *, global_only=False):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.global_only = global_only
        self.visual = VisualEncoder(c)
        self.grid_projection = nn.Linear(512, c.hidden)
        self.embedding = nn.Embedding(vocabulary_size, c.hidden, padding_idx=0)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(c.hidden, c.heads, c.feedforward,
            c.dropout, batch_first=True), c.layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(c.hidden, c.heads, c.feedforward,
            c.dropout, batch_first=True), c.layers)
        self.output = nn.Linear(c.hidden, vocabulary_size)
        # PyTorch stacks deep-copy layers; initialize independently.
        for stack in (self.encoder, self.decoder):
            for layer in stack.layers:
                for parameter in layer.parameters():
                    if parameter.dim() > 1:
                        nn.init.xavier_uniform_(parameter)

    def memory(self, images):
        grid, v, _ = self.visual(images)
        if self.global_only:
            tokens = self.grid_projection(v).unsqueeze(1)
        else:
            tokens = self.grid_projection(grid.flatten(2).transpose(1, 2))
            tokens = tokens + grid_positions(*grid.shape[-2:], self.config.hidden, device=grid.device, dtype=grid.dtype)
        return self.encoder(tokens)

    def decode(self, inputs, memory):
        tokens = self.embedding(inputs) * math.sqrt(self.config.hidden)
        tokens = tokens + sinusoidal(inputs.shape[1], self.config.hidden, device=tokens.device, dtype=tokens.dtype)
        states = self.decoder(tokens, memory, tgt_mask=causal_mask(inputs.shape[1], inputs.device),
                              tgt_key_padding_mask=inputs.eq(0))
        return self.output(states)

    def forward(self, images, inputs):
        return self.decode(inputs, self.memory(images))


class Hybrid(CaptionA):
    def __init__(self, vocabulary_size, config=None):
        super().__init__(vocabulary_size, config)
        h = self.config.hidden
        self.weight_head = nn.Sequential(nn.Linear(h * 2, h), nn.ReLU(), nn.Linear(h, 1), nn.Softplus())
        self.total_head = nn.Sequential(nn.Linear(512 + h, h), nn.ReLU(), nn.Linear(h, 5), nn.Softplus())

    def initialize_regression(self, statistics):
        # inverse softplus is approximately identity above 20 and avoids exp overflow.
        def inverse(values):
            x = torch.as_tensor(values).clamp_min(1e-4)
            return x + torch.log(-torch.expm1(-x))
        with torch.no_grad():
            self.weight_head[-2].bias.copy_(inverse([statistics.weight_mean]))
            self.total_head[-2].bias.copy_(inverse(statistics.total_mean))

    def forward(self, images, inputs, boundaries):
        _, v, z = self.visual(images)
        logits, states, _ = self.decode(inputs, z)
        selected = states.gather(1, boundaries.unsqueeze(-1).expand(-1, -1, states.shape[-1]))
        conditions = z.unsqueeze(1).expand(-1, boundaries.shape[1], -1)
        return {"logits": logits, "states": states,
                "weights": self.weight_head(torch.cat((selected, conditions), -1)).squeeze(-1),
                "totals": self.total_head(torch.cat((v, z), -1))}


def transfer_caption_a(source, source_vocab, target, target_vocab):
    if target_vocab.tokens[:len(source_vocab)] != source_vocab.tokens:
        raise ValueError("task2 vocabulary must retain every task1 ID")
    if asdict(source.config) != asdict(target.config):
        raise ValueError("transfer requires identical encoder/GRU configurations")
    target.visual.load_state_dict(source.visual.state_dict())
    target.gru.load_state_dict(source.gru.state_dict())
    with torch.no_grad():
        for token, old_id in source_vocab.index.items():
            new_id = target_vocab.index[token]
            target.embedding.weight[new_id].copy_(source.embedding.weight[old_id])
            target.output.weight[new_id].copy_(source.output.weight[old_id])
            target.output.bias[new_id].copy_(source.output.bias[old_id])
    return {"copied_tokens": len(source_vocab), "new_tokens": len(target_vocab) - len(source_vocab),
            "modules": ["visual", "gru", "embedding rows", "output rows"]}
