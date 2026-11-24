"""Inference-only incremental execution using the original TransformerDecoder weights.

No modules/parameters are registered here. This preserves old model and optimizer
state_dict layouts and keeps teacher forcing on PyTorch's original implementation.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class LayerKV:
    memory_key: torch.Tensor
    memory_value: torch.Tensor
    self_key: torch.Tensor | None = None
    self_value: torch.Tensor | None = None


@dataclass
class DecoderCache:
    owner: int
    memory: torch.Tensor
    layers: list[LayerKV]
    valid_keys: torch.Tensor
    position: int = 0


def _heads(x, attention):
    batch, length, dimension = x.shape
    return x.reshape(batch, length, attention.num_heads, dimension // attention.num_heads).transpose(1, 2)


def _check_attention(attention):
    if not attention._qkv_same_embed_dim or attention.bias_k is not None or attention.bias_v is not None or attention.add_zero_attn:
        raise ValueError("KV cache supports the project's standard equal-dimension attention only")


def init_decoder_cache(model, memory):
    layers = []
    for layer in model.decoder.layers:
        for attention in (layer.self_attn, layer.multihead_attn):
            _check_attention(attention)
        attention = layer.multihead_attn
        dimension = attention.embed_dim
        bias = attention.in_proj_bias
        # Match MultiheadAttention's packed shared K/V projection operation.
        key, value = F.linear(memory, attention.in_proj_weight[dimension:],
                              bias[dimension:] if bias is not None else None).chunk(2, dim=-1)
        layers.append(LayerKV(_heads(key, attention), _heads(value, attention)))
    return DecoderCache(id(model), memory, layers,
                        torch.empty(memory.shape[0], 0, device=memory.device, dtype=torch.bool))


def _attend(attention, query, key, value, valid_keys=None):
    # MHA boolean mask: True=blocked; SDPA boolean mask: True=allowed.
    mask = valid_keys[:, None, None, :] if valid_keys is not None else None
    # Keys contain past/current tokens only. Non-square is_causal=True would
    # align the one-token query to the *first* key rather than the current step.
    states = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                            dropout_p=0.0, is_causal=False)
    states = states.transpose(1, 2).contiguous().flatten(2)
    return attention.out_proj(states)


def _self_attention(layer, x, kv, valid_keys):
    attention = layer.self_attn
    query, key, value = F.linear(x, attention.in_proj_weight, attention.in_proj_bias).chunk(3, dim=-1)
    query, key, value = [_heads(t, attention) for t in (query, key, value)]
    kv.self_key = key if kv.self_key is None else torch.cat((kv.self_key, key), dim=2)
    kv.self_value = value if kv.self_value is None else torch.cat((kv.self_value, value), dim=2)
    return layer.dropout1(_attend(attention, query, kv.self_key, kv.self_value, valid_keys))


def _cross_attention(layer, x, kv):
    attention = layer.multihead_attn
    dimension = attention.embed_dim
    bias = attention.in_proj_bias
    query = F.linear(x, attention.in_proj_weight[:dimension], bias[:dimension] if bias is not None else None)
    return layer.dropout2(_attend(attention, _heads(query, attention), kv.memory_key, kv.memory_value))


def _feedforward(layer, x):
    return layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(x)))))


def decoder_step(decoder, x, valid_token, cache):
    cache.valid_keys = torch.cat((cache.valid_keys, valid_token), dim=1)
    if not cache.valid_keys.any(dim=1).all():
        raise ValueError("incremental prefix must contain a non-PAD token")
    for layer, kv in zip(decoder.layers, cache.layers, strict=True):
        if layer.norm_first:
            x = x + _self_attention(layer, layer.norm1(x), kv, cache.valid_keys)
            x = x + _cross_attention(layer, layer.norm2(x), kv)
            x = x + _feedforward(layer, layer.norm3(x))
        else:
            x = layer.norm1(x + _self_attention(layer, x, kv, cache.valid_keys))
            x = layer.norm2(x + _cross_attention(layer, x, kv))
            x = layer.norm3(x + _feedforward(layer, x))
    cache.position += 1
    return decoder.norm(x) if decoder.norm is not None else x
