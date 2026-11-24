"""Differentiable categorical generation and self-critical policy gradient."""
import torch
from .text import detokenize


def eos_mask(tokens, eos=2, pad=0):
    previous_eos = tokens.eq(eos).cumsum(1) - tokens.eq(eos).long()
    return previous_eos.eq(0) & tokens.ne(pad)


def policy_loss(log_probabilities, tokens, sampled_rewards, baseline_rewards):
    if log_probabilities.shape != tokens.shape:
        raise ValueError("sequence log-probability shape mismatch")
    advantage = (sampled_rewards - baseline_rewards).detach()
    return -(advantage * (log_probabilities * eos_mask(tokens)).sum(1)).mean()


def generate_policy(model, images, *, max_tokens, sample, gru_cache_enabled=False):
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if type(gru_cache_enabled) is not bool:
        raise ValueError("gru_cache_enabled must be boolean")
    prefix = torch.ones(len(images), 1, device=images.device, dtype=torch.long)
    finished = torch.zeros(len(images), device=images.device, dtype=torch.bool)
    memory = model.memory(images) if hasattr(model, "memory") else model.visual(images)[2]
    hidden = None
    probabilities, generated = [], []
    for _ in range(max_tokens):
        if gru_cache_enabled and not hasattr(model, "memory"):
            # Retain autograd through hidden state for the sampled policy. Do not
            # call inference_mode helpers or detach recurrent state here.
            logits, _, hidden = model.decode(prefix[:, -1:], memory, hidden)
        else:
            logits = model.decode(prefix, memory)
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits[:, -1].float()
        logits = logits.masked_fill(torch.arange(logits.shape[-1], device=images.device)[None, :] < 2, -torch.inf)
        if not torch.isfinite(logits[:, 2:]).all():
            raise RuntimeError("nonfinite generation logits")
        distribution = torch.distributions.Categorical(logits=logits)
        token = distribution.sample() if sample else logits.argmax(-1)
        logp = distribution.log_prob(token).masked_fill(finished, 0)
        token = token.masked_fill(finished, 0)
        probabilities.append(logp)
        generated.append(token)
        prefix = torch.cat((prefix, token[:, None]), 1)
        finished = finished | token.eq(2)
        if bool(finished.all()):
            break
    return torch.stack(generated, 1), torch.stack(probabilities, 1)


def captions_from_tokens(tokens, vocab):
    captions = []
    for row in tokens.tolist():
        content = []
        for token in row:
            if token == 2:
                break
            if token not in (0, 1):
                content.append(vocab.tokens[token])
        captions.append(detokenize(content))
    return captions


def scst_loss(model, images, sample_ids, reward, *, vocab, max_tokens=64, gru_cache_enabled=False):
    original_mode = model.training
    # Disable dropout in both branches while retaining gradients in sampled branch.
    model.eval()
    try:
        with torch.no_grad():
            greedy, _ = generate_policy(model, images, max_tokens=max_tokens, sample=False, gru_cache_enabled=gru_cache_enabled)
        sampled, logp = generate_policy(model, images, max_tokens=max_tokens, sample=True, gru_cache_enabled=gru_cache_enabled)
        sampled_text = captions_from_tokens(sampled, vocab)
        greedy_text = captions_from_tokens(greedy, vocab)
        rewards = logp.new_tensor(reward(sample_ids, sampled_text))
        baselines = logp.new_tensor(reward(sample_ids, greedy_text))
        loss = policy_loss(logp, sampled, rewards, baselines)
        diagnostics = [{"id": sample_id, "sample": s, "greedy": g, "reward": float(r),
                        "baseline_reward": float(b), "advantage": float(r-b),
                        "sample_eos": bool(2 in ids), "sample_length": int(mask.sum())}
                       for sample_id, s, g, r, b, ids, mask in zip(sample_ids, sampled_text, greedy_text,
                           rewards, baselines, sampled.tolist(), eos_mask(sampled), strict=True)]
        return loss, diagnostics
    finally:
        model.train(original_mode)
