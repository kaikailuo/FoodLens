import torch
from torch.nn import functional as F


def token_loss(logits, targets):
    count = targets.ne(0).sum().clamp_min(1)
    return F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=0, reduction="sum") / count


def hybrid_loss(output, batch, statistics, lambdas=(1.0, 1.0, 1.0)):
    sequence = token_loss(output["logits"], batch["targets"])
    mask = batch["item_mask"]
    if mask.any():
        weights = F.smooth_l1_loss((output["weights"][mask] - statistics.weight_mean) / statistics.weight_std,
            (batch["weights"][mask] - statistics.weight_mean) / statistics.weight_std)
    else:
        weights = output["weights"].sum() * 0
    means = output["totals"].new_tensor(statistics.total_mean)
    scales = output["totals"].new_tensor(statistics.total_std).clamp_min(1e-6)
    totals = F.smooth_l1_loss((output["totals"] - means) / scales, (batch["totals"] - means) / scales)
    return {"loss": lambdas[0]*sequence + lambdas[1]*weights + lambdas[2]*totals,
            "sequence": sequence, "weight": weights, "total": totals}
