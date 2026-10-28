"""Multi-scale contrastive feature loss for CamoDiffusion without Gated OFM."""
from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn.functional as F

def contrastive_feature_loss(feats: Sequence[torch.Tensor], gt01: torch.Tensor,
                             log_snr: torch.Tensor, delta: float = 0.3,
                             w_boundary: float = 3.0) -> torch.Tensor:
    """Object/background feature-margin loss with boundary and timestep weighting."""
    device_type = gt01.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        gt = gt01.float()
        w_t = torch.sigmoid(-log_snr.float() / 4.0)
        total = gt.new_zeros(())
        for f in feats:
            f = F.normalize(f.float(), dim=1)
            m = F.adaptive_avg_pool2d(gt, f.shape[-2:])
            mb = 1.0 - m
            p_o = F.normalize((f * m).sum((2, 3)) / (m.sum((2, 3)) + 1e-6), dim=1)
            p_b = F.normalize((f * mb).sum((2, 3)) / (mb.sum((2, 3)) + 1e-6), dim=1)
            margin = (f * p_o[:, :, None, None]).sum(1) - (f * p_b[:, :, None, None]).sum(1)
            mm = m[:, 0]
            sign = (mm > 0.5).float() * 2.0 - 1.0
            weights = 1.0 + (w_boundary - 1.0) * ((mm > 0.05) & (mm < 0.95)).float()
            valid = ((mm > 0.5).flatten(1).any(1) & (mm <= 0.5).flatten(1).any(1)).float()
            per_img = (F.relu(delta - sign * margin) * weights).flatten(1).sum(1) / weights.flatten(1).sum(1)
            total = total + (per_img * w_t * valid).sum() / (valid.sum() + 1e-6)
        return total / max(1, len(feats))


