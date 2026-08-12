"""The MLP architecture, mirrored from the Colab training notebook
(ml/notebooks/01_credit_card_default_training.ipynb, Section 4.1's `CreditDefaultMLP`).

This has to be byte-for-byte structurally identical to the training-time class —
`state_dict()` is just a flat dict of tensor names -> weights, with no record of
*how* the layers were assembled. If this class's `nn.Sequential` doesn't produce
the exact same layer order/shapes as training did, `load_state_dict` will either
raise a shape-mismatch error (the safe failure) or, in unlucky cases, silently
load weights into the wrong layers (the dangerous one). Keep this in lockstep
with the notebook if the architecture ever changes.

Deliberately NOT mirrored: the training-time `_init_weights` scheme (Xavier/He/
zero). Initialization only matters before training starts — this module always
follows a fresh construction with `load_state_dict`, which overwrites whatever
the constructor initialized. Carrying that logic into the serving path would be
dead code with a chance to silently do the wrong thing (e.g. skip re-init and
accidentally serve random untrained weights) for zero benefit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CreditDefaultMLP(nn.Module):
    """Must match ml/artifacts/model/model_config.json exactly: input_dim,
    hidden_dims, dropout, use_batchnorm are all read from that file at load
    time (see app/inference.py) — never hardcoded here."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        # Single output logit — sigmoid is applied at inference time in
        # inference.py, never inside the model (matches the training notebook;
        # BCEWithLogitsLoss expects raw logits during training).
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
