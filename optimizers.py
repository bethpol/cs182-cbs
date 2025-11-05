"""
Different optimizer configurations.
"""

import torch
import inspect

# Standard AdamW (default in train_cbs.py)
# =====================================================
def adamw(model, learning_rate=3e-4, weight_decay=0.1, beta1=0.9, beta2=0.95, device_type='cuda', ignore_fused=True):
    """Default AdamW configuration with weight decay groups."""

    # Use fused implementation if available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    if use_fused and not ignore_fused:
        extra_args = dict(fused=True)
        print(f"using fused AdamW: {use_fused}")
    else:
        extra_args = dict()
        print("Using AdamW")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        **extra_args
    )
    return optimizer


# Muon Optimizer
def muon(model, learning_rate=0.02, momentum=0.95, weight_decay=0.0, nesterov=True):
    """
    Muon optimizer - momentum-based optimizer with orthogonalization.

    Args:
        model: PyTorch model
        learning_rate: Learning rate (default: 0.02, typically higher than Adam)
        momentum: Momentum coefficient (default: 0.95)
        weight_decay: Weight decay coefficient (default: 0.0, often not needed)
        nesterov: Whether to use Nesterov momentum (default: True)
    """
    print(f"Using Muon optimizer")
    muon_params = []
    other_params = []

    for _, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim >= 2:
                muon_params.append(param)
            else:
                other_params.append(param)

    print(f"Using Muon optimizer")
    print(f"  - Muon (2D params): {len(muon_params)} parameters")
    print(f"  - other 1D params): {len(other_params)} parameters")

    # Muon for 2D parameters
    if len(muon_params) > 0:
        optimizer = torch.optim.Muon(
            muon_params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    return optimizer