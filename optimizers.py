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
def muon_w_adam(model, learning_rate=0.02, momentum=0.95, weight_decay=0.0, lr_adam=3e-4, wd_adam=0.1, beta1=0.9, beta2=0.95, nesterov=True):
    """
    Muon with AdamW optimizer - momentum-based optimizer with orthogonalization.

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
    print(f"  - AdamW (1D params): {len(other_params)} parameters")

    # optimizer list
    optimizers = []

    # Muon for 2D parameters
    if len(muon_params) > 0:
        muon_optimizer = torch.optim.Muon(
            muon_params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        optimizers.append(muon_optimizer)

    # Fallback optimizer: AdamW
    if len(other_params) > 0:
        adamw_optimizer = torch.optim.AdamW(
            other_params,
            lr=lr_adam,
            weight_decay=wd_adam,
            betas=(beta1, beta2)
        )
        optimizers.append(adamw_optimizer)

    return MultiOptimizer(optimizers)

class MultiOptimizer:
    """Wrapper to handle multiple optimizers as one."""

    def __init__(self, optimizers):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        """Combine param_groups from all optimizers.

        This property is essential for:
        1. Learning rate schedulers (they iterate over param_groups)
        2. Logging learning rates during training
        3. Any code that inspects optimizer settings

        Returns a combined list of all param_groups from both optimizers.
        """
        param_groups = []
        for optimizer in self.optimizers:
            param_groups.extend(optimizer.param_groups)
        return param_groups

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, state in zip(self.optimizers, state_dicts):
            opt.load_state_dict(state)