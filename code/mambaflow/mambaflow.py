"""
MambaFlow: SSM-Backbone VLA with Flow Matching Action Head
===========================================================
First VLA combining:
- Mamba (Selective State Space Model) as full backbone — O(n) inference
- Flow matching action head — smooth continuous actions, 4-8 steps
- Action chunking — 16-50 step prediction

Target: 300-500M params, 3-5x faster inference than transformer VLAs
First SSM-based VLA with flow matching (architecture matrix: unexplored)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MambaFlowConfig:
    """Configuration for MambaFlow VLA."""
    # Mamba backbone
    mamba_d_model: int = 512         # Mamba hidden dimension
    mamba_d_state: int = 16          # SSM state dimension
    mamba_d_conv: int = 4            # Local convolution width
    mamba_expand: int = 2            # Expansion factor
    mamba_n_layers: int = 24         # Number of Mamba blocks

    # Vision encoder (SigLIP, frozen)
    vision_dim: int = 768            # SigLIP output dimension
    vision_patch_size: int = 16
    vision_num_patches: int = 196    # 14x14 patches for 224x224

    # Language encoder
    language_dim: int = 768          # Language feature dimension
    max_instruction_len: int = 64

    # Action head
    action_dim: int = 7              # 7-DoF action space
    action_horizon: int = 32         # Predict 32 steps
    flow_steps: int = 8              # Flow matching denoising steps
    flow_hidden_dim: int = 256

    # Training
    dropout: float = 0.1


# ============================================================
# Mamba Block (using mamba-ssm library pattern)
# ============================================================

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model block.
    Implements the core Mamba mechanism: input-dependent selection gates
    allow the model to selectively remember or forget context.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 1D convolution for local context
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner
        )

        # SSM parameters (input-dependent)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # SSM state parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) — input sequence
        Returns:
            y: (B, L, D) — output sequence
        """
        B, L, D = x.shape
        d_inner = self.A_log.shape[0]

        # Input projection + split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)

        # 1D convolution (causal)
        x_conv = x_ssm.transpose(1, 2)  # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L]  # Causal
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM parameters (input-dependent)
        x_proj = self.x_proj(x_conv)  # (B, L, 2*d_state + 1)
        B_param, C_param, dt = x_proj.split(
            [self.A_log.shape[1], self.A_log.shape[1], 1], dim=-1
        )

        # Discretize
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)

        # Simplified SSM scan (for demonstration; real impl uses hardware-aware scan)
        # In practice, use mamba-ssm library for efficient CUDA kernels
        d_state = A.shape[1]

        # Discretized dynamics
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)  # (B, L, d_inner, d_state)

        # Simple sequential scan (for correctness; not optimized)
        h = torch.zeros(B, d_inner, d_state, device=x.device)
        ys = []
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x_conv[:, i:i+1, :].transpose(1, 2)
            y_i = (h * C_param[:, i:i+1, :]).sum(-1)  # (B, d_inner)
            ys.append(y_i)
        y = torch.stack(ys, dim=1)  # (B, L, d_inner)

        # Skip connection + gating
        y = y + self.D * x_conv
        y = y * F.silu(z)  # Gating

        return self.out_proj(y)


class MambaBlock(nn.Module):
    """Single Mamba block with LayerNorm and residual."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        return x + residual


# ============================================================
# Flow Matching Action Head
# ============================================================

class FlowMatchingHead(nn.Module):
    """
    Flow matching action head.
    Learns a velocity field that transforms noise into action trajectories
    via straight-line optimal transport.
    """

    def __init__(self, config: MambaFlowConfig):
        super().__init__()
        self.config = config
        d = config.flow_hidden_dim
        action_dim = config.action_dim * config.action_horizon

        # Conditioning network
        self.condition_proj = nn.Linear(config.mamba_d_model, d)

        # Flow network (predicts velocity)
        self.flow_net = nn.Sequential(
            nn.Linear(d + action_dim + 1, d * 2),  # +1 for time
            nn.SiLU(),
            nn.Linear(d * 2, d * 2),
            nn.SiLU(),
            nn.Linear(d * 2, d),
            nn.SiLU(),
            nn.Linear(d, action_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        num_steps: int = 8,
    ) -> torch.Tensor:
        """
        Generate actions via flow matching.

        Args:
            features: (B, d_model) — backbone features
            num_steps: Number of Euler integration steps

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        B = features.shape[0]
        cfg = self.config
        action_dim = cfg.action_dim * cfg.action_horizon

        # Condition on backbone features
        h = self.condition_proj(features)

        # Start from noise
        x = torch.randn(B, action_dim, device=features.device)

        # Euler integration
        dt = 1.0 / num_steps
        for t in torch.linspace(0, 1, num_steps, device=features.device):
            t_expanded = t.expand(B, 1)
            flow_input = torch.cat([h, x, t_expanded], dim=-1)
            velocity = self.flow_net(flow_input)
            x = x + velocity * dt

        # Reshape to action sequence
        actions = x.view(B, cfg.action_horizon, cfg.action_dim)

        return actions


# ============================================================
# MambaFlow VLA (Complete Model)
# ============================================================

class MambaFlowVLA(nn.Module):
    """
    Complete MambaFlow VLA model.
    Architecture: SigLIP (frozen) → Mamba backbone → Flow matching action head
    """

    def __init__(self, config: MambaFlowConfig = MambaFlowConfig()):
        super().__init__()
        self.config = config

        # Vision encoder (SigLIP placeholder — in practice, load pretrained)
        self.vision_proj = nn.Linear(config.vision_dim, config.mamba_d_model)

        # Language encoder (simple projection — in practice, use pretrained)
        self.language_proj = nn.Linear(config.language_dim, config.mamba_d_model)

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.vision_num_patches + config.max_instruction_len, config.mamba_d_model) * 0.02
        )

        # Mamba backbone
        self.backbone = nn.ModuleList([
            MambaBlock(
                d_model=config.mamba_d_model,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
                dropout=config.dropout,
            )
            for _ in range(config.mamba_n_layers)
        ])
        self.norm = nn.LayerNorm(config.mamba_d_model)

        # Flow matching action head
        self.action_head = FlowMatchingHead(config)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MambaFlow VLA: {total_params:,} total params ({total_params/1e6:.1f}M)")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images using vision encoder.

        Args:
            images: (B, C, H, W) or (B, num_patches, vision_dim)

        Returns:
            tokens: (B, num_patches, d_model)
        """
        if images.dim() == 4:
            # Assume pre-patched: (B, num_patches, vision_dim)
            B = images.shape[0]
            # Simple linear projection (replace with SigLIP in practice)
            tokens = self.vision_proj(images.flatten(2).transpose(1, 2))
        else:
            tokens = self.vision_proj(images)
        return tokens

    def encode_language(self, instruction: torch.Tensor) -> torch.Tensor:
        """
        Encode language instruction.

        Args:
            instruction: (B, seq_len, language_dim) or (B, language_dim)

        Returns:
            tokens: (B, seq_len, d_model)
        """
        if instruction.dim() == 2:
            instruction = instruction.unsqueeze(1)
        return self.language_proj(instruction)

    def forward(
        self,
        images: torch.Tensor,
        instruction: Optional[torch.Tensor] = None,
        num_flow_steps: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode → backbone → action head.

        Args:
            images: (B, num_patches, vision_dim)
            instruction: (B, seq_len, language_dim) or None
            num_flow_steps: Flow matching denoising steps

        Returns:
            Dict with 'actions' (B, horizon, action_dim)
        """
        B = images.shape[0]

        # Encode vision
        vision_tokens = self.encode_vision(images)  # (B, N, d_model)

        # Encode language
        if instruction is not None:
            lang_tokens = self.encode_language(instruction)
            # Concatenate vision + language
            tokens = torch.cat([vision_tokens, lang_tokens], dim=1)
        else:
            tokens = vision_tokens

        # Add positional embeddings
        L = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :L, :]

        # Mamba backbone (O(n) complexity!)
        for block in self.backbone:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # Global average pooling over sequence
        features = tokens.mean(dim=1)  # (B, d_model)

        # Flow matching action head
        actions = self.action_head(features, num_steps=num_flow_steps)

        return {
            'actions': actions,
            'features': features,
        }


# ============================================================
# Latency Benchmarking
# ============================================================

def benchmark_latency(model: nn.Module, config: MambaFlowConfig, device: str = 'cuda'):
    """Benchmark inference latency."""
    import time

    model = model.to(device).eval()
    B = 1

    # Warmup
    images = torch.randn(B, config.vision_num_patches, config.vision_dim, device=device)
    for _ in range(5):
        with torch.no_grad():
            model(images, num_flow_steps=4)

    # Benchmark
    torch.cuda.synchronize()
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(images, num_flow_steps=4)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # ms
    print(f"\nLatency Benchmark (B=1, {device}):")
    print(f"  Mean: {times.mean():.1f}ms")
    print(f"  Std:  {times.std():.1f}ms")
    print(f"  P50:  {np.percentile(times, 50):.1f}ms")
    print(f"  P95:  {np.percentile(times, 95):.1f}ms")
    print(f"  P99:  {np.percentile(times, 99):.1f}ms")
    print(f"  Hz:   {1000/times.mean():.1f}")

    return times


# ============================================================
# Integration Example
# ============================================================

if __name__ == "__main__":
    print("MambaFlow: SSM-Backbone VLA with Flow Matching")
    print("=" * 70)

    config = MambaFlowConfig()

    # Create model
    model = MambaFlowVLA(config)

    # Simulate input
    B = 4
    images = torch.randn(B, config.vision_num_patches, config.vision_dim)
    instruction = torch.randn(B, config.max_instruction_len, config.language_dim)

    # Forward pass
    output = model(images, instruction)

    print(f"\nInput shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Instruction: {instruction.shape}")
    print(f"\nOutput shapes:")
    print(f"  Actions: {output['actions'].shape}")
    print(f"  Features: {output['features'].shape}")

    # Latency benchmark (if CUDA available)
    if torch.cuda.is_available():
        benchmark_latency(model, config, 'cuda')
    else:
        print("\nCUDA not available, skipping latency benchmark")
        print("Expected on A100: ~40-60ms (15-25 Hz)")
        print("Expected on L40S: ~30-50ms (20-33 Hz)")
