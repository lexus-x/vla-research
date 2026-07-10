"""
MultiRes-Action: Coarse-to-Fine Multi-Resolution Action Prediction
==================================================================
Addresses #2 problem in VLA: long-horizon task failure (5-15% gap)

Architecture:
- Coarse Planner: Low-frequency subgoal prediction (2-5Hz, 50-step horizon)
- Fine Controller: High-frequency motor commands (50Hz, 4-step refinement)
- Confidence Gate: Skip fine control when coarse plan is confident

Module size: <10M params (plug-in to any VLA backbone)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MultiResConfig:
    """Configuration for multi-resolution action head."""
    # Coarse planner
    coarse_horizon: int = 50          # Steps predicted by coarse planner
    coarse_frequency: float = 5.0     # Hz (re-plans every 200ms)
    coarse_hidden_dim: int = 256
    coarse_num_layers: int = 4
    coarse_flow_steps: int = 8        # Flow matching denoising steps

    # Fine controller
    fine_horizon: int = 4             # Steps predicted by fine controller
    fine_frequency: float = 50.0      # Hz (re-plans every 20ms)
    fine_hidden_dim: int = 128
    fine_num_layers: int = 2

    # Confidence gate
    confidence_threshold: float = 0.8  # Skip fine if coarse confidence > this
    action_dim: int = 7               # 7-DoF action space

    # VLA backbone interface
    backbone_dim: int = 768           # VLA feature dimension
    language_dim: int = 768           # Language instruction dimension


# ============================================================
# Module 1: Coarse Planner (Flow Matching)
# ============================================================

class FlowMatchingCoarsePlanner(nn.Module):
    """
    Flow matching action head for coarse subgoal prediction.
    Predicts 50-step action chunks at low frequency (5Hz).
    """

    def __init__(self, config: MultiResConfig):
        super().__init__()
        self.config = config
        d = config.coarse_hidden_dim

        # Input projection
        self.state_proj = nn.Linear(config.backbone_dim, d)
        self.language_proj = nn.Linear(config.language_dim, d)

        # Flow matching network (MLP-based)
        self.flow_net = nn.Sequential(
            nn.Linear(d + config.action_dim + 1, d * 2),  # +1 for time
            nn.SiLU(),
            nn.Linear(d * 2, d * 2),
            nn.SiLU(),
            nn.Linear(d * 2, d),
            nn.SiLU(),
            nn.Linear(d, config.action_dim * config.coarse_horizon),
        )

        # Confidence predictor
        self.confidence_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        language_features: torch.Tensor,
        num_steps: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict coarse action chunk via flow matching.

        Args:
            visual_features: (B, backbone_dim)
            language_features: (B, language_dim)
            num_steps: Number of flow matching denoising steps

        Returns:
            Dict with 'actions' (B, coarse_horizon, action_dim) and 'confidence' (B, 1)
        """
        B = visual_features.shape[0]
        cfg = self.config

        # Fuse visual + language
        h = self.state_proj(visual_features) + self.language_proj(language_features)

        # Predict confidence
        confidence = self.confidence_head(h)

        # Flow matching: transform noise → actions
        # Start from Gaussian noise
        x = torch.randn(B, cfg.coarse_horizon * cfg.action_dim, device=h.device)

        # Euler integration of flow
        dt = 1.0 / num_steps
        for t in torch.linspace(0, 1, num_steps):
            t_expanded = t.expand(B, 1).to(h.device)
            # Concatenate: features + current noisy actions + time
            flow_input = torch.cat([h, x, t_expanded], dim=-1)
            velocity = self.flow_net(flow_input)
            x = x + velocity * dt

        # Reshape to action chunk
        actions = x.view(B, cfg.coarse_horizon, cfg.action_dim)

        return {
            'actions': actions,
            'confidence': confidence,
        }


# ============================================================
# Module 2: Fine Controller (Fast MLP)
# ============================================================

class FastFineController(nn.Module):
    """
    Lightweight MLP for fine motor control.
    Takes current observation + coarse subgoal, predicts precise 4-step actions.
    Runs at 50Hz for reactive control.
    """

    def __init__(self, config: MultiResConfig):
        super().__init__()
        self.config = config
        d = config.fine_hidden_dim

        self.net = nn.Sequential(
            nn.Linear(config.backbone_dim + config.action_dim + 6, d),  # +6 for proprioception
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, config.action_dim * config.fine_horizon),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        coarse_subgoal: torch.Tensor,
        proprioception: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict fine motor commands.

        Args:
            visual_features: (B, backbone_dim)
            coarse_subgoal: (B, action_dim) — next subgoal from coarse planner
            proprioception: (B, 6) — current joint state

        Returns:
            actions: (B, fine_horizon, action_dim)
        """
        B = visual_features.shape[0]
        cfg = self.config

        # Concatenate all inputs
        x = torch.cat([visual_features, coarse_subgoal, proprioception], dim=-1)

        # Fast MLP forward
        out = self.net(x)
        actions = out.view(B, cfg.fine_horizon, cfg.action_dim)

        return actions


# ============================================================
# Module 3: Confidence Gate
# ============================================================

class ConfidenceGate(nn.Module):
    """
    Decides whether to use fine controller or skip to coarse plan.
    High confidence → use coarse plan directly (fast)
    Low confidence → refine with fine controller (precise)
    """

    def __init__(self, config: MultiResConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        coarse_actions: torch.Tensor,
        coarse_confidence: torch.Tensor,
        fine_actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Gate between coarse and fine actions.

        Returns:
            Dict with 'actions', 'used_fine' (bool mask), 'confidence'
        """
        cfg = self.config
        B = coarse_actions.shape[0]

        # Decision: use fine if confidence < threshold
        use_fine = coarse_confidence < cfg.confidence_threshold

        if fine_actions is not None and use_fine.any():
            # Interleave: use fine where needed, coarse elsewhere
            # For simplicity, use fine for all (in practice, batch-level gating)
            actions = fine_actions
            used_fine = True
        else:
            actions = coarse_actions
            used_fine = False

        return {
            'actions': actions,
            'used_fine': used_fine,
            'confidence': coarse_confidence,
        }


# ============================================================
# Module 4: MultiRes-Action Head (Complete)
# ============================================================

class MultiResActionHead(nn.Module):
    """
    Complete multi-resolution action head.
    Plug-in module (<10M params) for any VLA backbone.
    """

    def __init__(self, config: MultiResConfig = MultiResConfig()):
        super().__init__()
        self.config = config
        self.coarse_planner = FlowMatchingCoarsePlanner(config)
        self.fine_controller = FastFineController(config)
        self.gate = ConfidenceGate(config)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"MultiRes-Action Head: {total_params:,} params ({total_params/1e6:.1f}M)")

    def forward(
        self,
        visual_features: torch.Tensor,
        language_features: torch.Tensor,
        proprioception: torch.Tensor,
        use_fine: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: coarse → gate → fine (if needed).

        Args:
            visual_features: (B, backbone_dim) from VLA backbone
            language_features: (B, language_dim) from instruction encoder
            proprioception: (B, 6) current joint state
            use_fine: whether to use fine controller (False for speed)

        Returns:
            Dict with 'actions', 'confidence', 'used_fine'
        """
        # Step 1: Coarse planning
        coarse_out = self.coarse_planner(visual_features, language_features)
        coarse_actions = coarse_out['actions']
        confidence = coarse_out['confidence']

        # Step 2: Fine refinement (if needed)
        if use_fine:
            # Use first step of coarse plan as subgoal
            subgoal = coarse_actions[:, 0, :]  # (B, action_dim)
            fine_actions = self.fine_controller(visual_features, subgoal, proprioception)

            # Step 3: Gate
            result = self.gate(coarse_actions, confidence, fine_actions)
        else:
            result = {
                'actions': coarse_actions,
                'used_fine': False,
                'confidence': confidence,
            }

        return result


# ============================================================
# Training
# ============================================================

class MultiResTrainer:
    """Training wrapper for MultiRes-Action head."""

    def __init__(
        self,
        vla_backbone: nn.Module,
        action_head: MultiResActionHead,
        learning_rate: float = 1e-4,
        coarse_loss_weight: float = 1.0,
        fine_loss_weight: float = 0.5,
        confidence_loss_weight: float = 0.1,
    ):
        self.backbone = vla_backbone
        self.action_head = action_head
        self.optimizer = torch.optim.AdamW(
            list(vla_backbone.parameters()) + list(action_head.parameters()),
            lr=learning_rate,
        )
        self.coarse_w = coarse_loss_weight
        self.fine_w = fine_loss_weight
        self.conf_w = confidence_loss_weight

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.backbone.train()
        self.action_head.train()

        # Extract features from backbone
        features = self.backbone(batch['images'])

        # Forward through multi-res head
        output = self.action_head(
            visual_features=features,
            language_features=batch.get('language_features', features),
            proprioception=batch.get('proprioception', torch.zeros(features.shape[0], 6, device=features.device)),
        )

        # Coarse action loss (main loss)
        coarse_loss = F.mse_loss(output['actions'], batch['target_actions'])

        # Confidence loss (predict high confidence for easy tasks)
        if 'task_difficulty' in batch:
            # Easy tasks should have high confidence
            target_confidence = (batch['task_difficulty'] < 0.5).float()
            conf_loss = F.binary_cross_entropy(output['confidence'], target_confidence.unsqueeze(-1))
        else:
            conf_loss = torch.tensor(0.0)

        total_loss = self.coarse_w * coarse_loss + self.conf_w * conf_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.backbone.parameters()) + list(self.action_head.parameters()), 1.0
        )
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'coarse_loss': coarse_loss.item(),
            'confidence_loss': conf_loss.item(),
            'avg_confidence': output['confidence'].mean().item(),
        }


# ============================================================
# Integration Example
# ============================================================

if __name__ == "__main__":
    print("MultiRes-Action: Coarse-to-Fine Multi-Resolution Action Head")
    print("=" * 70)

    config = MultiResConfig()
    action_head = MultiResActionHead(config)

    # Simulate VLA backbone features
    B = 4
    visual_features = torch.randn(B, config.backbone_dim)
    language_features = torch.randn(B, config.language_dim)
    proprioception = torch.randn(B, 6)

    # Forward pass
    output = action_head(visual_features, language_features, proprioception)

    print(f"Coarse actions shape: {output['actions'].shape}")
    print(f"Confidence shape: {output['confidence'].shape}")
    print(f"Used fine: {output['used_fine']}")
    print(f"Confidence values: {output['confidence'].squeeze().tolist()}")
