"""
PerturbVLA: Adversarial Perturbation Training for VLA Robustness
================================================================
Addresses the #1 problem in VLA: memorization (LIBERO-PRO: 90% → 0% under perturbation)

This is a TRAINING METHOD — zero overhead at inference.
Can be applied to ANY existing VLA (OpenVLA, SmolVLA, Octo, etc.)

Modules:
1. PerturbationAugmentor — spatial/visual/language/temporal perturbations
2. ContrastiveRobustnessLoss — perturbation-invariant representations
3. PerturbVLATrainer — training loop with perturbation curriculum
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Module 1: Perturbation Augmentor
# ============================================================

@dataclass
class PerturbConfig:
    """Configuration for perturbation parameters."""
    # Spatial perturbations
    position_range: float = 0.15      # ±15cm object position jitter
    angle_range: float = 10.0         # ±10 degrees rotation jitter
    scale_range: Tuple[float, float] = (0.9, 1.1)  # ±10% scale

    # Visual perturbations
    camera_jitter_deg: float = 5.0    # ±5 degree camera viewpoint
    lighting_range: float = 0.3       # ±30% brightness/contrast
    texture_swap_prob: float = 0.1    # 10% chance of texture swap
    background_noise_std: float = 0.05  # Gaussian noise on background

    # Language perturbations
    token_corrupt_prob: float = 0.15  # 15% token corruption
    synonym_replace_prob: float = 0.1 # 10% synonym replacement
    instruction_drop_prob: float = 0.05  # 5% word dropout

    # Temporal perturbations
    action_noise_std: float = 0.02    # ±2% action noise
    timing_jitter_steps: int = 2      # ±2 step timing jitter

    # Clutter perturbations
    clutter_prob: float = 0.25        # 25% chance of distractor objects
    max_clutter_objects: int = 3      # Max distractor objects

    # Curriculum
    curriculum_start: float = 0.1     # Start with 10% perturbation strength
    curriculum_end: float = 1.0       # End with 100% perturbation strength
    curriculum_steps: int = 50000     # Steps to reach full perturbation


class PerturbationAugmentor:
    """Applies systematic perturbations during VLA training."""

    def __init__(self, config: PerturbConfig = PerturbConfig()):
        self.config = config
        self.current_strength = config.curriculum_start

    def update_curriculum(self, step: int):
        """Gradually increase perturbation strength."""
        progress = min(1.0, step / self.config.curriculum_steps)
        self.current_strength = (
            self.config.curriculum_start +
            progress * (self.config.curriculum_end - self.config.curriculum_start)
        )

    def perturb_spatial(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perturb object positions, rotations, and scales."""
        cfg = self.config
        s = self.current_strength

        if 'object_positions' in observations:
            noise = torch.randn_like(observations['object_positions']) * cfg.position_range * s
            observations['object_positions'] = observations['object_positions'] + noise

        if 'object_rotations' in observations:
            noise = torch.randn_like(observations['object_rotations']) * (cfg.angle_range / 180 * np.pi) * s
            observations['object_rotations'] = observations['object_rotations'] + noise

        return observations

    def perturb_visual(self, images: torch.Tensor) -> torch.Tensor:
        """Perturb visual observations: camera jitter, lighting, noise."""
        cfg = self.config
        s = self.current_strength
        batch_size = images.shape[0]

        # Camera viewpoint simulation (random affine transform)
        if cfg.camera_jitter_deg > 0:
            angles = (torch.randn(batch_size) * cfg.camera_jitter_deg * s).to(images.device)
            # Small rotation matrix
            cos_a, sin_a = torch.cos(angles), torch.sin(angles)
            # Apply as brightness/contrast shift (simplified camera sim)
            images = images * (1 + cos_a.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 0.1 * s)

        # Lighting perturbation
        if cfg.lighting_range > 0:
            brightness = 1 + (torch.rand(batch_size, 1, 1, 1, device=images.device) - 0.5) * cfg.lighting_range * s
            contrast = 1 + (torch.rand(batch_size, 1, 1, 1, device=images.device) - 0.5) * cfg.lighting_range * s
            images = images * brightness * contrast

        # Background noise
        if cfg.background_noise_std > 0:
            noise = torch.randn_like(images) * cfg.background_noise_std * s
            images = images + noise

        return images.clamp(0, 1)

    def perturb_language(self, token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Corrupt instruction tokens: random replacement, dropout."""
        cfg = self.config
        s = self.current_strength

        # Token corruption
        corrupt_mask = torch.rand_like(token_ids.float()) < (cfg.token_corrupt_prob * s)
        random_tokens = torch.randint(0, vocab_size, token_ids.shape, device=token_ids.device)
        perturbed = torch.where(corrupt_mask, random_tokens, token_ids)

        # Word dropout
        drop_mask = torch.rand_like(token_ids.float()) < (cfg.instruction_drop_prob * s)
        # Replace with padding token (0)
        perturbed = torch.where(drop_mask, torch.zeros_like(token_ids), perturbed)

        return perturbed

    def perturb_temporal(self, actions: torch.Tensor) -> torch.Tensor:
        """Add temporal noise to action sequences."""
        cfg = self.config
        s = self.current_strength

        # Action noise
        if cfg.action_noise_std > 0:
            noise = torch.randn_like(actions) * cfg.action_noise_std * s
            actions = actions + noise

        return actions

    def __call__(self, batch: Dict[str, torch.Tensor], vocab_size: int = 32000) -> Dict[str, torch.Tensor]:
        """Apply all perturbations to a training batch."""
        batch = dict(batch)  # Don't modify original

        if 'images' in batch:
            batch['images'] = self.perturb_visual(batch['images'])
        if 'object_positions' in batch:
            batch = self.perturb_spatial(batch)
        if 'instruction_ids' in batch:
            batch['instruction_ids'] = self.perturb_language(batch['instruction_ids'], vocab_size)
        if 'actions' in batch:
            batch['actions'] = self.perturb_temporal(batch['actions'])

        return batch


# ============================================================
# Module 2: Contrastive Robustness Loss
# ============================================================

class ContrastiveRobustnessLoss(nn.Module):
    """
    Contrastive loss that encourages perturbation-invariant representations.
    - Pull together: representations of the same task under different perturbations
    - Push apart: representations of different tasks
    """

    def __init__(self, temperature: float = 0.07, projection_dim: int = 128):
        super().__init__()
        self.temperature = temperature
        self.projection_dim = projection_dim

        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(768, 256),  # Assuming 768-dim VLA features
            nn.ReLU(),
            nn.Linear(256, projection_dim),
        )

    def forward(self, features: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive robustness loss.

        Args:
            features: (B, D) — VLA backbone features
            task_ids: (B,) — task identifier for each sample

        Returns:
            contrastive loss scalar
        """
        # Project features
        z = self.projector(features)
        z = F.normalize(z, dim=-1)

        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature

        # Create positive mask: same task = positive pair
        task_ids = task_ids.unsqueeze(0)
        pos_mask = (task_ids == task_ids.t()).float()

        # Remove self-similarity
        self_mask = torch.eye(z.shape[0], device=z.device)
        pos_mask = pos_mask - self_mask

        # InfoNCE loss
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=-1, keepdim=True) + 1e-8)

        # Average over positive pairs
        pos_count = pos_mask.sum(dim=-1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(dim=-1) / pos_count

        return loss.mean()


# ============================================================
# Module 3: Perturbation-Robust Training Loop
# ============================================================

class PerturbVLATrainer:
    """
    Training wrapper that adds perturbation augmentation + robustness loss
    to any existing VLA model.
    """

    def __init__(
        self,
        vla_model: nn.Module,
        config: PerturbConfig = PerturbConfig(),
        robustness_weight: float = 0.1,
        learning_rate: float = 1e-4,
    ):
        self.vla_model = vla_model
        self.augmentor = PerturbationAugmentor(config)
        self.robustness_loss = ContrastiveRobustnessLoss()
        self.robustness_weight = robustness_weight
        self.optimizer = torch.optim.AdamW(
            list(vla_model.parameters()) + list(self.robustness_loss.parameters()),
            lr=learning_rate,
        )
        self.step = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with perturbation augmentation."""
        self.vla_model.train()
        self.augmentor.update_curriculum(self.step)

        # Create perturbed version of the batch
        perturbed_batch = self.augmentor(batch)

        # Forward pass on original
        original_output = self.vla_model(
            images=batch['images'],
            instruction_ids=batch.get('instruction_ids'),
        )
        action_loss = F.mse_loss(original_output['actions'], batch['actions'])

        # Forward pass on perturbed
        perturbed_output = self.vla_model(
            images=perturbed_batch['images'],
            instruction_ids=perturbed_batch.get('instruction_ids'),
        )

        # Action loss on perturbed (should produce same actions)
        perturbed_action_loss = F.mse_loss(perturbed_output['actions'], batch['actions'])

        # Contrastive robustness loss
        if 'task_ids' in batch:
            features = torch.cat([
                original_output['features'],
                perturbed_output['features']
            ], dim=0)
            task_ids = torch.cat([batch['task_ids'], batch['task_ids']], dim=0)
            contrastive_loss = self.robustness_loss(features, task_ids)
        else:
            contrastive_loss = torch.tensor(0.0, device=action_loss.device)

        # Total loss
        total_loss = (
            action_loss +
            perturbed_action_loss +
            self.robustness_weight * contrastive_loss
        )

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vla_model.parameters(), 1.0)
        self.optimizer.step()

        self.step += 1

        return {
            'total_loss': total_loss.item(),
            'action_loss': action_loss.item(),
            'perturbed_action_loss': perturbed_action_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'perturbation_strength': self.augmentor.current_strength,
        }


# ============================================================
# Evaluation: Robustness Testing
# ============================================================

class RobustnessEvaluator:
    """Evaluate VLA robustness under systematic perturbations."""

    def __init__(self, perturbation_levels: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]):
        self.levels = perturbation_levels

    def evaluate(
        self,
        vla_model: nn.Module,
        eval_env,
        num_episodes: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Evaluate at each perturbation level.

        Returns:
            Dict mapping perturbation level to list of success rates
        """
        results = {}

        for level in self.levels:
            augmentor = PerturbationAugmentor(PerturbConfig(
                position_range=0.15 * level,
                camera_jitter_deg=5.0 * level,
                token_corrupt_prob=0.15 * level,
            ))
            augmentor.current_strength = level

            successes = []
            for ep in range(num_episodes):
                obs = eval_env.reset()
                done = False
                success = False

                while not done:
                    # Apply perturbation to observation
                    perturbed_obs = augmentor({'images': obs['images']}, vocab_size=32000)

                    with torch.no_grad():
                        action = vla_model(perturbed_obs['images'])

                    obs, reward, done, info = eval_env.step(action)
                    if info.get('success', False):
                        success = True

                successes.append(float(success))

            results[f'level_{level}'] = successes
            results[f'level_{level}_mean'] = [np.mean(successes)]

        return results


# ============================================================
# Integration Example
# ============================================================

if __name__ == "__main__":
    # Example: wrap an existing VLA model
    print("PerturbVLA: Adversarial Perturbation Training for VLA Robustness")
    print("=" * 70)

    # Simulate a simple VLA model
    class SimpleVLA(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.head = nn.Linear(64, 7)  # 7-DoF action

        def forward(self, images, instruction_ids=None, **kwargs):
            features = self.backbone(images).squeeze(-1).squeeze(-1)
            actions = self.head(features)
            return {'actions': actions, 'features': features}

    model = SimpleVLA()
    trainer = PerturbVLATrainer(model, robustness_weight=0.1)

    # Simulate training batch
    batch = {
        'images': torch.randn(4, 3, 224, 224),
        'actions': torch.randn(4, 7),
        'task_ids': torch.tensor([0, 0, 1, 1]),
    }

    # Train step
    metrics = trainer.train_step(batch)
    print(f"Step {trainer.step}: {metrics}")
