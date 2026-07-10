"""
PR-VLA: Perturbation-Robust Vision-Language-Action Model
Addresses the memorization problem exposed by LIBERO-PRO.

Core idea: Adversarial perturbation training + contrastive robustness loss
forces the model to learn task understanding, not trajectory memorization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class PerturbationAugmentor:
    """Systematic perturbation augmentation for VLA training."""
    
    def __init__(
        self,
        position_range: float = 0.15,  # ±15cm
        angle_range: float = 10.0,  # ±10 degrees
        clutter_prob: float = 0.3,
        instruction_corrupt_prob: float = 0.2,
        object_swap_prob: float = 0.2,
    ):
        self.position_range = position_range
        self.angle_range = angle_range
        self.clutter_prob = clutter_prob
        self.instruction_corrupt_prob = instruction_corrupt_prob
        self.object_swap_prob = object_swap_prob
    
    def perturb_position(self, positions: torch.Tensor) -> torch.Tensor:
        """Randomize object positions."""
        noise = torch.randn_like(positions) * self.position_range
        return positions + noise
    
    def perturb_camera(self, camera_params: torch.Tensor) -> torch.Tensor:
        """Perturb camera viewpoint."""
        noise = torch.randn_like(camera_params) * (self.angle_range / 180.0 * np.pi)
        return camera_params + noise
    
    def corrupt_instruction(self, tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Corrupt instruction tokens with noise."""
        mask = torch.rand(tokens.shape) < self.instruction_corrupt_prob
        random_tokens = torch.randint(0, vocab_size, tokens.shape)
        return torch.where(mask, random_tokens, tokens)
    
    def inject_clutter(self, scene: torch.Tensor) -> torch.Tensor:
        """Add distractor objects to scene."""
        if torch.rand(1) > self.clutter_prob:
            return scene
        # Add random noise patches as "clutter"
        clutter = torch.randn_like(scene) * 0.1
        mask = torch.rand_like(scene) < 0.05
        return torch.where(mask, scene + clutter, scene)
    
    def augment(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply all perturbations to a batch."""
        augmented = batch.copy()
        
        if 'positions' in batch:
            augmented['positions'] = self.perturb_position(batch['positions'])
        
        if 'camera_params' in batch:
            augmented['camera_params'] = self.perturb_camera(batch['camera_params'])
        
        if 'instruction_tokens' in batch:
            augmented['instruction_tokens'] = self.corrupt_instruction(
                batch['instruction_tokens'], batch.get('vocab_size', 32000)
            )
        
        if 'observation' in batch:
            augmented['observation'] = self.inject_clutter(batch['observation'])
        
        return augmented


class ContrastiveRobustnessHead(nn.Module):
    """
    Contrastive learning head that forces the model to learn
    robust action representations rather than memorizing trajectories.
    
    Key insight: Actions that achieve the same goal should be close
    in embedding space, regardless of perturbations.
    """
    
    def __init__(self, action_dim: int, embed_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
        # Action encoder: maps actions to contrastive space
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )
        
        # Observation encoder: maps observations to contrastive space
        self.obs_encoder = nn.Sequential(
            nn.Linear(512, 256),  # 512 = typical VLA feature dim
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )
        
        # Projection heads for contrastive learning
        self.action_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.obs_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(
        self,
        action_features: torch.Tensor,
        obs_features: torch.Tensor,
        positive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between actions and observations.
        
        Args:
            action_features: [B, action_dim] action features
            obs_features: [B, 512] observation features
            positive_mask: [B, B] mask indicating positive pairs
        
        Returns:
            Contrastive loss scalar
        """
        # Encode to contrastive space
        a_embed = self.action_proj(self.action_encoder(action_features))
        o_embed = self.obs_proj(self.obs_encoder(obs_features))
        
        # Normalize
        a_embed = F.normalize(a_embed, dim=-1)
        o_embed = F.normalize(o_embed, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(a_embed, o_embed.T) / self.temperature
        
        # InfoNCE loss
        # Positive pairs are on the diagonal (same task, perturbed vs original)
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class PRVLA(nn.Module):
    """
    Perturbation-Robust VLA wrapper.
    
    Wraps any existing VLA model and adds:
    1. Perturbation augmentation during training
    2. Contrastive robustness loss
    3. Standard action prediction loss
    
    Total added parameters: <10M
    """
    
    def __init__(
        self,
        base_vla: nn.Module,
        action_dim: int = 7,
        robustness_weight: float = 0.1,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.base_vla = base_vla
        self.robustness_weight = robustness_weight
        
        # Perturbation augmentor
        self.augmentor = PerturbationAugmentor()
        
        # Contrastive robustness head
        self.robustness_head = ContrastiveRobustnessHead(
            action_dim=action_dim,
            embed_dim=embed_dim,
        )
        
        # Feature extractor from base VLA (hook into intermediate layer)
        self.feature_dim = 512  # Typical VLA feature dimension
        self.feature_proj = nn.Linear(self.feature_dim, 512)
    
    def forward(
        self,
        observation: torch.Tensor,
        instruction: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        return_robustness_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional robustness loss computation.
        
        Args:
            observation: [B, C, H, W] visual observation
            instruction: [B, seq_len] language instruction tokens
            action: [B, action_dim] ground truth action (training only)
            return_robustness_loss: whether to compute robustness loss
        
        Returns:
            Dictionary with predicted action and optional losses
        """
        # Get base VLA prediction
        base_output = self.base_vla(observation, instruction)
        predicted_action = base_output['action'] if isinstance(base_output, dict) else base_output
        
        result = {'action': predicted_action}
        
        if action is not None and return_robustness_loss:
            # Create perturbed version
            batch = {
                'observation': observation,
                'instruction_tokens': instruction,
            }
            perturbed_batch = self.augmentor.augment(batch)
            
            # Get prediction on perturbed input
            perturbed_output = self.base_vla(
                perturbed_batch['observation'],
                perturbed_batch['instruction_tokens'],
            )
            perturbed_action = perturbed_output['action'] if isinstance(perturbed_output, dict) else perturbed_output
            
            # Standard BC loss
            bc_loss = F.mse_loss(predicted_action, action)
            
            # Robustness loss: original and perturbed should predict similar actions
            robustness_loss = F.mse_loss(predicted_action, perturbed_action)
            
            # Contrastive loss: learn robust action representations
            # Extract features from base VLA (use hook or intermediate output)
            obs_features = self.feature_proj(
                base_output.get('features', torch.zeros(observation.shape[0], self.feature_dim, device=observation.device))
            )
            contrastive_loss = self.robustness_head(
                action,
                obs_features,
                positive_mask=torch.eye(observation.shape[0], device=observation.device),
            )
            
            result['bc_loss'] = bc_loss
            result['robustness_loss'] = robustness_loss
            result['contrastive_loss'] = contrastive_loss
            result['total_loss'] = (
                bc_loss + 
                self.robustness_weight * robustness_loss + 
                0.05 * contrastive_loss
            )
        
        return result


def create_pr_vla(
    base_model_name: str = "smolvla",
    action_dim: int = 7,
    robustness_weight: float = 0.1,
) -> PRVLA:
    """
    Factory function to create a PR-VLA model.
    
    Args:
        base_model_name: name of base VLA ("smolvla", "progvla", "openvla")
        action_dim: action dimensionality
        robustness_weight: weight for robustness loss
    
    Returns:
        PRVLA model instance
    """
    # Load base VLA (placeholder - actual loading depends on model)
    if base_model_name == "smolvla":
        from transformers import AutoModel
        base_vla = AutoModel.from_pretrained("HuggingFaceTB/SmolVLA-450M")
    elif base_model_name == "progvla":
        # Placeholder for ProgVLA
        base_vla = nn.Linear(512, action_dim)  # Placeholder
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    return PRVLA(
        base_vla=base_vla,
        action_dim=action_dim,
        robustness_weight=robustness_weight,
    )


if __name__ == "__main__":
    # Quick test
    model = PRVLA(
        base_vla=nn.Linear(512, 7),  # Placeholder
        action_dim=7,
    )
    
    # Count added parameters
    total_params = sum(p.numel() for p in model.parameters())
    base_params = sum(p.numel() for p in model.base_vla.parameters())
    added_params = total_params - base_params
    
    print(f"Base model params: {base_params:,}")
    print(f"Added params: {added_params:,}")
    print(f"Total params: {total_params:,}")
    print(f"Overhead: {added_params/total_params*100:.2f}%")
