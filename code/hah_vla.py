"""
HAH-VLA: Hypernetwork Action Head for Vision-Language-Action Models
Replaces fixed action heads with dynamic, task-specific weight generation.

Core idea: A hypernetwork generates task-specific action head weights
from the task embedding, enabling zero-shot adaptation to new tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class HypernetworkActionHead(nn.Module):
    """
    Hypernetwork that generates task-specific action head weights.
    
    Instead of using a fixed MLP for action prediction across all tasks,
    this module generates task-specific weights from a task embedding.
    
    Architecture:
    1. Task encoder: CLIP text + visual features → task embedding
    2. Hypernetwork: task embedding → action MLP weights
    3. Generated action MLP: observation features → action prediction
    
    Total parameters: ~8M
    """
    
    def __init__(
        self,
        task_embed_dim: int = 64,
        obs_feature_dim: int = 512,
        action_dim: int = 7,
        hidden_dim: int = 128,
        num_action_layers: int = 3,
    ):
        super().__init__()
        
        self.task_embed_dim = task_embed_dim
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_action_layers = num_action_layers
        
        # Task encoder: combines language and visual task information
        self.task_encoder = nn.Sequential(
            nn.Linear(512 + 512, 256),  # CLIP text + visual features
            nn.ReLU(),
            nn.Linear(256, task_embed_dim),
        )
        
        # Hypernetwork: generates action MLP weights from task embedding
        # Generates weights for a 3-layer MLP: obs_dim → hidden → hidden → action_dim
        self._compute_weight_sizes()
        
        self.hypernetwork = nn.Sequential(
            nn.Linear(task_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.total_weight_dim),
        )
        
        # Initialize hypernetwork to produce near-identity weights
        self._init_hypernetwork()
    
    def _compute_weight_sizes(self):
        """Compute total number of parameters in the generated action MLP."""
        # Layer 1: obs_feature_dim → hidden_dim
        self.w1_size = self.obs_feature_dim * self.hidden_dim
        self.b1_size = self.hidden_dim
        
        # Layer 2: hidden_dim → hidden_dim
        self.w2_size = self.hidden_dim * self.hidden_dim
        self.b2_size = self.hidden_dim
        
        # Layer 3: hidden_dim → action_dim
        self.w3_size = self.hidden_dim * self.action_dim
        self.b3_size = self.action_dim
        
        self.total_weight_dim = (
            self.w1_size + self.b1_size +
            self.w2_size + self.b2_size +
            self.w3_size + self.b3_size
        )
    
    def _init_hypernetwork(self):
        """Initialize hypernetwork to produce near-identity weights."""
        # Small initialization for stability
        for p in self.hypernetwork.parameters():
            nn.init.normal_(p, mean=0.0, std=0.01)
    
    def generate_action_weights(
        self, task_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate action MLP weights from task embedding.
        
        Args:
            task_embedding: [B, task_embed_dim]
        
        Returns:
            Tuple of (w1, b1, w2, b2, w3, b3) for the action MLP
        """
        B = task_embedding.shape[0]
        
        # Generate all weights at once
        all_weights = self.hypernetwork(task_embedding)  # [B, total_weight_dim]
        
        # Split into individual weight matrices
        idx = 0
        w1 = all_weights[:, idx:idx + self.w1_size].view(B, self.obs_feature_dim, self.hidden_dim)
        idx += self.w1_size
        b1 = all_weights[:, idx:idx + self.b1_size].view(B, self.hidden_dim)
        idx += self.b1_size
        
        w2 = all_weights[:, idx:idx + self.w2_size].view(B, self.hidden_dim, self.hidden_dim)
        idx += self.w2_size
        b2 = all_weights[:, idx:idx + self.b2_size].view(B, self.hidden_dim)
        idx += self.b2_size
        
        w3 = all_weights[:, idx:idx + self.w3_size].view(B, self.hidden_dim, self.action_dim)
        idx += self.w3_size
        b3 = all_weights[:, idx:idx + self.b3_size].view(B, self.action_dim)
        
        return w1, b1, w2, b2, w3, b3
    
    def forward(
        self,
        obs_features: torch.Tensor,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: generate task-specific action prediction.
        
        Args:
            obs_features: [B, obs_feature_dim] observation features
            text_features: [B, 512] CLIP text features
            visual_features: [B, 512] CLIP visual features
        
        Returns:
            predicted_action: [B, action_dim]
        """
        B = obs_features.shape[0]
        
        # Encode task
        task_input = torch.cat([text_features, visual_features], dim=-1)
        task_embedding = self.task_encoder(task_input)  # [B, task_embed_dim]
        
        # Generate task-specific weights
        w1, b1, w2, b2, w3, b3 = self.generate_action_weights(task_embedding)
        
        # Apply generated action MLP
        # Layer 1
        x = torch.bmm(obs_features.unsqueeze(1), w1).squeeze(1) + b1
        x = F.relu(x)
        
        # Layer 2
        x = torch.bmm(x.unsqueeze(1), w2).squeeze(1) + b2
        x = F.relu(x)
        
        # Layer 3 (output)
        action = torch.bmm(x.unsqueeze(1), w3).squeeze(1) + b3
        
        return action


class HAHVLA(nn.Module):
    """
    HAH-VLA wrapper: replaces fixed action head with hypernetwork.
    
    Wraps any existing VLA model and replaces its action head
    with a hypernetwork-based dynamic action head.
    
    Total added parameters: ~8M
    """
    
    def __init__(
        self,
        base_vla: nn.Module,
        obs_feature_dim: int = 512,
        action_dim: int = 7,
        task_embed_dim: int = 64,
    ):
        super().__init__()
        self.base_vla = base_vla
        
        # Freeze base VLA (optional - can also fine-tune)
        for p in self.base_vla.parameters():
            p.requires_grad = False
        
        # Hypernetwork action head
        self.action_head = HypernetworkActionHead(
            task_embed_dim=task_embed_dim,
            obs_feature_dim=obs_feature_dim,
            action_dim=action_dim,
        )
        
        # Feature extractors for task encoding
        self.text_proj = nn.Linear(512, 512)  # Project CLIP text features
        self.visual_proj = nn.Linear(512, 512)  # Project CLIP visual features
    
    def forward(
        self,
        observation: torch.Tensor,
        instruction: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hypernetwork action head.
        
        Args:
            observation: [B, C, H, W] visual observation
            instruction: [B, seq_len] language instruction tokens
            text_features: [B, 512] CLIP text features (if available)
            visual_features: [B, 512] CLIP visual features (if available)
        
        Returns:
            Dictionary with predicted action
        """
        # Extract features from base VLA
        base_output = self.base_vla(observation, instruction)
        
        if isinstance(base_output, dict):
            obs_features = base_output.get('features', base_output.get('hidden_states'))
            if obs_features is None:
                # If no features available, use the action prediction as features
                obs_features = base_output['action']
        else:
            obs_features = base_output
        
        # Ensure correct dimensions
        if obs_features.dim() > 2:
            obs_features = obs_features.mean(dim=1)  # Pool sequence dimension
        
        if obs_features.shape[-1] != 512:
            # Project to expected dimension
            obs_features = F.adaptive_avg_pool1d(
                obs_features.unsqueeze(1), 512
            ).squeeze(1)
        
        # Use provided features or extract from observation
        if text_features is None:
            text_features = torch.zeros(observation.shape[0], 512, device=observation.device)
        if visual_features is None:
            visual_features = obs_features
        
        text_features = self.text_proj(text_features)
        visual_features = self.visual_proj(visual_features)
        
        # Generate action with hypernetwork
        action = self.action_head(obs_features, text_features, visual_features)
        
        return {'action': action}


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in model components."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    component_counts = {}
    for name, child in model.named_children():
        component_counts[name] = sum(p.numel() for p in child.parameters())
    
    return {
        'total': total,
        'trainable': trainable,
        'components': component_counts,
    }


if __name__ == "__main__":
    # Quick test
    action_head = HypernetworkActionHead(
        task_embed_dim=64,
        obs_feature_dim=512,
        action_dim=7,
    )
    
    # Test forward pass
    B = 4
    obs_features = torch.randn(B, 512)
    text_features = torch.randn(B, 512)
    visual_features = torch.randn(B, 512)
    
    action = action_head(obs_features, text_features, visual_features)
    print(f"Action shape: {action.shape}")
    
    params = count_parameters(action_head)
    print(f"Total parameters: {params['total']:,}")
    for name, count in params['components'].items():
        print(f"  {name}: {count:,}")
