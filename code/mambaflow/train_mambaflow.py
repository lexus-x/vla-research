#!/usr/bin/env python3
"""
MambaFlow Training Script — SSM Backbone + Flow Matching VLA
Runs on: L4 (24GB) or L40S (48GB)
Produces: run_manifest.json with all provenance info
"""
import os, sys, json, time, hashlib, argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Mamba-like SSM block (pure PyTorch, no mamba-ssm dependency) ---
class SelectiveSSM(nn.Module):
    """Simplified selective state space model (Mamba-style) in pure PyTorch."""
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv-1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(d_state, self.d_inner)
        
        # SSM parameters
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(self, x):
        """x: (B, L, D)"""
        B, L, D = x.shape
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_proj, z = xz.chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # Selective scan (simplified)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        deltaBC = self.x_proj(x_conv)  # (B, L, 2*d_state+1)
        delta, B_param, C_param = deltaBC.split([1, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_inner)
        
        # Discretize and scan
        y = torch.zeros(B, L, self.d_inner, device=x.device)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
        
        for t in range(L):
            dt = delta[:, t, :].unsqueeze(-1)  # (B, d_inner, 1)
            A_disc = torch.exp(A.unsqueeze(0) * dt)  # (B, d_inner, d_state)
            B_disc = B_param[:, t, :].unsqueeze(1) * dt  # (B, d_inner, d_state)
            
            h = A_disc * h + B_disc * x_conv[:, t, :].unsqueeze(-1)
            y[:, t, :] = (h * C_param[:, t, :].unsqueeze(1)).sum(-1) + self.D * x_conv[:, t, :]
        
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
    
    def forward(self, x):
        return x + self.ssm(self.norm(x))


class MambaBackbone(nn.Module):
    """Mamba SSM backbone for VLA."""
    def __init__(self, d_model=256, n_layers=12, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class FlowMatchingHead(nn.Module):
    """Flow matching action head — learns velocity field v(x_t, t) → dx/dt."""
    def __init__(self, d_model=256, action_dim=7, n_steps=50):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, d_model)
        )
        self.net = nn.Sequential(
            nn.Linear(d_model + action_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, action_dim),
        )
        self.n_steps = n_steps
    
    def forward(self, z, t, x_t):
        """
        z: (B, D) conditioning from backbone
        t: (B, 1) time in [0, 1]
        x_t: (B, action_dim) noisy action at time t
        Returns: predicted velocity (B, action_dim)
        """
        t_emb = self.time_embed(t)  # (B, D)
        h = z + t_emb  # (B, D)
        inp = torch.cat([h, x_t], dim=-1)
        return self.net(inp)
    
    @torch.no_grad()
    def sample(self, z, action_dim, n_steps=None):
        """Euler integration from noise to action."""
        n_steps = n_steps or self.n_steps
        B = z.shape[0]
        x = torch.randn(B, action_dim, device=z.device)
        dt = 1.0 / n_steps
        
        for i in range(n_steps):
            t = torch.full((B, 1), i * dt, device=z.device)
            v = self.forward(z, t, x)
            x = x + v * dt
        
        return x


class MambaFlowVLA(nn.Module):
    """Full MambaFlow model: Vision encoder + Mamba backbone + Flow matching head."""
    def __init__(self, d_model=256, n_layers=8, action_dim=7, action_chunk=16):
        super().__init__()
        # Vision encoder (SigLIP-like, simplified)
        self.vision_proj = nn.Sequential(
            nn.Linear(768, d_model),  # SigLIP output dim
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Language encoder (simplified)
        self.lang_proj = nn.Sequential(
            nn.Linear(512, d_model),  # text encoder output dim
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Mamba backbone
        self.backbone = MambaBackbone(d_model, n_layers)
        
        # Flow matching head for action chunks
        self.action_head = FlowMatchingHead(d_model, action_dim * action_chunk)
        self.action_dim = action_dim
        self.action_chunk = action_chunk
    
    def forward(self, vision_feat, lang_feat, action_gt=None, t=None):
        """
        vision_feat: (B, N_vis, 768) vision features
        lang_feat: (B, 512) language features
        action_gt: (B, action_chunk * action_dim) ground truth actions (for training)
        """
        # Project to shared space
        v = self.vision_proj(vision_feat)  # (B, N_vis, d_model)
        l = self.lang_proj(lang_feat).unsqueeze(1)  # (B, 1, d_model)
        
        # Concatenate and process through Mamba
        x = torch.cat([l, v], dim=1)  # (B, 1+N_vis, d_model)
        z = self.backbone(x).mean(dim=1)  # (B, d_model) — pool
        
        if action_gt is not None and t is not None:
            # Training: predict velocity at noisy action
            noise = torch.randn_like(action_gt)
            x_t = (1 - t) * noise + t * action_gt  # interpolation
            v_pred = self.action_head(z, t, x_t)
            # Target velocity: action_gt - noise
            v_target = action_gt - noise
            loss = F.mse_loss(v_pred, v_target)
            return loss
        else:
            # Inference: sample actions
            return self.action_head.sample(z, self.action_dim * self.action_chunk)


def compute_model_hash(model):
    """Compute hash of model state dict for provenance."""
    buf = bytearray()
    for k, v in sorted(model.state_dict().items()):
        buf.extend(k.encode())
        buf.extend(v.cpu().numpy().tobytes())
    return hashlib.sha256(buf).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Run manifest ---
    manifest = {
        "model": "MambaFlow",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": None,
        "gpu_name": None,
        "gpu_memory_mb": None,
        "args": vars(args),
        "checkpoints": [],
        "training_log": [],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        manifest["device"] = str(device)
        manifest["gpu_name"] = torch.cuda.get_device_name(0)
        manifest["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024*1024)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Model ---
    model = MambaFlowVLA(
        d_model=args.d_model,
        n_layers=args.n_layers,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    manifest["param_count"] = param_count
    print(f"MambaFlow params: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # --- Synthetic training loop (replaces with real LIBERO data later) ---
    print(f"Starting training: {args.steps} steps on {device}")
    start_time = time.time()
    
    for step in range(1, args.steps + 1):
        model.train()
        
        # Synthetic data (replace with real dataloader)
        B = args.batch_size
        vision_feat = torch.randn(B, 49, 768, device=device)  # 7x7 patches
        lang_feat = torch.randn(B, 512, device=device)
        action_gt = torch.randn(B, args.action_chunk * args.action_dim, device=device)
        
        # Random time for flow matching
        t = torch.rand(B, 1, device=device)
        
        loss = model(vision_feat, lang_feat, action_gt, t)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            print(f"step={step}/{args.steps} loss={loss.item():.4f} lr={lr:.2e} elapsed={elapsed:.0f}s")
            manifest["training_log"].append({
                "step": step,
                "loss": loss.item(),
                "lr": lr,
                "elapsed_s": elapsed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        
        if step % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"mambaflow_step{step}.pt")
            ckpt_hash = compute_model_hash(model)
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "args": vars(args),
                "model_hash": ckpt_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, ckpt_path)
            manifest["checkpoints"].append({
                "step": step,
                "path": ckpt_path,
                "hash": ckpt_hash,
                "loss": loss.item(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            print(f"  Saved checkpoint: {ckpt_path} (hash={ckpt_hash})")

    # --- Final ---
    manifest["end_time"] = datetime.now(timezone.utc).isoformat()
    manifest["total_time_s"] = time.time() - start_time
    manifest["final_loss"] = manifest["training_log"][-1]["loss"] if manifest["training_log"] else None
    manifest["final_model_hash"] = compute_model_hash(model)

    # Save final checkpoint
    final_path = os.path.join(args.output_dir, "mambaflow_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "model_hash": manifest["final_model_hash"],
        "manifest": manifest,
    }, final_path)

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nTraining complete. Manifest: {manifest_path}")
    print(f"Final loss: {manifest['final_loss']:.4f}")
    print(f"Total time: {manifest['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
