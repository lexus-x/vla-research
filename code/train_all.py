"""
Training script for all 3 VLA ideas on AWS GPU.
Clean standalone version — no external module imports needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
import numpy as np


# ============================================================
# Shared Components
# ============================================================

class SimpleVLABackbone(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=768):
        super().__init__()
        self.input_dim = input_dim
        self.vision_proj = nn.Linear(input_dim, hidden_dim)
        self.lang_proj = nn.Linear(input_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, vision_input, lang_input=None):
        v = self.vision_proj(vision_input[:, :self.input_dim])
        if lang_input is None:
            lang_input = torch.zeros_like(vision_input)
        l = self.lang_proj(lang_input[:, :self.input_dim])
        return self.fusion(torch.cat([v, l], dim=-1))


def make_batch(batch_size=8, device='cuda'):
    return {
        'images': torch.randn(batch_size, 768, device=device),
        'instruction': torch.randn(batch_size, 768, device=device),
        'actions': torch.randn(batch_size, 7, device=device),
        'task_ids': torch.randint(0, 10, (batch_size,), device=device),
        'proprioception': torch.randn(batch_size, 6, device=device),
    }


# ============================================================
# Idea 1: PerturbVLA
# ============================================================

def train_perturb_vla():
    print("\n" + "="*70)
    print("TRAINING: PerturbVLA (Adversarial Perturbation Training)")
    print("="*70)

    device = 'cuda'
    backbone = SimpleVLABackbone().to(device)
    # Contrastive projector
    projector = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)
    action_head = nn.Linear(768, 7).to(device)

    params = list(backbone.parameters()) + list(projector.parameters()) + list(action_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    results = {'losses': []}
    num_steps = 500
    temperature = 0.07

    print(f"Training for {num_steps} steps...")
    start = time.time()

    for step in range(num_steps):
        backbone.train()
        strength = min(1.0, 0.1 + step / num_steps * 0.9)  # curriculum

        batch = make_batch(8, device)

        # Original forward
        feat_o = backbone(batch['images'], batch['instruction'])
        act_o = action_head(feat_o)
        loss_act = F.mse_loss(act_o, batch['actions'])

        # Perturbed forward (inline perturbation — avoids shape issues)
        noise_s = torch.randn_like(batch['images']) * 0.1 * strength
        noise_l = torch.randn_like(batch['instruction']) * 0.05 * strength
        feat_p = backbone(batch['images'] + noise_s, batch['instruction'] + noise_l)
        act_p = action_head(feat_p)
        loss_pert = F.mse_loss(act_p, batch['actions'])

        # Contrastive loss
        z_o = F.normalize(projector(feat_o), dim=-1)
        z_p = F.normalize(projector(feat_p), dim=-1)
        sim = torch.mm(z_o, z_p.t()) / temperature
        # Same-task pairs are diagonal (same samples, different perturbations)
        labels = torch.arange(z_o.shape[0], device=device)
        loss_con = F.cross_entropy(sim, labels)

        total = loss_act + loss_pert + 0.1 * loss_con

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        results['losses'].append(total.item())

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{num_steps} | Loss: {total.item():.4f} | "
                  f"Act: {loss_act.item():.4f} | Pert: {loss_pert.item():.4f} | "
                  f"Con: {loss_con.item():.4f} | Str: {strength:.2f}")

    torch.save(backbone.state_dict(), '/home/ubuntu/vla-research/results/perturb_vla.pth')
    print(f"PerturbVLA done. Final loss: {results['losses'][-1]:.4f} | Time: {time.time()-start:.0f}s")
    return results


# ============================================================
# Idea 2: MultiRes-Action
# ============================================================

def train_multires_action():
    print("\n" + "="*70)
    print("TRAINING: MultiRes-Action (Coarse-to-Fine)")
    print("="*70)

    device = 'cuda'
    backbone = SimpleVLABackbone().to(device)

    # Coarse planner (flow matching-style MLP)
    coarse_planner = nn.Sequential(
        nn.Linear(768 + 7*16 + 1, 512), nn.SiLU(),  # +7 action +1 time
        nn.Linear(512, 512), nn.SiLU(),
        nn.Linear(512, 7 * 16),  # 16-step horizon
    ).to(device)

    # Fine controller
    fine_controller = nn.Sequential(
        nn.Linear(768 + 7 + 6, 128), nn.ReLU(),  # +7 subgoal +6 proprio
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 7 * 4),  # 4-step refinement
    ).to(device)

    # Confidence gate
    confidence = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid()).to(device)

    all_params = (list(backbone.parameters()) + list(coarse_planner.parameters()) +
                  list(fine_controller.parameters()) + list(confidence.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=1e-4)

    results = {'losses': [], 'confidences': []}
    num_steps = 500

    print(f"Training for {num_steps} steps...")
    start = time.time()

    for step in range(num_steps):
        batch = make_batch(8, device)
        features = backbone(batch['images'], batch['instruction'])

        # Coarse: flow matching with 4 Euler steps
        h = features  # (B, 768)
        x = torch.randn(8, 7 * 16, device=device)
        for t in torch.linspace(0, 1, 4, device=device):
            inp = torch.cat([h, x, t.expand(8, 1)], dim=-1)
            vel = coarse_planner(inp)
            x = x + vel * 0.25
        coarse_actions = x.view(8, 16, 7)

        # Fine: refine using first coarse step as subgoal
        subgoal = coarse_actions[:, 0, :]
        fine_inp = torch.cat([features, subgoal, batch['proprioception']], dim=-1)
        fine_actions = fine_controller(fine_inp).view(8, 4, 7)

        # Confidence
        conf = confidence(features)
        results['confidences'].append(conf.mean().item())

        # Loss: coarse predicts full horizon, fine predicts first 4 steps
        target = batch['actions'].unsqueeze(1).expand(-1, 16, -1)  # (B, 16, 7)
        loss_coarse = F.mse_loss(coarse_actions, target)
        loss_fine = F.mse_loss(fine_actions, target[:, :4, :])
        total = loss_coarse + 0.5 * loss_fine

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        results['losses'].append(total.item())

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{num_steps} | Loss: {total.item():.4f} | "
                  f"Coarse: {loss_coarse.item():.4f} | Fine: {loss_fine.item():.4f} | "
                  f"Conf: {conf.mean().item():.3f}")

    torch.save({'backbone': backbone.state_dict(),
                'coarse': coarse_planner.state_dict(),
                'fine': fine_controller.state_dict()},
               '/home/ubuntu/vla-research/results/multires.pth')
    print(f"MultiRes done. Final loss: {results['losses'][-1]:.4f} | Time: {time.time()-start:.0f}s")
    return results


# ============================================================
# Idea 3: MambaFlow
# ============================================================

def train_mambaflow():
    print("\n" + "="*70)
    print("TRAINING: MambaFlow (SSM Backbone + Flow Matching)")
    print("="*70)

    device = 'cuda'

    # SSM-inspired backbone (simplified Mamba-like blocks)
    class SSMBlock(nn.Module):
        def __init__(self, d=256):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.in_proj = nn.Linear(d, d * 2, bias=False)
            self.conv = nn.Conv1d(d, d, 4, padding=3, groups=d)
            self.out_proj = nn.Linear(d, d, bias=False)
            self.D = nn.Parameter(torch.ones(d))

        def forward(self, x):
            residual = x
            x = self.norm(x)
            xz = self.in_proj(x)
            x_s, z = xz.chunk(2, dim=-1)
            # Conv + gating (simplified SSM)
            x_c = F.silu(self.conv(x_s.transpose(1, 2)).transpose(1, 2)[:, :x.shape[1], :])
            y = (x_c + self.D * x_s) * torch.sigmoid(z)
            return self.out_proj(y) + residual

    class MambaFlowModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(768, 256)
            self.pos_embed = nn.Parameter(torch.randn(1, 20, 256) * 0.02)
            self.blocks = nn.Sequential(*[SSMBlock(256) for _ in range(8)])
            self.norm = nn.LayerNorm(256)
            # Flow matching head
            self.flow_head = nn.Sequential(
                nn.Linear(256 + 7*16 + 1, 256), nn.SiLU(),
                nn.Linear(256, 256), nn.SiLU(),
                nn.Linear(256, 7 * 16),
            )
            self.action_dim = 7
            self.horizon = 16

        def forward(self, x, flow_steps=4):
            B = x.shape[0]
            x = self.input_proj(x)  # (B, 1, 256)
            x = x + self.pos_embed[:, :1, :]
            x = self.blocks(x)
            features = self.norm(x).mean(dim=1)  # (B, 256)

            # Flow matching
            z = torch.randn(B, self.action_dim * self.horizon, device=x.device)
            for t in torch.linspace(0, 1, flow_steps, device=x.device):
                inp = torch.cat([features, z, t.expand(B, 1)], dim=-1)
                z = z + self.flow_head(inp) * (1.0 / flow_steps)

            return z.view(B, self.horizon, self.action_dim), features

    model = MambaFlowModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"MambaFlow params: {total_params:,} ({total_params/1e6:.1f}M)")

    results = {'losses': [], 'latencies': []}
    num_steps = 500

    print(f"Training for {num_steps} steps...")
    start = time.time()

    for step in range(num_steps):
        model.train()
        batch = make_batch(4, device)

        # Simulate sequence input (20 tokens of 768-dim)
        x = batch['images'].unsqueeze(1).expand(-1, 20, -1)  # (4, 20, 768)
        actions_pred, features = model(x, flow_steps=4)

        # Target: expand to horizon
        target = batch['actions'].unsqueeze(1).expand(-1, 16, -1)
        loss = F.mse_loss(actions_pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        results['losses'].append(loss.item())

        if (step + 1) % 100 == 0:
            # Latency benchmark
            model.eval()
            with torch.no_grad():
                t0 = time.time()
                for _ in range(20):
                    model(x[:1], flow_steps=4)
                lat = (time.time() - t0) / 20 * 1000
            results['latencies'].append(lat)
            print(f"  Step {step+1}/{num_steps} | Loss: {loss.item():.4f} | "
                  f"Latency: {lat:.1f}ms | Time: {time.time()-start:.0f}s")

    torch.save(model.state_dict(), '/home/ubuntu/vla-research/results/mambaflow.pth')
    if results['latencies']:
        print(f"MambaFlow done. Final loss: {results['losses'][-1]:.4f} | "
              f"Avg latency: {np.mean(results['latencies']):.1f}ms")
    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.makedirs('/home/ubuntu/vla-research/results', exist_ok=True)

    print("VLA Research: Training Top 3 Ideas")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    results = {}
    results['perturb_vla'] = train_perturb_vla()
    results['multires'] = train_multires_action()
    results['mambaflow'] = train_mambaflow()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, r in results.items():
        print(f"  {name}: Final Loss = {r['losses'][-1]:.4f}")

    with open('/home/ubuntu/vla-research/results/summary.json', 'w') as f:
        json.dump({k: {'final_loss': v['losses'][-1], 'num_steps': len(v['losses'])} for k, v in results.items()}, f)
    print("\nAll results saved to /home/ubuntu/vla-research/results/")
