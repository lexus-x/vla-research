#!/usr/bin/env python3
"""
Diagnosis Experiments — Direct OpenVLA Load
============================================
Uses OpenVLA's own loading method, bypassing Auto classes.
"""
import torch
import json
import time
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("/tmp/vla_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# Load Model — Use OpenVLA's own loading
# ============================================================
print("\n[Loader] Importing OpenVLA...")
from transformers import AutoProcessor

# OpenVLA uses PrismaticForActionPrediction with trust_remote_code
MODEL_ID = "openvla/openvla-7b-finetuned-libero-spatial"
print(f"[Loader] Loading {MODEL_ID}...")

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load model via the custom class registered by trust_remote_code
# The model's auto_map should point to the right class
config = processor.model.config if hasattr(processor.model, 'config') else None
print(f"[Loader] Processor loaded ({type(processor).__name__})")

# Try loading with AutoModelForVision2Seq from the cached module
try:
    # Import the custom model class directly from the cached module
    import importlib.util
    import sys
    
    # The cached module path
    cache_dir = Path.home() / ".cache/huggingface/modules/transformers_modules"
    model_module = None
    
    # Find the prismatic model module
    for p in cache_dir.rglob("*prismatic*modeling*.py"):
        if "modeling" in p.name:
            spec = importlib.util.spec_from_file_location("prismatic_model", str(p))
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            print(f"[Loader] Found model module: {p}")
            break
    
    if model_module is not None:
        # Get the model class
        model_class = getattr(model_module, 'PrismaticForActionPrediction', None)
        if model_class is None:
            # Try other names
            for name in dir(model_module):
                obj = getattr(model_module, name)
                if isinstance(obj, type) and hasattr(obj, 'from_pretrained'):
                    model_class = obj
                    print(f"[Loader] Using class: {name}")
                    break
        
        if model_class:
            model = model_class.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        else:
            raise ValueError("Could not find model class in cached module")
    else:
        raise ValueError("Could not find prismatic model module")
        
except Exception as e:
    print(f"[Loader] Direct load failed: {e}")
    print("[Loader] Trying transformers AutoModelForVision2Seq...")
    from transformers import AutoModelForVision2Seq
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

model = model.to(DEVICE)
model.eval()
vram = torch.cuda.memory_allocated() / 1e9
print(f"[Loader] Model loaded in {time.time()-t0:.0f}s, VRAM: {vram:.2f}GB")

# ============================================================
# Experiments
# ============================================================

def predict(image, instruction):
    """Run single-step action prediction."""
    inputs = processor(instruction, image).to(DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        action = model.predict_action(inputs, unnorm_key="libero_spatial", do_sample=False)
    return action

def make_synthetic_episodes(n=10):
    episodes = []
    tasks = [
        "pick up the red bowl and place it on the plate",
        "open the top drawer of the cabinet",
        "push the mug to the left side of the stove",
    ]
    for i in range(n):
        img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        episodes.append({"image": img, "instruction": tasks[i % len(tasks)]})
    return episodes

def perturb_image(img, strength):
    from PIL import ImageFilter, ImageEnhance
    if strength == 0:
        return img
    img = img.filter(ImageFilter.GaussianBlur(radius=2*strength))
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(1.0 + 0.3*strength*(1 if np.random.rand()>0.5 else -1))

def perturb_instruction(instr, strength):
    if strength == 0:
        return instr
    words = instr.split()
    n_drop = max(1, int(len(words) * 0.15 * strength))
    indices = set(np.random.choice(len(words), min(n_drop, len(words)), replace=False))
    return " ".join(w for i, w in enumerate(words) if i not in indices)

print("\n[Setup] Loading episodes...")
episodes = make_synthetic_episodes(10)
print(f"[Setup] {len(episodes)} synthetic episodes ready")

# ---- Exp1: Perturbation Sensitivity ----
print("\n" + "="*40)
print("EXP1: Perturbation Sensitivity")
print("="*40)
exp1 = {"experiment": "exp1", "strengths": {}}
for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
    actions = []
    latencies = []
    for ep in episodes:
        img = perturb_image(ep["image"], s)
        t0 = time.time()
        a = predict(img, ep["instruction"])
        latencies.append((time.time()-t0)*1000)
        actions.append(a if isinstance(a, list) else a.tolist() if hasattr(a, 'tolist') else list(a))
    arr = np.array(actions)
    exp1["strengths"][str(s)] = {
        "n": len(actions),
        "mean_lat_ms": round(np.mean(latencies),1),
        "action_mean": arr.mean(0).tolist(),
        "action_std": arr.std(0).tolist(),
    }
    print(f"  s={s:.2f}: lat={np.mean(latencies):.0f}ms, action_std={arr.std(0).mean():.6f}")

(OUTPUT_DIR / "exp1_perturbation_sensitivity.json").write_text(json.dumps(exp1, indent=2, default=str))

# ---- Exp2: Language Ablation ----
print("\n" + "="*40)
print("EXP2: Language Ablation")
print("="*40)
exp2 = {"experiment": "exp2", "dropouts": {}}
for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
    actions = []
    latencies = []
    for ep in episodes:
        instr = perturb_instruction(ep["instruction"], s) if s < 1.0 else ""
        t0 = time.time()
        a = predict(ep["image"], instr)
        latencies.append((time.time()-t0)*1000)
        actions.append(a if isinstance(a, list) else a.tolist() if hasattr(a, 'tolist') else list(a))
    arr = np.array(actions)
    exp2["dropouts"][str(s)] = {
        "n": len(actions),
        "mean_lat_ms": round(np.mean(latencies),1),
        "action_mean": arr.mean(0).tolist(),
        "action_std": arr.std(0).tolist(),
        "actions_identical": bool(arr.std(0).sum() < 1e-10),
    }
    print(f"  dropout={s:.2f}: lat={np.mean(latencies):.0f}ms, identical={arr.std(0).sum() < 1e-10}")

(OUTPUT_DIR / "exp2_language_ablation.json").write_text(json.dumps(exp2, indent=2, default=str))

# ---- Exp3: Temporal Consistency ----
print("\n" + "="*40)
print("EXP3: Temporal Consistency")
print("="*40)
exp3 = {"experiment": "exp3", "episodes": []}
for ep in episodes[:3]:
    actions = []
    for _ in range(10):
        a = predict(ep["image"], ep["instruction"])
        actions.append(a if isinstance(a, list) else a.tolist() if hasattr(a, 'tolist') else list(a))
    arr = np.array(actions)
    var = float(arr.var(0).mean())
    exp3["episodes"].append({"variance": var, "deterministic": var < 1e-10})
    print(f"  episode: variance={var:.12f}, deterministic={var < 1e-10}")

(OUTPUT_DIR / "exp3_temporal.json").write_text(json.dumps(exp3, indent=2, default=str))

# ---- Summary ----
print("\n" + "="*40)
print("ALL EXPERIMENTS COMPLETE")
print("="*40)
summary = {
    "model": MODEL_ID,
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if DEVICE=="cuda" else "cpu",
    "vram_gb": round(torch.cuda.memory_allocated()/1e9, 2),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "experiments": ["exp1_perturbation_sensitivity", "exp2_language_ablation", "exp3_temporal"],
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"Results in: {OUTPUT_DIR}")
