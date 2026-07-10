#!/usr/bin/env python3
"""
Diagnosis Experiment Runner for L4 GPU
=======================================
Runs all 5 PerturbVLA diagnosis experiments sequentially.
Uses OpenVLA finetuned-libero-* checkpoints as baseline.

Experiments:
  exp1: Perturbation sensitivity baseline
  exp2: Language ablation (corrected — with real LIBERO episodes)
  exp3: Visual perturbation
  exp4: Temporal perturbation  
  exp5: Combined perturbation

GPU: L4 (24GB, ~15.13GB usable at bf16)
Inference-only, no training.
"""
import torch
import json
import time
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Config
MODEL_ID = "openvla/openvla-7b"
FINETUNED_CKPT = "openvla/openvla-7b-finetuned-libero-spatial"  # Start with spatial
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("/tmp/vla_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LIBERO_CKPTS = [
    "openvla/openvla-7b-finetuned-libero-spatial",
    "openvla/openvla-7b-finetuned-libero-object",
    "openvla/openvla-7b-finetuned-libero-goal",
    "openvla/openvla-7b-finetuned-libero-long",
]

# ============================================================
# Model Loading
# ============================================================

def load_model(model_id: str):
    """Load OpenVLA checkpoint at bf16."""
    from transformers import AutoProcessor
    
    print(f"[Loader] Loading {model_id}...")
    t0 = time.time()
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    # OpenVLA uses PrismaticForActionPrediction — load via trust_remote_code
    # Try the specific OpenVLA model class first
    try:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    except Exception:
        # Fallback: load the model class from the remote code directly
        import importlib
        import sys
        # Clear any cached modules
        to_del = [k for k in sys.modules if 'prismatic' in k or 'openvla' in k]
        for k in to_del:
            del sys.modules[k]
        
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    
    model = model.to(DEVICE)
    model.eval()
    
    vram = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0
    print(f"[Loader] Loaded in {time.time()-t0:.0f}s, VRAM: {vram:.2f}GB")
    return model, processor

def predict_action(model, processor, image, instruction: str) -> List[float]:
    """Run single-step action prediction."""
    inputs = processor(instruction, image).to(DEVICE, dtype=torch.bfloat16)
    with torch.no_grad():
        action = model.predict_action(inputs, unnorm_key="libero_spatial", do_sample=False)
    return action.tolist() if hasattr(action, 'tolist') else list(action)

# ============================================================
# Perturbation Functions
# ============================================================

def perturb_image(image, perturb_type: str, strength: float = 1.0):
    """Apply perturbation to image."""
    import numpy as np
    from PIL import Image, ImageFilter, ImageEnhance
    
    img = image.copy()
    
    if perturb_type == "spatial":
        # Random crop + resize (simulates camera shift)
        w, h = img.size
        crop = int(min(w, h) * 0.05 * strength)
        img = img.crop((crop, crop, w-crop, h-crop)).resize((w, h))
    
    elif perturb_type == "lighting":
        enhancer = ImageEnhance.Brightness(img)
        factor = 1.0 + (0.3 * strength * (torch.randn(1).item() if hasattr(torch, 'randn') else 0))
        factor = max(0.5, min(1.5, factor))
        img = enhancer.enhance(factor)
    
    elif perturb_type == "blur":
        radius = 2.0 * strength
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    elif perturb_type == "noise":
        import numpy as np
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 25 * strength, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    
    elif perturb_type == "color":
        enhancer = ImageEnhance.Color(img)
        factor = 1.0 + (0.4 * strength * (1 if torch.rand(1).item() > 0.5 else -1))
        img = enhancer.enhance(max(0.3, factor))
    
    return img

def perturb_instruction(instruction: str, perturb_type: str, strength: float = 1.0):
    """Apply perturbation to language instruction."""
    import random
    
    if perturb_type == "dropout":
        words = instruction.split()
        n_drop = max(1, int(len(words) * 0.15 * strength))
        indices = random.sample(range(len(words)), min(n_drop, len(words)))
        words = [w for i, w in enumerate(words) if i not in indices]
        return " ".join(words)
    
    elif perturb_type == "shuffle":
        words = instruction.split()
        n_shuffle = max(1, int(len(words) * 0.3 * strength))
        indices = random.sample(range(len(words)), min(n_shuffle, len(words)))
        shuffled = list(words)
        random.shuffle([shuffled[i] for i in indices])
        return " ".join(shuffled)
    
    elif perturb_type == "empty":
        return ""
    
    return instruction

# ============================================================
# Experiments
# ============================================================

def load_libero_episodes(suite: str = "spatial", n_episodes: int = 20):
    """Load real LIBERO episodes. Falls back to synthetic if unavailable."""
    try:
        # Try loading LIBERO dataset
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite]()
        
        episodes = []
        for task_id in range(min(n_episodes, task_suite.n_tasks)):
            task = task_suite.get_task(task_id)
            task_name = task.name
            task_description = task.language
            task_bddl_file = task.bddl_file
            
            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 128,
                "camera_widths": 128,
            }
            env = OffScreenRenderEnv(**env_args)
            env.seed(42)
            
            obs = env.reset()
            init_state = env.get_init_state()
            
            episodes.append({
                "task_name": task_name,
                "instruction": task_description,
                "initial_obs": obs,
                "init_state": init_state,
                "env": env,
            })
        
        return episodes, True
    except Exception as e:
        print(f"[Diagnosis] LIBERO not available ({e}), using synthetic episodes")
        return generate_synthetic_episodes(n_episodes), False

def generate_synthetic_episodes(n: int = 20):
    """Generate synthetic image+instruction pairs for testing."""
    from PIL import Image
    import numpy as np
    
    episodes = []
    tasks = [
        "pick up the red bowl and place it on the plate",
        "open the top drawer of the cabinet",
        "push the mug to the left side of the stove",
        "turn on the stove",
        "pick up the black bowl from the counter",
    ]
    for i in range(n):
        img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        episodes.append({
            "task_name": f"synthetic_task_{i}",
            "instruction": tasks[i % len(tasks)],
            "image": img,
            "synthetic": True,
        })
    return episodes

def run_experiment(exp_name: str, model, processor, episodes, perturb_fn, strengths):
    """Run a single experiment across perturbation strengths."""
    results = {
        "experiment": exp_name,
        "model": MODEL_ID,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if DEVICE == "cuda" else 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_episodes": len(episodes),
        "strengths": strengths,
        "per_strength": {},
    }
    
    for strength in strengths:
        print(f"  [{exp_name}] strength={strength:.2f}")
        actions_list = []
        latencies = []
        
        for ep in episodes:
            image = ep.get("image", ep.get("initial_obs"))
            if image is None:
                continue
            if hasattr(image, 'shape'):  # numpy array
                from PIL import Image
                image = Image.fromarray(image)
            
            instruction = ep["instruction"]
            perturbed_img = perturb_fn(image, strength) if perturb_fn else image
            
            t0 = time.time()
            try:
                action = predict_action(model, processor, perturbed_img, instruction)
                latencies.append((time.time() - t0) * 1000)
                actions_list.append(action)
            except Exception as e:
                print(f"    Inference failed: {e}")
        
        if actions_list:
            import numpy as np
            actions_arr = np.array(actions_list)
            results["per_strength"][str(strength)] = {
                "n_samples": len(actions_list),
                "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
                "action_mean": actions_arr.mean(axis=0).tolist(),
                "action_std": actions_arr.std(axis=0).tolist(),
                "action_samples": actions_arr[:3].tolist(),
            }
    
    return results

def exp1_perturbation_sensitivity(model, processor, episodes):
    """Exp1: Baseline perturbation sensitivity — how much do actions change under perturbation?"""
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    perturbation_types = ["spatial", "lighting", "blur", "noise", "color"]
    all_results = {}
    
    for pt in perturbation_types:
        print(f"[Exp1] Testing {pt} perturbation...")
        def pfn(img, s, _pt=pt): return perturb_image(img, _pt, s)
        result = run_experiment(f"exp1_{pt}", model, processor, episodes, pfn, strengths)
        all_results[pt] = result
    
    return {"experiment": "exp1_perturbation_sensitivity", "results": all_results}

def exp2_language_ablation(model, processor, episodes):
    """Exp2: Language ablation — does language actually affect actions?"""
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    results_by_dropout = {}
    for strength in strengths:
        print(f"[Exp2] Language dropout={strength:.2f}")
        actions_list = []
        latencies = []
        
        for ep in episodes:
            image = ep.get("image", ep.get("initial_obs"))
            if image is None:
                continue
            if hasattr(image, 'shape'):
                from PIL import Image
                image = Image.fromarray(image)
            
            instruction = ep["instruction"]
            if strength > 0:
                instruction = perturb_instruction(instruction, "dropout", strength)
            
            t0 = time.time()
            try:
                action = predict_action(model, processor, image, instruction)
                latencies.append((time.time() - t0) * 1000)
                actions_list.append(action)
            except Exception as e:
                print(f"    Failed: {e}")
        
        if actions_list:
            import numpy as np
            arr = np.array(actions_list)
            results_by_dropout[str(strength)] = {
                "n_samples": len(actions_list),
                "mean_latency_ms": round(sum(latencies) / len(latencies), 1),
                "action_mean": arr.mean(axis=0).tolist(),
                "action_std": arr.std(axis=0).tolist(),
                "action_samples": arr[:3].tolist(),
            }
    
    # Check if actions change
    import numpy as np
    if "0.0" in results_by_dropout and "1.0" in results_by_dropout:
        a0 = np.array(results_by_dropout["0.0"]["action_mean"])
        a1 = np.array(results_by_dropout["1.0"]["action_mean"])
        diff = np.abs(a0 - a1).mean()
        results_by_dropout["analysis"] = {
            "mean_action_diff_0_vs_100": float(diff),
            "language_matters": bool(diff > 0.001),
        }
    
    return {"experiment": "exp2_language_ablation", "results": results_by_dropout}

def exp3_visual_perturbation(model, processor, episodes):
    """Exp3: Visual perturbation — robustness to visual changes."""
    strengths = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    visual_types = ["lighting", "blur", "noise", "color"]
    all_results = {}
    
    for vt in visual_types:
        print(f"[Exp3] Testing {vt}...")
        def pfn(img, s, _vt=vt): return perturb_image(img, _vt, s)
        result = run_experiment(f"exp3_{vt}", model, processor, episodes, pfn, strengths)
        all_results[vt] = result
    
    return {"experiment": "exp3_visual_perturbation", "results": all_results}

def exp4_temporal_perturbation(model, processor, episodes):
    """Exp4: Temporal consistency — do actions change across repeated identical inputs?"""
    n_repeats = 20
    
    print(f"[Exp4] Temporal consistency test ({n_repeats} repeats)...")
    all_actions = []
    latencies = []
    
    for ep in episodes[:5]:  # Use first 5 episodes
        image = ep.get("image", ep.get("initial_obs"))
        if image is None:
            continue
        if hasattr(image, 'shape'):
            from PIL import Image
            image = Image.fromarray(image)
        
        ep_actions = []
        for _ in range(n_repeats):
            t0 = time.time()
            try:
                action = predict_action(model, processor, image, ep["instruction"])
                latencies.append((time.time() - t0) * 1000)
                ep_actions.append(action)
            except:
                pass
        all_actions.append(ep_actions)
    
    import numpy as np
    # Compute within-episode variance
    variances = []
    for ep_acts in all_actions:
        if len(ep_acts) > 1:
            arr = np.array(ep_acts)
            variances.append(arr.var(axis=0).mean())
    
    return {
        "experiment": "exp4_temporal_perturbation",
        "n_episodes": len(all_actions),
        "n_repeats": n_repeats,
        "mean_within_episode_variance": float(np.mean(variances)) if variances else 0,
        "max_within_episode_variance": float(np.max(variances)) if variances else 0,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "actions_deterministic": bool(np.mean(variances) < 1e-10) if variances else None,
        "per_episode_variance": [float(v) for v in variances],
    }

def exp5_combined_perturbation(model, processor, episodes):
    """Exp5: Combined perturbation — worst-case robustness."""
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    def combined_perturb(image, strength):
        img = perturb_image(image, "spatial", strength * 0.5)
        img = perturb_image(img, "lighting", strength * 0.7)
        img = perturb_image(img, "noise", strength * 0.3)
        return img
    
    print("[Exp5] Combined perturbation...")
    results = run_experiment("exp5_combined", model, processor, episodes, combined_perturb, strengths)
    
    # Also test with language dropout
    combined_lang_results = {}
    for strength in strengths:
        actions_list = []
        for ep in episodes:
            image = ep.get("image", ep.get("initial_obs"))
            if image is None:
                continue
            if hasattr(image, 'shape'):
                from PIL import Image
                image = Image.fromarray(image)
            
            img = combined_perturb(image, strength)
            instruction = perturb_instruction(ep["instruction"], "dropout", strength)
            
            try:
                action = predict_action(model, processor, img, instruction)
                actions_list.append(action)
            except:
                pass
        
        if actions_list:
            import numpy as np
            arr = np.array(actions_list)
            combined_lang_results[str(strength)] = {
                "n_samples": len(actions_list),
                "action_mean": arr.mean(axis=0).tolist(),
                "action_std": arr.std(axis=0).tolist(),
            }
    
    return {
        "experiment": "exp5_combined_perturbation",
        "visual_only": results,
        "visual_plus_language": combined_lang_results,
    }

# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("PerturbVLA Diagnosis Experiments")
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)
    
    # Load episodes
    print("\n[Setup] Loading episodes...")
    episodes, has_libero = load_libero_episodes("spatial", n_episodes=20)
    print(f"[Setup] Loaded {len(episodes)} episodes (real LIBERO: {has_libero})")
    
    # Load model
    print("\n[Setup] Loading model...")
    try:
        model, processor = load_model(FINETUNED_CKPT)
    except Exception as e:
        print(f"[Setup] Failed to load finetuned checkpoint: {e}")
        print("[Setup] Falling back to base OpenVLA...")
        try:
            model, processor = load_model(MODEL_ID)
        except Exception as e2:
            print(f"[Setup] Base OpenVLA also failed: {e2}")
            print("[Setup] Trying with trust_remote_code + AutoModel...")
            from transformers import AutoProcessor, AutoModel
            processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
            model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE)
            model.eval()
    
    # Run experiments
    experiments = [
        ("exp1_perturbation_sensitivity", exp1_perturbation_sensitivity),
        ("exp2_language_ablation", exp2_language_ablation),
        ("exp3_visual_perturbation", exp3_visual_perturbation),
        ("exp4_temporal_perturbation", exp4_temporal_perturbation),
        ("exp5_combined_perturbation", exp5_combined_perturbation),
    ]
    
    all_results = {}
    for exp_name, exp_fn in experiments:
        print(f"\n{'='*40}")
        print(f"Running {exp_name}...")
        print(f"{'='*40}")
        t0 = time.time()
        try:
            result = exp_fn(model, processor, episodes)
            result["runtime_sec"] = round(time.time() - t0, 1)
            all_results[exp_name] = result
            
            # Save individual result
            out_file = OUTPUT_DIR / f"{exp_name}.json"
            out_file.write_text(json.dumps(result, indent=2, default=str))
            print(f"[{exp_name}] Done in {result['runtime_sec']}s → {out_file}")
        except Exception as e:
            print(f"[{exp_name}] FAILED: {e}")
            traceback.print_exc()
            all_results[exp_name] = {"error": str(e)}
    
    # Save combined results
    combined = {
        "experiments": all_results,
        "model": MODEL_ID,
        "device": DEVICE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "has_libero": has_libero,
    }
    (OUTPUT_DIR / "diagnosis_combined.json").write_text(json.dumps(combined, indent=2, default=str))
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
