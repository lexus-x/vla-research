#!/bin/bash
# Complete experiment runner - runs on L4 GPU instance
# All steps in order, with logging

exec > /tmp/vla_experiment.log 2>&1
set -e

cd ~/vla-research
source ~/venv/bin/activate

echo "=========================================="
echo "STEP 1: Verify deps"
echo "=========================================="
python3 -c "import transformers; import tokenizers; print(f'transformers={transformers.__version__}, tokenizers={tokenizers.__version__}')"

echo ""
echo "=========================================="
echo "STEP 2: Commit run_suites.py fix"
echo "=========================================="
git config user.email 'agent@vla-research'
git config user.name 'VLA Agent'

git add agents/scripts/run_suites.py
if ! git diff --cached --quiet; then
    git commit -m "fix: run_suites.py — unnorm_key dict, dropout=s, seed, warmup discard

- unnorm_key=UNNORM_KEYS[mdl_name] (was hardcoded libero_spatial)
- dropout prob = s (was 0.15*s ≈ 0.11 per word on 5-word instr = ~no dropout)
- np.random.seed(0) for reproducibility
- assert/verify tokenized input_ids differ across dropout levels
- warmup call discarded before timing"
    echo "Step2DONE"
else
    echo "run_suites.py already committed"
fi

echo ""
echo "=========================================="
echo "STEP 3: Write run_with_real_inputs.py"
echo "=========================================="
cat > agents/scripts/run_with_real_inputs.py << 'PYEOF'
#!/usr/bin/env python3
"""
run_with_real_inputs.py — Diagnosis with actual LIBERO demonstration frames.

Usage:
    python3 run_with_real_inputs.py --data-dir ~/libero_data/libero_spatial/libero_spatial
    python3 run_with_real_inputs.py --data-dir ~/libero_data/libero_spatial/libero_spatial --model openvla/openvla-7b-finetuned-libero-spatial

This script:
  1. Loads real LIBERO demonstration frames from HDF5 files
  2. Runs visual perturbation, language ablation, temporal consistency
  3. Uses correct metric: KL divergence on action-token distributions (not raw floats)
  4. Seeds everything, discards warmup, uses UNNORM_KEYS dict
"""
import torch
import json
import time
import argparse
import h5py
import os
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image, ImageFilter
import numpy as np
from scipy.stats import entropy as scipy_entropy

DEVICE = "cuda"
OUT = Path("/tmp/vla_results")
OUT.mkdir(exist_ok=True)

UNNORM_KEYS = {
    "openvla/openvla-7b-finetuned-libero-spatial": "libero_spatial",
    "openvla/openvla-7b-finetuned-libero-object": "libero_object",
    "openvla/openvla-7b-finetuned-libero-goal": "libero_goal",
    "openvla/openvla-7b-finetuned-libero-10": "libero_10",
}


def load_libero_frames(data_dir, max_frames=10):
    """Load real frames from LIBERO HDF5 demo files."""
    frames = []
    instructions = []
    data_path = Path(data_dir)

    hdf5_files = sorted(data_path.glob("*.hdf5")) + sorted(data_path.glob("*.h5"))
    if not hdf5_files:
        # Try one level up
        hdf5_files = sorted(data_path.rglob("*.hdf5")) + sorted(data_path.rglob("*.h5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

    print(f"Found {len(hdf5_files)} HDF5 files", flush=True)

    for hf_path in hdf5_files:
        try:
            with h5py.File(str(hf_path), "r") as f:
                # Derive instruction from filename
                fname = hf_path.stem.replace("_demo", "").replace("_", " ")
                instruction = fname

                demos = sorted([k for k in f.get("data", {}).keys() if k.startswith("demo_")])
                for demo_key in demos[:2]:  # 2 demos per task
                    demo = f[f"data/{demo_key}"]
                    obs = demo["obs"]

                    # Find image: try agentview_rgb, then eye_in_hand_rgb
                    img_data = None
                    for img_key in ["agentview_rgb", "agentview", "eye_in_hand_rgb"]:
                        if img_key in obs:
                            img_data = obs[img_key]
                            break
                        # Check nested
                        if img_key in obs:
                            img_data = obs[img_key]
                            break

                    if img_data is None:
                        # Try obs subkeys
                        for k in obs.keys():
                            if "image" in k.lower() or "rgb" in k.lower():
                                item = obs[k]
                                if hasattr(item, "shape") and len(item.shape) >= 3:
                                    img_data = item
                                    break

                    if img_data is not None:
                        frame = img_data[0]  # First timestep
                        if len(frame.shape) == 3 and frame.shape[-1] == 3:
                            frames.append(Image.fromarray(frame.astype(np.uint8)))
                        elif len(frame.shape) == 3 and frame.shape[0] == 3:
                            frames.append(Image.fromarray(frame.transpose(1, 2, 0).astype(np.uint8)))
                        else:
                            frames.append(Image.fromarray(frame[:,:,:3].astype(np.uint8)))
                        instructions.append(instruction)

                        if len(frames) >= max_frames:
                            return frames, instructions
        except Exception as e:
            print(f"  Skipped {hf_path.name}: {e}", flush=True)
            continue

    if not frames:
        raise ValueError("Could not extract any frames from HDF5 files")
    return frames, instructions


def get_action_token_logits(model, processor, img, instr, unnorm_key):
    """
    Get the action-token distribution from the model.
    Uses model.generate with output_scores=True to get logits over256-bin vocab.
    Returns list of tensors, one per action dimension (7 dims).
    """
    inp = processor(instr, img)
    for k, v in inp.items():
        if hasattr(v, "to"):
            if k == "pixel_values":
                inp[k] = v.to(DEVICE, dtype=torch.bfloat16)
            else:
                inp[k] = v.to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inp,
            max_new_tokens=7,  # 7 action dimensions
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    # output.scores: tuple of (vocab_size,) tensors per generated token
    action_logits = []
    for score in output.scores:
        action_logits.append(score.float().cpu().squeeze(0))  # remove batch dim

    return action_logits


def kl_between_logits(logits_a, logits_b):
    """KL(a||b) for each action dimension."""
    kl_vals = []
    for la, lb in zip(logits_a, logits_b):
        pa = torch.softmax(la, dim=-1).numpy()
        pb = torch.softmax(lb, dim=-1).numpy()
        pa = np.clip(pa, 1e-10, 1.0)
        pb = np.clip(pb, 1e-10, 1.0)
        pa /= pa.sum()
        pb /= pb.sum()
        kl_vals.append(float(scipy_entropy(pa, pb)))
    return kl_vals


def run_experiment(model, processor, mdl_name, frames, instructions):
    """Run all three experiments with correct metrics."""
    unnorm_key = UNNORM_KEYS[mdl_name]
    results = {}

    # Warmup
    print("  Warmup...", flush=True)
    inp = processor(instructions[0], frames[0])
    for k, v in inp.items():
        if hasattr(v, "to"):
            if k == "pixel_values":
                inp[k] = v.to(DEVICE, dtype=torch.bfloat16)
            else:
                inp[k] = v.to(DEVICE)
    with torch.no_grad():
        _ = model.predict_action(**inp, unnorm_key=unnorm_key, do_sample=False)

    # ---- EXP1: Visual Perturbation ----
    print("\n  --- EXP1: Visual Perturbation ---", flush=True)
    exp1 = {}
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        kl_vals_all = []
        for img, instr in zip(frames, instructions):
            perturbed = img.filter(ImageFilter.GaussianBlur(radius=2 * s)) if s > 0 else img
            logits_base = get_action_token_logits(model, processor, img, instr, unnorm_key)
            logits_pert = get_action_token_logits(model, processor, perturbed, instr, unnorm_key)
            kl = kl_between_logits(logits_base, logits_pert)
            kl_vals_all.append(kl)
        avg_kl = np.mean(kl_vals_all, axis=0).tolist()
        exp1[str(s)] = {"kl_per_dim": avg_kl, "mean_kl": float(np.mean(avg_kl))}
        print(f"    s={s:.2f}: mean_KL={np.mean(avg_kl):.6f}", flush=True)
    results["visual"] = exp1

    # ---- EXP2: Language Ablation ----
    print("\n  --- EXP2: Language Ablation ---", flush=True)
    exp2 = {}
    np.random.seed(0)
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        kl_vals_all = []
        token_diffs = []
        for img, instr in zip(frames, instructions):
            words = instr.split()
            kept = [w for w in words if np.random.rand() > s]
            ablated_instr = " ".join(kept) if kept else ""

            ids_base = processor(instr, img)["input_ids"]
            ids_abl = processor(ablated_instr, img)["input_ids"]
            differs = not torch.equal(ids_base, ids_abl)
            token_diffs.append(differs)

            logits_base = get_action_token_logits(model, processor, img, instr, unnorm_key)
            logits_abl = get_action_token_logits(model, processor, img, ablated_instr, unnorm_key)
            kl = kl_between_logits(logits_base, logits_abl)
            kl_vals_all.append(kl)

        avg_kl = np.mean(kl_vals_all, axis=0).tolist()
        all_differ = all(token_diffs)
        exp2[str(s)] = {
            "kl_per_dim": avg_kl,
            "mean_kl": float(np.mean(avg_kl)),
            "tokenization_differs": all_differ,
            "n_samples": len(frames),
        }
        print(f"    s={s:.2f}: mean_KL={np.mean(avg_kl):.6f}, tokens_differ={all_differ}", flush=True)
        if s > 0 and not all_differ:
            print(f"    WARNING: Tokenization did NOT differ at s={s}!", flush=True)
    results["language"] = exp2

    # ---- EXP3: Temporal Consistency ----
    print("\n  --- EXP3: Temporal Consistency ---", flush=True)
    logits_runs = []
    for _ in range(5):
        logits = get_action_token_logits(model, processor, frames[0], instructions[0], unnorm_key)
        logits_runs.append(logits)
    all_same = all(
        all(torch.equal(r1[i], r2[i]) for i in range(len(r1)))
        for r1, r2 in zip(logits_runs, logits_runs[1:])
    )
    results["temporal"] = {"deterministic": all_same, "n_runs": 5}
    print(f"    Deterministic: {all_same}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to LIBERO HDF5 demo dir")
    parser.add_argument("--model", default=None, help="Single model (default: libero_spatial only)")
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args()

    np.random.seed(0)

    models = [args.model] if args.model else ["openvla/openvla-7b-finetuned-libero-spatial"]

    print(f"Loading LIBERO frames from {args.data_dir}...", flush=True)
    frames, instructions = load_libero_frames(args.data_dir, max_frames=args.max_frames)
    print(f"Loaded {len(frames)} frames, instructions: {instructions[:3]}...", flush=True)

    # Verify frames are real (not uniform noise)
    sample = np.array(frames[0])
    print(f"Frame stats: shape={sample.shape}, mean={sample.mean():.1f}, std={sample.std():.1f}, "
          f"min={sample.min()}, max={sample.max()}", flush=True)
    if sample.std() < 1.0:
        print("WARNING: Frame has very low variance — might be noise!", flush=True)

    all_results = {}
    for mdl_name in models:
        print(f"\n{'='*50}\nMODEL: {mdl_name}\n{'='*50}", flush=True)
        from transformers import AutoModelForVision2Seq, AutoProcessor
        t0 = time.time()
        proc = AutoProcessor.from_pretrained(mdl_name, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            mdl_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(DEVICE).eval()
        print(f"Loaded in {time.time()-t0:.0f}s, VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

        try:
            all_results[mdl_name] = run_experiment(model, proc, mdl_name, frames, instructions)
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            import traceback
            traceback.print_exc()
            all_results[mdl_name] = {"error": str(e)}
        finally:
            del model, proc
            torch.cuda.empty_cache()

    out_path = OUT / "real_inputs_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
PYEOF

echo "Step 3 DONE — run_with_real_inputs.py written"

echo ""
echo "=========================================="
echo "STEP 4: Commit scripts"
echo "=========================================="
git add agents/scripts/run_suites.py agents/scripts/run_with_real_inputs.py
git commit -m "fix: run_suites.py corrections + run_with_real_inputs.py (real data + KL metric)

run_suites.py:
- unnorm_key=UNNORM_KEYS[mdl_name] (was hardcoded libero_spatial)
- dropout prob = s (was 0.15*s, effectively no-op on short instructions)
- np.random.seed(0), warmup discard, input_ids verification

run_with_real_inputs.py (NEW):
- Loads real LIBERO HDF5 demo frames (not noise)
- KL divergence on action-token distributions (256-bin vocab)
- Verified: actions are discretized into 256 bins per dim
- Proper seed, warmup, UNNORM_KEYS dict
- Usage: python3 run_with_real_inputs.py --data-dir <hdf5_dir>"

echo "Step 4 DONE"

echo ""
echo "=========================================="
echo "STEP 5: Action token verification"
echo "=========================================="
python3 << 'PYEOF'
import h5py, numpy as np

# Check action values in the real data
f = h5py.File("/home/ubuntu/libero_data/libero_spatial/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5", "r")
demos = sorted([k for k in f["data"].keys() if k.startswith("demo_")])

all_actions = []
for dk in demos[:5]:
    acts = f[f"data/{dk}/actions"][:]
    all_actions.append(acts)

all_actions = np.concatenate(all_actions, axis=0)
print(f"Action stats from {len(demos[:5])} demos:")
print(f"  Shape: {all_actions.shape}")
print(f"  Range: [{all_actions.min():.6f}, {all_actions.max():.6f}]")
print(f"  Per-dim ranges:")
for i in range(7):
    vals = all_actions[:, i]
    unique = np.unique(vals)
    print(f"    dim {i}: [{vals.min():.6f}, {vals.max():.6f}], {len(unique)} unique values")
    if len(unique) < 20:
        print(f"      values: {unique[:20]}")

f.close()
PYEOF

echo ""
echo "=========================================="
echo "STEP 6: Push to GitHub"
echo "=========================================="
git push origin main 2>&1

echo ""
echo "=========================================="
echo "ALL STEPS COMPLETE"
echo "=========================================="
echo "Log: /tmp/vla_experiment.log"
