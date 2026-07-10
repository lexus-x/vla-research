#!/usr/bin/env python3
"""
run_with_real_inputs.py - Diagnosis with real LIBERO demonstration frames.
Loads actual HDF5 demo data, runs perturbation/ablation with KL metric.
"""
import torch, json, time, argparse, h5py, os
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image, ImageFilter
import numpy as np
from scipy.stats import entropy as scipy_entropy

DEVICE = "cuda"
OUT = Path("/tmp/vla_results"); OUT.mkdir(exist_ok=True)
UNNORM_KEYS = {
    "openvla/openvla-7b-finetuned-libero-spatial": "libero_spatial",
    "openvla/openvla-7b-finetuned-libero-object": "libero_object",
    "openvla/openvla-7b-finetuned-libero-goal": "libero_goal",
    "openvla/openvla-7b-finetuned-libero-10": "libero_10",
}

def load_libero_frames(data_dir, max_frames=10):
    frames, instructions = [], []
    hdf5_files = sorted(Path(data_dir).glob("*.hdf5")) + sorted(Path(data_dir).rglob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files in {data_dir}")
    print(f"Found {len(hdf5_files)} HDF5 files", flush=True)
    for hf in hdf5_files:
        try:
            with h5py.File(str(hf), "r") as f:
                fname = hf.stem.replace("_demo","").replace("_"," ")
                demos = sorted([k for k in f.get("data",{}).keys() if k.startswith("demo_")])
                for dk in demos[:2]:
                    obs = f[f"data/{dk}/obs"]
                    img_data = None
                    for ik in ["agentview_rgb","agentview","eye_in_hand_rgb"]:
                        if ik in obs:
                            img_data = obs[ik]; break
                    if img_data is None:
                        for k in obs.keys():
                            if "image" in k.lower() or "rgb" in k.lower():
                                if hasattr(obs[k],"shape") and len(obs[k].shape)>=3:
                                    img_data = obs[k]; break
                    if img_data is not None:
                        fr = img_data[0]
                        if len(fr.shape)==3 and fr.shape[-1]==3:
                            frames.append(Image.fromarray(fr.astype(np.uint8)))
                        elif len(fr.shape)==3 and fr.shape[0]==3:
                            frames.append(Image.fromarray(fr.transpose(1,2,0).astype(np.uint8)))
                        instructions.append(fname)
                        if len(frames)>=max_frames: return frames, instructions
        except Exception as e:
            print(f"  Skipped {hf.name}: {e}", flush=True)
    if not frames: raise ValueError("No frames extracted")
    return frames, instructions

def get_action_logits(model, proc, img, instr, unnorm_key):
    inp = proc(instr, img)
    for k,v in inp.items():
        if hasattr(v,"to"):
            inp[k] = v.to(DEVICE, dtype=torch.bfloat16) if k=="pixel_values" else v.to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=7, do_sample=False,
                             output_scores=True, return_dict_in_generate=True)
    return [s.float().cpu().squeeze(0) for s in out.scores]

def kl_logits(la, lb):
    kl = []
    for a,b in zip(la,lb):
        pa = np.clip(torch.softmax(a,-1).numpy(), 1e-10, 1.0)
        pb = np.clip(torch.softmax(b,-1).numpy(), 1e-10, 1.0)
        pa /= pa.sum(); pb /= pb.sum()
        kl.append(float(scipy_entropy(pa, pb)))
    return kl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model", default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--max-frames", type=int, default=10)
    args = parser.parse_args()
    np.random.seed(0)

    print(f"Loading from {args.data_dir}...", flush=True)
    frames, instrs = load_libero_frames(args.data_dir, args.max_frames)
    print(f"Loaded {len(frames)} frames", flush=True)
    sample = np.array(frames[0])
    print(f"Frame: shape={sample.shape}, mean={sample.mean():.1f}, std={sample.std():.1f}", flush=True)
    if sample.std() < 1.0:
        print("WARNING: very low variance - might be noise!", flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    unnorm_key = UNNORM_KEYS[args.model]
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                    trust_remote_code=True).to(DEVICE).eval()
    print(f"Loaded in {time.time()-t0:.0f}s, VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # Warmup
    inp = proc(instrs[0], frames[0])
    for k,v in inp.items():
        if hasattr(v,"to"): inp[k] = v.to(DEVICE, dtype=torch.bfloat16) if k=="pixel_values" else v.to(DEVICE)
    with torch.no_grad(): _ = model.predict_action(**inp, unnorm_key=unnorm_key, do_sample=False)

    results = {}

    # EXP1: Visual Perturbation
    print("\n--- EXP1: Visual Perturbation ---", flush=True)
    exp1 = {}
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        kls = []
        for img, instr in zip(frames, instrs):
            p = img.filter(ImageFilter.GaussianBlur(radius=2*s)) if s>0 else img
            lb = get_action_logits(model, proc, img, instr, unnorm_key)
            lp = get_action_logits(model, proc, p, instr, unnorm_key)
            kls.append(kl_logits(lb, lp))
        avg = np.mean(kls, axis=0).tolist()
        exp1[str(s)] = {"kl_per_dim": avg, "mean_kl": float(np.mean(avg))}
        print(f"  s={s:.2f}: mean_KL={np.mean(avg):.6f}", flush=True)
    results["visual"] = exp1

    # EXP2: Language Ablation
    print("\n--- EXP2: Language Ablation ---", flush=True)
    exp2 = {}
    np.random.seed(0)
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        kls, diffs = [], []
        for img, instr in zip(frames, instrs):
            words = instr.split()
            kept = [w for w in words if np.random.rand() > s]
            abl = " ".join(kept) if kept else ""
            ids_b = proc(instr, img)["input_ids"]
            ids_a = proc(abl, img)["input_ids"]
            diffs.append(not torch.equal(ids_b, ids_a))
            lb = get_action_logits(model, proc, img, instr, unnorm_key)
            la = get_action_logits(model, proc, img, abl, unnorm_key)
            kls.append(kl_logits(lb, la))
        avg = np.mean(kls, axis=0).tolist()
        all_d = all(diffs)
        exp2[str(s)] = {"kl_per_dim": avg, "mean_kl": float(np.mean(avg)),
                        "tokens_differ": all_d, "n": len(frames)}
        print(f"  s={s:.2f}: mean_KL={np.mean(avg):.6f}, tokens_differ={all_d}", flush=True)
        if s>0 and not all_d:
            print(f"  WARNING: tokens identical at s={s}!", flush=True)
    results["language"] = exp2

    # EXP3: Temporal
    print("\n--- EXP3: Temporal ---", flush=True)
    runs = [get_action_logits(model, proc, frames[0], instrs[0], unnorm_key) for _ in range(5)]
    same = all(all(torch.equal(a,b) for a,b in zip(r1,r2)) for r1,r2 in zip(runs,runs[1:]))
    results["temporal"] = {"deterministic": same, "n_runs": 5}
    print(f"  Deterministic: {same}", flush=True)

    out = OUT / "real_inputs_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults: {out}", flush=True)

if __name__ == "__main__":
    main()
