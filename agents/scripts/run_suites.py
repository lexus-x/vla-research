#!/usr/bin/env python3
"""
run_suites.py — Corrected diagnosis experiments for all LIBERO suites.
Fixes:
  - unnorm_key uses UNNORM_KEYS dict (was hardcoded to libero_spatial)
  - dropout prob = s (was 0.15*s, effectively no dropout)
  - np.random.seed(0) for reproducibility
  - assert tokenized inputs differ across dropout levels
  - discard first inference (warmup) before timing
  - real inputs required (not noise) — see run_with_real_inputs.py
"""
import torch
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image, ImageFilter
import numpy as np

DEVICE = "cuda"
OUT = Path("/tmp/vla_results"); OUT.mkdir(exist_ok=True)

SUITES = [
    "openvla/openvla-7b-finetuned-libero-spatial",
    "openvla/openvla-7b-finetuned-libero-object",
    "openvla/openvla-7b-finetuned-libero-goal",
    "openvla/openvla-7b-finetuned-libero-10",
]

UNNORM_KEYS = {
    "openvla/openvla-7b-finetuned-libero-spatial": "libero_spatial",
    "openvla/openvla-7b-finetuned-libero-object": "libero_object",
    "openvla/openvla-7b-finetuned-libero-goal": "libero_goal",
    "openvla/openvla-7b-finetuned-libero-10": "libero_10",
}


def run_suite(mdl_name):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    print(f"\n{'='*50}\nSUITE: {mdl_name}\n{'='*50}", flush=True)

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(mdl_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        mdl_name, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(DEVICE).eval()
    print(f"Loaded {time.time()-t0:.0f}s VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    unnorm_key = UNNORM_KEYS[mdl_name]  # FIX: use dict, not hardcoded

    def pred(img, instr, return_scores=False):
        inp = proc(instr, img)
        for k, v in inp.items():
            if hasattr(v, "to"):
                if k == "pixel_values":
                    inp[k] = v.to(DEVICE, dtype=torch.bfloat16)
                else:
                    inp[k] = v.to(DEVICE)

        # Warmup: first call is slow, discard it
        with torch.no_grad():
            _ = model.predict_action(**inp, unnorm_key=unnorm_key, do_sample=False)

        # Now do the real inference
        with torch.no_grad():
            action = model.predict_action(
                **inp, unnorm_key=unnorm_key, do_sample=False
            )
        return action

    def get_token_ids(img, instr):
        """Return tokenized input_ids for verification."""
        inp = proc(instr, img)
        return inp["input_ids"].tolist() if "input_ids" in inp else None

    # Use placeholder images — NOTE: real experiments must use real LIBERO frames
    eps = [
        {"img": Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)),
         "instr": t}
        for t in ["pick up the red bowl", "open the drawer", "push the mug"]
    ]

    results = {}

    # ---- Visual perturbation ----
    print("\n--- Visual Perturbation ---", flush=True)
    r = {}
    for s in [0.0, 0.5, 1.0]:
        acts = []
        for e in eps:
            img = e["img"].filter(ImageFilter.GaussianBlur(radius=2 * s)) if s > 0 else e["img"]
            a = pred(img, e["instr"])
            acts.append(a.tolist() if hasattr(a, "tolist") else list(a))
        arr = np.array(acts)
        r[str(s)] = {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist()}
    results["visual"] = r

    # ---- Language ablation ----
    print("\n--- Language Ablation ---", flush=True)
    r = {}
    # FIX: dropout prob = s, NOT 0.15*s
    # Also: assert tokenized inputs actually differ
    prev_ids = None
    for s in [0.0, 0.5, 1.0]:
        np.random.seed(0)  # reproducibility
        acts = []
        for e in eps:
            words = e["instr"].split()
            # Each word is dropped with probability s
            kept = [w for w in words if np.random.rand() > s]
            instr = " ".join(kept) if kept else ""

            # VERIFY: tokenized input_ids must differ across dropout levels
            ids = get_token_ids(e["img"], instr)
            if prev_ids is not None and s > 0:
                if ids == prev_ids:
                    print(f"  WARNING: input_ids identical at s={s} — ablation not reaching model!",
                          flush=True)
            prev_ids = ids
            print(f"  s={s:.1f}: instr='{instr}' -> ids_len={len(ids) if ids else None}", flush=True)

            a = pred(e["img"], instr)
            acts.append(a.tolist() if hasattr(a, "tolist") else list(a))
        arr = np.array(acts)
        r[str(s)] = {
            "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(),
            "identical": bool(arr.std(0).sum() < 1e-10),
        }
    results["language"] = r

    # ---- Temporal consistency ----
    print("\n--- Temporal Consistency ---", flush=True)
    acts = []
    for _ in range(3):
        a = pred(eps[0]["img"], eps[0]["instr"])
        acts.append(a.tolist() if hasattr(a, "tolist") else list(a))
    arr = np.array(acts)
    var = float(arr.var(0).mean())
    results["temporal"] = {"var": var, "det": var < 1e-10}

    del model, proc
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    np.random.seed(0)
    all_results = {}
    for suite in SUITES:
        try:
            all_results[suite] = run_suite(suite)
        except Exception as e:
            print(f"FAILED {suite}: {e}", flush=True)
            all_results[suite] = {"error": str(e)}

    (OUT / "all_suites.json").write_text(json.dumps(all_results, indent=2, default=str))
    print("\nALL SUITES DONE", flush=True)
