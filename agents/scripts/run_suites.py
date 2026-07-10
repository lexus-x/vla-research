#!/usr/bin/env python3
import torch, json, time
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

def run_suite(mdl_name):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    print(f"\n{'='*50}\nSUITE: {mdl_name}\n{'='*50}", flush=True)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(mdl_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(mdl_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE).eval()
    print(f"Loaded {time.time()-t0:.0f}s VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    unnorm = mdl_name.split("/")[-1].replace("openvla-7b-finetuned-", "")
    def pred(img, instr):
        inp = proc(instr, img)
        for k, v in inp.items():
            if hasattr(v, "to"):
                inp[k] = v.to(DEVICE, dtype=torch.bfloat16) if k == "pixel_values" else v.to(DEVICE)
        with torch.no_grad():
            return model.predict_action(**inp, unnorm_key=unnorm, do_sample=False)
    eps = [{"img": Image.fromarray(np.random.randint(0,255,(128,128,3),dtype=np.uint8)), "i": t} for t in ["pick up the red bowl","open the drawer","push the mug"]]
    results = {}
    r = {}
    for s in [0.0, 0.5, 1.0]:
        acts = []
        for e in eps:
            img = e["img"].filter(ImageFilter.GaussianBlur(radius=2*s)) if s > 0 else e["img"]
            a = pred(img, e["i"])
            acts.append(a.tolist() if hasattr(a,"tolist") else list(a))
        arr = np.array(acts)
        r[str(s)] = {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist()}
    results["visual"] = r
    r = {}
    for s in [0.0, 0.5, 1.0]:
        acts = []
        for e in eps:
            instr = e["i"] if s == 0 else ("" if s >= 1 else " ".join(w for i,w in enumerate(e["i"].split()) if np.random.rand() > 0.15*s))
            a = pred(e["img"], instr)
            acts.append(a.tolist() if hasattr(a,"tolist") else list(a))
        arr = np.array(acts)
        r[str(s)] = {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(), "identical": bool(arr.std(0).sum() < 1e-10)}
    results["language"] = r
    acts = []
    for _ in range(3):
        a = pred(eps[0]["img"], eps[0]["i"])
        acts.append(a.tolist() if hasattr(a,"tolist") else list(a))
    arr = np.array(acts)
    var = float(arr.var(0).mean())
    results["temporal"] = {"var": var, "det": var < 1e-10}
    del model, proc
    torch.cuda.empty_cache()
    return results

all_results = {}
for suite in SUITES:
    try:
        all_results[suite] = run_suite(suite)
    except Exception as e:
        print(f"FAILED {suite}: {e}", flush=True)
        all_results[suite] = {"error": str(e)}

(OUT/"all_suites.json").write_text(json.dumps(all_results, indent=2, default=str))
print("\nALL SUITES DONE", flush=True)
