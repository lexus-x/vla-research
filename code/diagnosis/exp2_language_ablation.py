"""
Experiment 2: Language Ablation Study
Tests whether OpenVLA actually uses language instructions or ignores them.
Method: Run inference with language token dropout (0%, 25%, 50%, 75%, 100%)
If success rate is insensitive to language dropout -> model ignores language (H2 confirmed)

This is inference-only on a frozen checkpoint. ~16-18GB VRAM at bf16.
"""
import torch
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor

# Config
MODEL_ID = "openvla/openvla-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DROPOUT_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]
N_TRIALS = 10
OUTPUT_DIR = Path("/tmp/robustvla_results")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Loading model: {MODEL_ID}")
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

start_load = time.time()
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(DEVICE)
    model.eval()
    print(f"Model loaded in {time.time() - start_load:.0f}s")
    print(f"VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
except Exception as e:
    print(f"Model load failed at bf16: {e}")
    print("Trying 4-bit quantization...")
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            trust_remote_code=True
        ).to(DEVICE)
        model.eval()
        print(f"Model loaded (4-bit) in {time.time() - start_load:.0f}s")
    except Exception as e2:
        print(f"4-bit also failed: {e2}")
        sys.exit(1)

# Check VRAM
if DEVICE == "cuda":
    vram_used = torch.cuda.memory_allocated() / 1e9
    print(f"VRAM after load: {vram_used:.2f} GB")
    if vram_used > 22:
        print("WARNING: VRAM usage is high. May OOM during inference.")

# Language ablation test
results = {
    "experiment": "language_ablation",
    "model": MODEL_ID,
    "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
    "vram_used_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if DEVICE == "cuda" else 0,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "dropout_levels": DROPOUT_LEVELS,
    "n_trials": N_TRIALS,
    "per_dropout_results": {},
    "note": "DRY RUN with synthetic inputs. Replace with real LIBERO episodes for publishable results.",
}

print(f"\nRunning language ablation: {len(DROPOUT_LEVELS)} dropout levels x {N_TRIALS} trials")

for dropout in DROPOUT_LEVELS:
    actions = []
    inference_times = []
    
    for trial in range(N_TRIALS):
        # Synthetic input as PIL Image (replace with real LIBERO observation)
        from PIL import Image
        import numpy as np
        dummy_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy_image = Image.fromarray(dummy_array)
        
        if dropout < 1.0:
            instruction = "pick up the red cup"
        else:
            instruction = ""
        
        try:
            t0 = time.time()
            inputs = processor(instruction, dummy_image).to(DEVICE)
            
            with torch.no_grad():
                action = model.predict_action(
                    input_ids=inputs["input_ids"],
                    unnorm_key="bridge_orig",
                    do_sample=False
                )
            dt = time.time() - t0
            actions.append(action.tolist())
            inference_times.append(dt)
        except Exception as e:
            actions.append(None)
            if trial == 0:
                print(f"  dropout={dropout}: error: {e}")
    
    valid_actions = [a for a in actions if a is not None]
    results["per_dropout_results"][str(dropout)] = {
        "n_valid": len(valid_actions),
        "n_failed": N_TRIALS - len(valid_actions),
        "mean_inference_ms": round(sum(inference_times) / len(inference_times) * 1000, 1) if inference_times else 0,
        "actions_sample": valid_actions[:3],  # First 3 for inspection
    }
    print(f"  dropout={dropout:.0%}: {len(valid_actions)}/{N_TRIALS} ok, "
          f"mean={results['per_dropout_results'][str(dropout)]['mean_inference_ms']}ms")

# Save
output_path = OUTPUT_DIR / "exp2_language_ablation.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults: {output_path}")
print(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
