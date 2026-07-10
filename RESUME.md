# RESUME.md — Project Status

## ⚠️ ZERO Valid Experimental Results Exist

**This project has NO valid experimental results as of this writing.** All prior runs used synthetic random noise inputs (`np.random.randint` / random tensors), not real LIBERO HDF5 demonstration frames. Any metrics from those runs are meaningless and must not be cited or relied upon.

Real LIBERO HDF5 data files must be obtained and placed before any valid experiment can produce publishable results. The guard in `agents/scripts/run_suites.py` (line 41: `raise RuntimeError("No real LIBERO frames. Refusing to run on synthetic noise — see RESUME.md")`) enforces this and must not be bypassed.

### What was done this session:
1. **Deps fixed**: transformers 4.40.1, tokenizers 0.19.1 installed and verified on L4
2. **Deleted results/real_diagnosis/**: All noise-input results removed (git committed)
3. **Fixed agents/scripts/run_suites.py**: (git committed)
   - unnorm_key=UNNORM_KEYS[mdl_name] (was hardcoded libero_spatial)
   - dropout prob = s (was 0.15*s ≈ 0.11 per word = no dropout on 5-word instructions)
   - np.random.seed(0), warmup discard, input_ids verification added
4. **Created agents/scripts/run_with_real_inputs.py**: (git committed)
   - Loads real LIBERO HDF5 demo frames
   - KL divergence on action-token distributions (256-bin vocab)
   - Correct metric: model.generate(output_scores=True) -> softmax -> KL
5. **LIBERO data downloaded**: libero_spatial (10 HDF5 demos, 5.9GB) at ~/libero_data/libero_spatial/libero_spatial/

### Action data verified:
- Actions: shape (T, 7), float64 continuous values
- dims 0-5: continuous (57-126 unique values per dimension)
- dim6: binary gripper {-1.0, 1.0}
- OpenVLA discretizes these into 256 bins at inference time

### What remains:
- **Run the experiment**: `python3 agents/scripts/run_with_real_inputs.py --data-dir ~/libero_data/libero_spatial/libero_spatial`
- **Verify256-bin action tokenizer**: Check prismatic/vla/action_tokenizer.py after first model load
- **Run across all4suites**: Need libero_object, libero_goal, libero_10 data downloaded
- **Compare with baselines**: RobustVLA (github.com/gakakulicc/RobustVLA), RoVLA (github.com/HCPLab-SYSU/RoVLA)

### Instance state:
- Instance: i-0d86ee8a3dff3d6d1 (g6.xlarge, L424GB)
- IP: 44.234.88.211
- Region: us-west-2d
- SSH key: vla-research-key (not available in this session, used EC2 Instance Connect)

### Critical notes:
- **ZERO valid experimental results exist.** All prior runs used synthetic random noise, not real LIBERO frames.
- Previous results from noise inputs are DELETED. Do not reference them.
- The guard in `agents/scripts/run_suites.py` refuses to run on synthetic data. Do not bypass it.
- The "language ignored" finding was an artifact of 0.15*s dropout (≈0.11 per word)
- The "variance=0.0" was a tautology (do_sample=False on identical input)
- run_suites.py unnorm fix was broken (defined dict then hardcoded value)
