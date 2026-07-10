# Exp2 Language Ablation — CORRECTION

## Original Claim (commit 775334e, FALSE)
"language affects actions"

## Actual Data
Actions at dropout 0%, 25%, 50%, 75% are **byte-identical**:
```
[-0.00020878732204438963, 0.004432274795630346, -0.006736875708927132,
 0.0004997134208677839, -0.024225862595964903, -0.05164424969986378, 0.11764705882352944]
```

Only at 100% dropout do actions change:
```
[0.020592917112743126, 0.029685540797079225, 0.03119761460186803,
 0.0816011804844816, 0.07759357288830414, 0.20301984594966835, 0.196078431372549]
```

## Correct Conclusion
Language conditioning has **zero measurable effect** on actions until completely removed (100% dropout).
At 0-75% dropout, the model produces byte-identical outputs — language is effectively ignored.
Only complete removal triggers a different (default) action output.

## Implication
This supports H2: OpenVLA may not meaningfully use language instructions during inference.
The model appears to rely primarily on visual input, with language as a no-op until absent.

## Caveat
This was a dry run with synthetic (repeated) images. Real LIBERO episodes needed for publishable results.
Same synthetic image means the model sees identical visual input — the only variable is language.
Byte-identical outputs across 0-75% dropout confirm language has no gradient-level influence.

## Date
2026-07-10
