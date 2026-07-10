#!/usr/bin/env python3
"""
Agent 5: Red-Team Critic
Adversarial review of all proposals, methods, and results.
Runs on control machine.
"""
import json
import time
from pathlib import Path
from datetime import datetime, timezone

RESULTS_DIR = Path("/tmp/vla_agents/results/red_team")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ATTACK_QUESTIONS = {
    "perturbvla": [
        "Is adversarial perturbation training actually novel, or just data augmentation with a fancy name?",
        "LIBERO-PRO showed 0% under perturbation — but is that because the test perturbations were adversarial, not because the model memorized?",
        "If you add perturbation augmentation during training, isn't that just domain randomization (a well-known technique)?",
        "The contrastive loss assumes perturbed versions of the same trajectory should have similar representations. Is that actually true?",
        "How do you distinguish 'the model is robust' from 'the perturbations weren't strong enough'?",
        "If PerturbVLA works on OpenVLA, does it work on ANY VLA? Or is this model-specific?",
        "LIBERO-PRO uses specific perturbation types. What if real-world perturbations are completely different?",
        "Zero extra params at inference — but how much extra training time? If it's 10x slower to train, is it practical?",
        "The claim 'maintain >80% SR under perturbation' is based on what evidence? Has any experiment been run?",
        "If the model's actions are byte-identical across language dropout levels (exp2 data), does language perturbation even matter?",
    ],
    "mambaflow": [
        "Mamba-2 already exists. Is this just Mamba-2 + flow matching bolted together?",
        "If inference is 3-5x faster but SR drops by 5%, is that a win?",
        "The claim 'first SSM-based VLA with flow matching' — has anyone actually tried and failed? Or just hasn't bothered?",
        "Mamba's O(n) is for long sequences. VLA input sequences are ~200 tokens. Does O(n) vs O(n²) even matter?",
        "Flow matching needs 4-8 denoising steps. Each step is a forward pass. Is the bottleneck really the backbone?",
        "If SmolVLA at 450M is already fast, does MambaFlow at 300-500M actually improve anything?",
        "RoboMamba exists. AnoleVLA exists. What specifically is different beyond bolting on flow matching?",
        "Training a new backbone from scratch needs massive compute. Is 1x L40S enough?",
    ],
    "general": [
        "Every idea claims 'novel' — but novelty ≠ value. What's the actual impact?",
        "The field is moving fast. By the time this is published, will it matter?",
        "Are we optimizing for benchmark performance or real-world applicability?",
        "LIBERO is 'essentially solved'. Why are we still using it as the primary benchmark?",
        "All experiments are simulation-only. Does any of this transfer to real robots?",
    ]
}

def run_critique(idea_name: str, questions: list):
    """Generate adversarial critique for an idea."""
    critique = {
        "idea": idea_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(questions),
        "questions": questions,
        "severity_ratings": [],
        "kill_questions": [],  # Questions that could kill the idea
    }
    
    for q in questions:
        # Rate severity (simulated — in practice, use LLM)
        severity = "high" if any(kw in q.lower() for kw in ["actually", "just", "if", "what if", "hasn't", "does"]) else "medium"
        if "kill" in q.lower() or "novel ≠" in q.lower():
            severity = "critical"
        critique["severity_ratings"].append({"question": q, "severity": severity})
        if severity == "critical":
            critique["kill_questions"].append(q)
    
    return critique

def run():
    print("[RedTeam] Starting adversarial review...")
    while True:
        all_critiques = {}
        
        for idea, questions in ATTACK_QUESTIONS.items():
            print(f"  Critiquing: {idea}...")
            critique = run_critique(idea, questions)
            all_critiques[idea] = critique
            
            n_kill = len(critique["kill_questions"])
            if n_kill > 0:
                print(f"  ⚠️  {idea}: {n_kill} kill questions identified")
        
        # Summary
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ideas_reviewed": len(all_critiques),
            "total_questions": sum(c["n_questions"] for c in all_critiques.values()),
            "total_kill_questions": sum(len(c["kill_questions"]) for c in all_critiques.values()),
            "critiques": all_critiques,
        }
        
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        (RESULTS_DIR / f"critique_{ts}.json").write_text(json.dumps(summary, indent=2))
        (RESULTS_DIR / "latest.json").write_text(json.dumps(summary, indent=2))
        
        print(f"[RedTeam] Review complete: {summary['total_kill_questions']} kill questions across {summary['ideas_reviewed']} ideas")
        
        # Sleep 2 hours
        time.sleep(7200)

if __name__ == "__main__":
    run()
