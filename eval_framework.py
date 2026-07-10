"""
VLA Research Evaluation Framework
Evaluates candidate ideas against constraint gates.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CandidateIdea:
    name: str
    description: str
    category: str  # "standalone_model" | "module" | "action_head" | "training_method"
    origin_domain: str  # where the idea comes from
    
    # Gate 1: Novelty
    novelty_score: float = 0.0  # 0-10
    novelty_evidence: list = field(default_factory=list)
    similar_papers: list = field(default_factory=list)
    
    # Gate 2: Feasibility
    estimated_params: int = 0  # total params
    module_params: int = 0  # if plug-in, module params only
    fits_l40s: bool = False  # can train on 1x L40S
    training_time_hours: float = 0.0
    
    # Gate 3: Performance
    expected_sr_improvement: float = 0.0  # % over baseline
    expected_speed_improvement: float = 0.0  # x faster
    expected_size_reduction: float = 0.0  # x smaller
    expected_robustness_improvement: float = 0.0  # %
    baselines_to_beat: list = field(default_factory=list)
    
    # Gate 4: Quality
    novelty_significance: float = 0.0  # 0-10
    rigor_score: float = 0.0  # 0-10
    impact_score: float = 0.0  # 0-10
    
    # Computed
    weighted_score: float = 0.0
    passed_gates: bool = False
    gate_failures: list = field(default_factory=list)

CONSTRAINT_GATES = {
    "novelty_min": 8.0,
    "sr_improvement_min": 5.0,  # %
    "size_reduction_min": 10.0,  # x smaller than 7B baseline
    "speed_improvement_min": 2.0,  # x faster
    "max_params_standalone": 500_000_000,
    "max_params_module": 10_000_000,
    "pass_threshold": 7.5,
}

BASELINES = {
    "OpenVLA": {"params": 7_000_000_000, "libero_sr": 0.85},
    "Octo": {"params": 93_000_000, "libero_sr": 0.72},
    "pi_0": {"params": 3_000_000_000, "libero_sr": 0.88},
}

def evaluate_gate1(idea: CandidateIdea) -> tuple[bool, list]:
    """Novelty gate"""
    failures = []
    if idea.novelty_score < CONSTRAINT_GATES["novelty_min"]:
        failures.append(f"Novelty {idea.novelty_score} < {CONSTRAINT_GATES['novelty_min']}")
    if len(idea.similar_papers) > 0:
        failures.append(f"Similar papers found: {idea.similar_papers}")
    return len(failures) == 0, failures

def evaluate_gate2(idea: CandidateIdea) -> tuple[bool, list]:
    """Feasibility gate"""
    failures = []
    if idea.category == "standalone_model":
        if idea.estimated_params > CONSTRAINT_GATES["max_params_standalone"]:
            failures.append(f"Params {idea.estimated_params} > {CONSTRAINT_GATES['max_params_standalone']}")
    elif idea.category == "module":
        if idea.module_params > CONSTRAINT_GATES["max_params_module"]:
            failures.append(f"Module params {idea.module_params} > {CONSTRAINT_GATES['max_params_module']}")
    if not idea.fits_l40s:
        failures.append("Cannot train on 1x L40S")
    return len(failures) == 0, failures

def evaluate_gate3(idea: CandidateIdea) -> tuple[bool, list]:
    """Performance gate"""
    failures = []
    if idea.expected_sr_improvement < CONSTRAINT_GATES["sr_improvement_min"]:
        failures.append(f"SR improvement {idea.expected_sr_improvement}% < {CONSTRAINT_GATES['sr_improvement_min']}%")
    return len(failures) == 0, failures

def evaluate_gate4(idea: CandidateIdea) -> tuple[bool, list]:
    """Quality gate"""
    failures = []
    if idea.novelty_significance < 7.0:
        failures.append(f"Novelty significance {idea.novelty_significance} < 7.0")
    if idea.rigor_score < 7.0:
        failures.append(f"Rigor score {idea.rigor_score} < 7.0")
    return len(failures) == 0, failures

def compute_weighted_score(idea: CandidateIdea) -> float:
    """Compute weighted score (0-10)"""
    weights = {
        "novelty": 0.30,
        "performance": 0.25,
        "size": 0.20,
        "speed": 0.15,
        "reproducibility": 0.10,
    }
    scores = {
        "novelty": idea.novelty_score,
        "performance": min(10, idea.expected_sr_improvement / 2),  # 20% improvement = 10
        "size": min(10, idea.expected_size_reduction / 10),  # 100x smaller = 10
        "speed": min(10, idea.expected_speed_improvement / 5),  # 5x faster = 10
        "reproducibility": idea.rigor_score,
    }
    return sum(weights[k] * scores[k] for k in weights)

def evaluate_candidate(idea: CandidateIdea) -> CandidateIdea:
    """Run all gates and compute final score"""
    gates = [
        ("Novelty", evaluate_gate1),
        ("Feasibility", evaluate_gate2),
        ("Performance", evaluate_gate3),
        ("Quality", evaluate_gate4),
    ]
    
    all_passed = True
    for gate_name, gate_fn in gates:
        passed, failures = gate_fn(idea)
        if not passed:
            all_passed = False
            idea.gate_failures.extend([f"[{gate_name}] {f}" for f in failures])
    
    idea.weighted_score = compute_weighted_score(idea)
    idea.passed_gates = all_passed and idea.weighted_score >= CONSTRAINT_GATES["pass_threshold"]
    return idea

def rank_candidates(ideas: list[CandidateIdea]) -> list[CandidateIdea]:
    """Rank passed candidates by weighted score"""
    evaluated = [evaluate_candidate(idea) for idea in ideas]
    passed = [i for i in evaluated if i.passed_gates]
    failed = [i for i in evaluated if not i.passed_gates]
    
    passed.sort(key=lambda x: x.weighted_score, reverse=True)
    return passed, failed

if __name__ == "__main__":
    # Example usage
    idea = CandidateIdea(
        name="SparseMoE-ActionHead",
        description="Mixture-of-Experts action head with top-2 routing",
        category="module",
        origin_domain="NLP (Switch Transformer)",
        novelty_score=9.0,
        estimated_params=0,
        module_params=5_000_000,
        fits_l40s=True,
        expected_sr_improvement=8.0,
        expected_speed_improvement=3.0,
        expected_size_reduction=1.0,
        novelty_significance=8.5,
        rigor_score=8.0,
    )
    result = evaluate_candidate(idea)
    print(f"Idea: {result.name}")
    print(f"Passed: {result.passed_gates}")
    print(f"Score: {result.weighted_score:.2f}")
    if result.gate_failures:
        print(f"Failures: {result.gate_failures}")
