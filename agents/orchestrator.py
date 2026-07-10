#!/usr/bin/env python3
"""
Multi-Agent VLA Research Orchestrator
=====================================
Dynamic workflow manager that spawns, monitors, and coordinates agents.
Runs on control machine; dispatches GPU work to L4/L40S via SSH.

Agents:
  1. Literature Hunter   — continuous literature scanning
  2. Novelty Auditor     — novelty verification for ideas
  3. Systems Architect   — architecture design + ablation plans
  4. Benchmark Engineer  — benchmark setup + data pipeline
  5. Red-Team Critic      — adversarial review of all proposals
  6. Implementation Eng  — code implementation
  7. Training Engineer   — training pipeline + monitoring
  8. Evaluation Engineer — eval pipeline + metrics
  9. Paper Reviewer      — simulate reviewer feedback
  10. Ranking Committee  — score + rank ideas

GPU Allocation:
  L4  (24GB, 15.13GB VRAM) — inference-only: diagnosis exps, eval, baseline repro
  L40S (48GB) — LoRA fine-tuning (deferred until diagnosis complete)
"""

import json
import time
import subprocess
import threading
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from enum import Enum

# ============================================================
# Configuration
# ============================================================

REPO_ROOT = Path("/home/work/.openclaw/workspace/vla-research")
AGENTS_DIR = REPO_ROOT / "agents"
RESULTS_DIR = REPO_ROOT / "results"
SSH_KEY = Path.home() / ".ssh" / "vla-temp"

GPU_HOSTS = {
    "l4": {"ip": "44.234.88.211", "user": "ubuntu", "instance_id": "i-0d86ee8a3dff3d6d1",
           "vram_gb": 24, "role": "inference"},
    "l40s": {"ip": "34.211.170.34", "user": "ubuntu", "instance_id": "i-011cf5208ba58a803",
             "vram_gb": 48, "role": "finetuning"},
}

GITHUB_REPO = "https://admin-sai@github.com/lexus-x/vla-research.git"
PUSH_INTERVAL_MIN = 30

class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"

@dataclass
class AgentStatus:
    agent_id: str
    name: str
    state: AgentState = AgentState.IDLE
    current_task: str = ""
    gpu: str = ""
    progress: float = 0.0
    last_update: str = ""
    error: str = ""
    output_file: str = ""

@dataclass
class OrchestratorState:
    agents: Dict[str, AgentStatus] = field(default_factory=dict)
    gpu_utilization: Dict[str, float] = field(default_factory=dict)
    active_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    last_push: str = ""
    start_time: str = ""

# ============================================================
# SSH Executor
# ============================================================

def ssh_exec(gpu: str, command: str, timeout: int = 300) -> tuple:
    """Execute command on remote GPU via SSH. Returns (stdout, stderr, returncode)."""
    host = GPU_HOSTS[gpu]
    # Re-send Instance Connect key (valid 60s)
    ic_cmd = (
        f"aws ec2-instance-connect send-ssh-public-key "
        f"--region us-west-2 --instance-id {host['instance_id']} "
        f"--instance-os-user {host['user']} "
        f"--ssh-public-key file://{SSH_KEY}.pub"
    )
    subprocess.run(ic_cmd, shell=True, capture_output=True, timeout=15)

    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-i", str(SSH_KEY),
        f"{host['user']}@{host['ip']}",
        command
    ]
    try:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "SSH timeout", 1

def scp_upload(gpu: str, local: str, remote: str) -> bool:
    """Upload file to remote GPU."""
    host = GPU_HOSTS[gpu]
    ic_cmd = (
        f"aws ec2-instance-connect send-ssh-public-key "
        f"--region us-west-2 --instance-id {host['instance_id']} "
        f"--instance-os-user {host['user']} "
        f"--ssh-public-key file://{SSH_KEY}.pub"
    )
    subprocess.run(ic_cmd, shell=True, capture_output=True, timeout=15)

    scp_cmd = [
        "scp", "-o", "StrictHostKeyChecking=no",
        "-i", str(SSH_KEY),
        local,
        f"{host['user']}@{host['ip']}:{remote}"
    ]
    result = subprocess.run(scp_cmd, capture_output=True, timeout=60)
    return result.returncode == 0

def scp_download(gpu: str, remote: str, local: str) -> bool:
    """Download file from remote GPU."""
    host = GPU_HOSTS[gpu]
    ic_cmd = (
        f"aws ec2-instance-connect send-ssh-public-key "
        f"--region us-west-2 --instance-id {host['instance_id']} "
        f"--instance-os-user {host['user']} "
        f"--ssh-public-key file://{SSH_KEY}.pub"
    )
    subprocess.run(ic_cmd, shell=True, capture_output=True, timeout=15)

    scp_cmd = [
        "scp", "-o", "StrictHostKeyChecking=no",
        "-i", str(SSH_KEY),
        f"{host['user']}@{host['ip']}:{remote}",
        local
    ]
    result = subprocess.run(scp_cmd, capture_output=True, timeout=60)
    return result.returncode == 0

# ============================================================
# Agent Base Class
# ============================================================

class Agent:
    def __init__(self, agent_id: str, name: str, gpu: str = "l4"):
        self.id = agent_id
        self.name = name
        self.gpu = gpu
        self.status = AgentStatus(agent_id=agent_id, name=name, gpu=gpu)
        self.thread: Optional[threading.Thread] = None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.status, k, v)
        self.status.last_update = datetime.now(timezone.utc).isoformat()

    def run_task(self, task_name: str, command: str, timeout: int = 600):
        """Run a task on the assigned GPU."""
        self.update(state=AgentState.RUNNING, current_task=task_name, progress=0.0)
        try:
            stdout, stderr, rc = ssh_exec(self.gpu, command, timeout=timeout)
            if rc == 0:
                self.update(state=AgentState.DONE, progress=100.0)
                return stdout
            else:
                self.update(state=AgentState.FAILED, error=stderr[:500])
                return None
        except Exception as e:
            self.update(state=AgentState.FAILED, error=str(e)[:500])
            return None

    def start(self, task_fn):
        """Start agent in background thread."""
        self.thread = threading.Thread(target=task_fn, daemon=True)
        self.thread.start()

# ============================================================
# GitHub Auto-Push
# ============================================================

def git_push_loop(interval_min=PUSH_INTERVAL_MIN):
    """Background thread: commit + push every interval."""
    while True:
        time.sleep(interval_min * 60)
        try:
            os.chdir(REPO_ROOT)
            # Add all changes
            subprocess.run(["git", "add", "-A"], capture_output=True, timeout=30)
            # Check if there's anything to commit
            result = subprocess.run(["git", "diff", "--cached", "--quiet"], capture_output=True)
            if result.returncode != 0:
                # There are changes
                ts = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
                subprocess.run(
                    ["git", "commit", "-m", f"auto: agent progress update {ts}"],
                    capture_output=True, timeout=30
                )
            # Push
            subprocess.run(
                ["git", "push", "origin", "main"],
                capture_output=True, timeout=60
            )
            print(f"[{datetime.now().isoformat()}] Git push completed")
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] Git push failed: {e}")

# ============================================================
# GPU Monitoring
# ============================================================

def monitor_gpu(gpu: str) -> dict:
    """Get GPU utilization stats."""
    stdout, stderr, rc = ssh_exec(gpu, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits", timeout=15)
    if rc == 0 and stdout.strip():
        parts = stdout.strip().split(", ")
        return {
            "gpu_util": float(parts[0]),
            "mem_used_mb": float(parts[1]),
            "mem_total_mb": float(parts[2]),
            "temp_c": float(parts[3]),
            "mem_pct": float(parts[1]) / float(parts[2]) * 100
        }
    return {"gpu_util": 0, "mem_used_mb": 0, "mem_total_mb": 0, "temp_c": 0, "mem_pct": 0}

def monitor_loop(state: OrchestratorState, interval_sec=30):
    """Background thread: monitor GPU utilization."""
    while True:
        for gpu_name in GPU_HOSTS:
            stats = monitor_gpu(gpu_name)
            state.gpu_utilization[gpu_name] = stats["gpu_util"]
        time.sleep(interval_sec)

# ============================================================
# Orchestrator
# ============================================================

class Orchestrator:
    def __init__(self):
        self.state = OrchestratorState(
            start_time=datetime.now(timezone.utc).isoformat()
        )
        self.agents: Dict[str, Agent] = {}
        self._setup_agents()

    def _setup_agents(self):
        """Initialize all 10 agents."""
        agent_defs = [
            ("literature_hunter", "Literature Hunter", "l4"),
            ("novelty_auditor", "Novelty Auditor", "l4"),
            ("systems_architect", "Systems Architect", "l4"),
            ("benchmark_engineer", "Benchmark Engineer", "l4"),
            ("red_team_critic", "Red-Team Critic", "l4"),
            ("impl_engineer", "Implementation Engineer", "l4"),
            ("training_engineer", "Training Engineer", "l4"),
            ("eval_engineer", "Evaluation Engineer", "l4"),
            ("paper_reviewer", "Paper Reviewer", "l4"),
            ("ranking_committee", "Ranking Committee", "l4"),
        ]
        for aid, name, gpu in agent_defs:
            agent = Agent(aid, name, gpu)
            self.agents[aid] = agent
            self.state.agents[aid] = agent.status

    def get_idle_agents(self) -> List[Agent]:
        return [a for a in self.agents.values() if a.status.state == AgentState.IDLE]

    def get_agent(self, agent_id: str) -> Agent:
        return self.agents[agent_id]

    def save_state(self):
        """Persist orchestrator state to disk."""
        state_file = AGENTS_DIR / "orchestrator_state.json"
        data = {
            "agents": {k: asdict(v) for k, v in self.state.agents.items()},
            "gpu_utilization": self.state.gpu_utilization,
            "active_tasks": self.state.active_tasks,
            "completed_tasks": self.state.completed_tasks,
            "last_push": self.state.last_push,
            "start_time": self.state.start_time,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        # Convert AgentState enums to strings
        for aid in data["agents"]:
            s = data["agents"][aid]["state"]
            if isinstance(s, AgentState):
                data["agents"][aid]["state"] = s.value
        state_file.write_text(json.dumps(data, indent=2))

    def start_background(self):
        """Start background threads: git push + GPU monitor."""
        push_thread = threading.Thread(target=git_push_loop, daemon=True)
        push_thread.start()
        monitor_thread = threading.Thread(target=monitor_loop, args=(self.state,), daemon=True)
        monitor_thread.start()
        print("[Orchestrator] Background threads started (git push + GPU monitor)")

    def deploy_script(self, gpu: str, script_name: str, script_content: str, remote_dir="/tmp/vla_agents"):
        """Upload and prepare a script on remote GPU."""
        local_path = AGENTS_DIR / "scripts" / script_name
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(script_content)
        ssh_exec(gpu, f"mkdir -p {remote_dir}")
        scp_upload(gpu, str(local_path), f"{remote_dir}/{script_name}")
        ssh_exec(gpu, f"chmod +x {remote_dir}/{script_name}")
        return f"{remote_dir}/{script_name}"

    def run_remote(self, gpu: str, script_path: str, timeout: int = 3600):
        """Run a script on remote GPU."""
        return ssh_exec(gpu, f"cd /tmp/vla_agents && python3 {script_path}", timeout=timeout)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    orch = Orchestrator()
    orch.start_background()
    orch.save_state()
    print(f"[Orchestrator] Initialized with {len(orch.agents)} agents")
    print(f"[Orchestrator] L4: {GPU_HOSTS['l4']['ip']} (inference)")
    print(f"[Orchestrator] L40S: {GPU_HOSTS['l40s']['ip']} (finetuning, deferred)")
