"""Task-independent NatureBench boundary for the pinned MLEvolve adapter."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from naturebench_bridge.clock import NatureBenchClock
from naturebench_bridge.jsonio import write_json_atomic


MODEL = "gpt-5.6-luna"
EFFORT = "xhigh"
PROMPT_ADAPTER_VERSION = "mlevolve-naturebench-v1"
_CLOCK: NatureBenchClock | None = None
_CLOCK_LOCK = threading.Lock()


def is_naturebench(cfg: Any) -> bool:
    return str(getattr(cfg, "task_mode", "mlebench")) == "naturebench"


def clock(cfg: Any) -> NatureBenchClock:
    if not is_naturebench(cfg):
        raise RuntimeError("NatureBench clock requested outside NatureBench task mode")
    global _CLOCK
    with _CLOCK_LOCK:
        if _CLOCK is None:
            _CLOCK = NatureBenchClock.from_environment()
        return _CLOCK


def effective_elapsed_seconds(cfg: Any, wall_start: float | None = None) -> float:
    if is_naturebench(cfg):
        return clock(cfg).effective_elapsed_seconds
    import time

    return time.time() - float(wall_start or time.time())


def remaining_seconds(cfg: Any) -> float:
    if is_naturebench(cfg):
        return clock(cfg).remaining_seconds
    return float(cfg.agent.time_limit)


def fraction_elapsed(cfg: Any, wall_start: float | None = None) -> float:
    if is_naturebench(cfg):
        return clock(cfg).fraction_elapsed
    return min(1.0, effective_elapsed_seconds(cfg, wall_start) / float(cfg.agent.time_limit))


def expired(cfg: Any) -> bool:
    return clock(cfg).expired if is_naturebench(cfg) else False


def llm_timeout_seconds(cfg: Any, requested: float = 1200.0) -> float:
    if not is_naturebench(cfg):
        return requested
    return clock(cfg).clip_timeout(requested, reserve_seconds=0.5)


_CONTRACT = """NATUREBENCH EXECUTION CONTRACT (overrides inherited competition wording):
- Produce the complete Python contents of `run.py`; do not create or expect a competition-style tabular submission.
- Read task inputs only through `Path(os.environ["DATA_DIR"])` and write every required
  task-specific output beneath `Path(os.environ["OUTPUT_DIR"])`.
- Follow the exact public README/data-description output schema. The official bridge,
  not a printed local validation score, determines `aggregate_improvement` (higher is better).
- The candidate has no network. Do not download anything or call web/search services.
- Evaluator internals, ground truth, SOTA metadata, other runs, and framework artifacts are unavailable.
- A complete candidate must run autonomously from the preinstalled task environment.
"""


_REPLACEMENTS = (
    (r"(?i)kaggle", "scientific benchmark"),
    (r"(?i)mle[- ]bench", "NatureBench"),
    (r"(?i)sample_submission\.csv", "public task output specification"),
    (r"(?i)submission\.csv", "task-specific output files"),
    (r"(?i)\./submission/?", "OUTPUT_DIR"),
    (r"(?i)\./input/?", "DATA_DIR"),
    (r"(?i)real leaderboard", "official evaluator"),
    (r"(?i)leaderboard", "official evaluator"),
    (
        r"(?i)the very last line of execution must be[^\n]*",
        "The candidate must finish only after writing the complete public output contract.",
    ),
    (
        r"(?i)must print[^\n]*final validation score[^\n]*",
        "Local diagnostic metrics are optional and are never used for official ranking.",
    ),
    (
        r"(?i)during code development, you can and should use online resources[^\n]*",
        "Network access is unavailable; use only preinstalled packages and prestaged assets.",
    ),
    (
        r'(?i)-?\s*the\s+["“]?no internet access["”]?\s+restriction[^\n]*',
        "Network access is unavailable to both the solver and candidate runtime.",
    ),
    (
        r"(?i)-?\s*\*\*do not question internet access concerns[^\n]*",
        "Use only preinstalled packages and prestaged assets; remote model loading is unavailable.",
    ),
    (
        r"(?i)📦\s*\*\*packages\s*&\s*internet\*\*:[^\n]*",
        "📦 **Packages**: use only libraries and assets already present in the pinned task image.",
    ),
    (
        r"(?is)\*\*competition data strategy[^\n]*\*\*.*?"
        r"\*\*note\*\*: if existing code already implements this strategy, i will skip this step\.\*\*",
        "**PUBLIC DATA SPLITS**: follow the public task description exactly; do not infer competition-specific split semantics.",
    ),
)


def adapt_prompt(text: str, cfg: Any, *, include_contract: bool = True) -> str:
    if not is_naturebench(cfg):
        return text
    adapted = text
    for pattern, replacement in _REPLACEMENTS:
        adapted = re.sub(pattern, replacement, adapted)
    return f"{_CONTRACT}\n{adapted}" if include_contract else adapted


def assert_startup_config(cfg: Any) -> dict[str, Any]:
    if not is_naturebench(cfg):
        return {"task_mode": "mlebench", "validated": True}
    roles = {
        "code": {
            "model": str(cfg.agent.code.model),
            "reasoning_effort": str(cfg.agent.code.reasoning_effort),
        },
        "feedback": {
            "model": str(cfg.agent.feedback.model),
            "reasoning_effort": str(cfg.agent.feedback.reasoning_effort),
        },
    }
    if {row["model"] for row in roles.values()} != {MODEL}:
        raise RuntimeError(f"solver model allowlist must be exactly {{{MODEL!r}}}")
    if {row["reasoning_effort"] for row in roles.values()} != {EFFORT}:
        raise RuntimeError("every MLEvolve generative role must use reasoning_effort=xhigh")
    required_environment = (
        "NATUREBENCH_LUNA_SOCKET",
        "NATUREBENCH_START_REPO",
        "NATUREBENCH_PUBLIC_PROBLEM",
    )
    missing = [name for name in required_environment if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"NatureBench runtime environment is incomplete: {missing}")
    payload = {
        "schema": "naturebench.mlevolve-provider-audit",
        "task_mode": "naturebench",
        "provider": "host-unix-responses-relay",
        "model_allowlist": [MODEL],
        "reasoning_effort_allowlist": [EFFORT],
        "roles": roles,
        "prompt_adapter_version": PROMPT_ADAPTER_VERSION,
        "automatic_fallback": False,
        "web_search": False,
        "validated": True,
    }
    audit_path = os.environ.get("NATUREBENCH_PROVIDER_AUDIT", "").strip()
    if audit_path:
        write_json_atomic(Path(audit_path), payload)
    return payload


def materialize_candidate_command(
    *, cfg: Any, code: str, candidate_id: str, node: Any, cpu_ids: set[int]
) -> tuple[list[str], Path]:
    if not is_naturebench(cfg):
        raise RuntimeError("NatureBench candidate command requested in MLE-bench mode")
    if expired(cfg):
        raise TimeoutError("NatureBench effective solve budget is exhausted")
    starter = Path(os.environ["NATUREBENCH_START_REPO"]).resolve(strict=True)
    worktree = Path(cfg.workspace_dir) / "candidate_worktrees" / candidate_id
    if worktree.exists() or worktree.is_symlink():
        raise FileExistsError(f"candidate worktree already exists: {worktree}")
    shutil.copytree(starter, worktree, symlinks=True)
    (worktree / "run.py").write_text(code, encoding="utf-8")
    subprocess.run(["git", "add", "run.py"], cwd=worktree, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=NatureBench MLEvolve",
            "-c",
            "user.email=naturebench@example.invalid",
            "commit",
            "--allow-empty",
            "-m",
            f"MLEvolve candidate {candidate_id}",
        ],
        cwd=worktree,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    candidate_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=worktree, text=True
    ).strip()
    parent_ids: list[str] = []
    current_parent = getattr(node, "parent", None)
    if current_parent is not None:
        parent_ids.append(str(current_parent.id))
    metadata = {
        "stage": str(getattr(node, "stage", "unknown")),
        "branch_id": getattr(node, "branch_id", None),
        "framework_search_step": getattr(node, "step", None),
        "from_topk": bool(getattr(node, "from_topk", False)),
        "llm_model": MODEL,
        "reasoning_effort": EFFORT,
        "prompt_adapter_version": PROMPT_ADAPTER_VERSION,
        "candidate_worktree_commit": candidate_commit,
    }
    command = [
        sys.executable,
        "-m",
        "naturebench_bridge.solver_client",
        "--repo",
        str(worktree),
        "--candidate-id",
        candidate_id,
        "--metadata",
        json.dumps(metadata, sort_keys=True),
        "--gpus",
        str(int(cfg.agent.search.num_gpus)),
        "--candidate-timeout-seconds",
        str(int(cfg.exec.timeout)),
    ]
    for parent_id in parent_ids:
        command.extend(["--parent-id", parent_id])
    taskset = shutil.which("taskset")
    if taskset and cpu_ids:
        command = [taskset, "-c", ",".join(map(str, sorted(cpu_ids))), *command]
    return command, worktree


def parse_candidate_result(term_out: str) -> dict[str, Any]:
    for line in reversed(term_out.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("schema") == "naturebench.framework-candidate-result":
            value = payload.get("metric_value")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError("NatureBench bridge result has no numeric metric_value")
            return payload
    raise RuntimeError("execution output contains no official NatureBench bridge result")
