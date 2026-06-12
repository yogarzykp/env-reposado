"""Skill-task data via self-instruct + Reflexion + execution-verified reward.

This is the env-reposado skill stream (anticipated NL->bash round). It is the
same ReST shape as the games -- rejection sampling on a verifiable signal -- but
deliberately distinct from the baseline's intercode (ReAct over a fixed eval set
with fs-diff/MD5/tfidf scoring):

  - SELF-INSTRUCT tasks: synthetic bash problems over a random sandbox whose gold
    answer is computed by *running a reference command* (contamination-free,
    always verifiable). Self-Instruct (Wang et al. 2022, arXiv 2212.10560).
  - FUNCTIONAL EXEC-MATCH reward: the model's submitted answer must equal the
    gold command's output (unit-test semantics), not a filesystem hash.
  - REFLEXION retry: on a wrong submission the model gets feedback and tries
    again (Shinn et al. 2023, arXiv 2303.11366).

Only solved episodes become SFT samples ({messages}, multi-turn ReAct), merged
into the same accumulated ReST dataset as the games.

Safety: model-generated bash runs in an ephemeral temp dir with a deny-list,
no path escapes (``..``), a minimal PATH and a hard timeout.
"""

from __future__ import annotations

import os
import re
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from selfplay.rollout_collector import GenerateFn, Messages

SKILL_SYSTEM_PROMPT = (
    "You are a command-line assistant. Solve the task by interacting with a bash "
    "shell in the current directory. Work step by step. On each turn respond in "
    "exactly this format:\n"
    "Thought:\n<your reasoning>\n\nAction:\n<one of: execute[<bash command>]  or  "
    "submit[<final answer>]>\n\n"
    "Use execute[...] to run a command and read its Observation; use submit[...] "
    "once you know the final answer."
)

_EXEC_RE = re.compile(r"execute\[(.*?)\]", re.DOTALL)
_SUBMIT_RE = re.compile(r"submit\[(.*?)\]", re.DOTALL)

# Commands containing any of these (or a path escape) are refused by the sandbox.
_DENY = (
    "rm -rf", "rm -r ", "sudo", "curl", "wget", "mkfs", "dd if=", ":(){",
    "/etc", "/dev", "/sys", "chmod -r", "chown", "ssh", "scp", "nc ", "telnet",
    "shutdown", "reboot", "kill ", "pkill", "mv /", "cp /",
)


# --------------------------------------------------------------------------- #
# Ephemeral sandbox
# --------------------------------------------------------------------------- #


def is_safe_command(cmd: str) -> bool:
    low = (cmd or "").lower()
    if not low.strip():
        return False
    if ".." in cmd:                      # no path escape
        return False
    if re.search(r"(^|[\s=])/", cmd):    # no absolute paths
        return False
    return not any(tok in low for tok in _DENY)


class EphemeralSandbox:
    """Materialise a task's files in a temp dir; run commands confined to it."""

    def __init__(self, files: Dict[str, str]):
        self.files = files
        self.dir: Optional[str] = None

    def __enter__(self) -> "EphemeralSandbox":
        self.dir = tempfile.mkdtemp(prefix="rest_skill_")
        for rel, content in self.files.items():
            path = os.path.join(self.dir, rel)
            os.makedirs(os.path.dirname(path) or self.dir, exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
        return self

    def run(self, cmd: str, timeout: int = 10) -> str:
        if not is_safe_command(cmd):
            return "ERROR: command rejected by sandbox policy"
        try:
            p = subprocess.run(
                ["/bin/bash", "-c", cmd], cwd=self.dir, capture_output=True,
                text=True, timeout=timeout,
                env={"PATH": "/usr/bin:/bin:/usr/local/bin", "LC_ALL": "C"},
            )
            return (p.stdout + p.stderr).strip()
        except subprocess.TimeoutExpired:
            return "ERROR: timeout"
        except Exception as e:  # pragma: no cover - defensive
            return f"ERROR: {e}"

    def __exit__(self, *exc) -> None:
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Self-instruct task generation (gold computed by running a reference command)
# --------------------------------------------------------------------------- #


@dataclass
class SkillTask:
    instruction: str
    files: Dict[str, str]
    gold_command: str
    gold_output: str = ""


_EXTS = ["txt", "log", "csv", "md", "py"]
_STEMS = ["alpha", "beta", "gamma", "delta", "notes", "data", "report", "main", "util"]


def _random_files(rng: random.Random) -> Dict[str, str]:
    files: Dict[str, str] = {}
    n = rng.randint(5, 11)
    dirs = ["", "sub/", "sub/deep/"]
    for _ in range(n):
        ext = rng.choice(_EXTS)
        stem = rng.choice(_STEMS) + str(rng.randint(1, 99))
        rel = rng.choice(dirs) + f"{stem}.{ext}"
        lines = rng.randint(1, 6)
        files[rel] = "\n".join(f"line {i}" for i in range(lines)) + "\n"
    return files


def _present_exts(files: Dict[str, str]) -> List[str]:
    return sorted({name.rsplit(".", 1)[-1] for name in files})


def _tmpl_count_ext(rng, files):
    ext = rng.choice(_present_exts(files))
    return SkillTask(
        instruction=f"How many files in the directory tree have the .{ext} extension? Answer with just the number.",
        files=files,
        gold_command=f"find . -type f -name '*.{ext}' | wc -l | tr -d ' '",
    )


def _tmpl_list_sorted(rng, files):
    ext = rng.choice(_present_exts(files))
    return SkillTask(
        instruction=f"List the relative paths of all .{ext} files, one per line, sorted alphabetically.",
        files=files,
        gold_command=f"find . -type f -name '*.{ext}' | sed 's|^\\./||' | sort",
    )


def _tmpl_total_lines(rng, files):
    ext = rng.choice(_present_exts(files))
    return SkillTask(
        instruction=f"Report the total number of lines across all .{ext} files. Answer with just the number.",
        files=files,
        gold_command=f"find . -type f -name '*.{ext}' -exec cat {{}} \\; | wc -l | tr -d ' '",
    )


_TEMPLATES = [_tmpl_count_ext, _tmpl_list_sorted, _tmpl_total_lines]


def generate_tasks(n: int, seed: int = 0) -> List[SkillTask]:
    """Build N verifiable bash tasks; gold_output is the reference command's run."""
    rng = random.Random(seed)
    tasks: List[SkillTask] = []
    for _ in range(n):
        files = _random_files(rng)
        task = rng.choice(_TEMPLATES)(rng, files)
        with EphemeralSandbox(task.files) as sb:
            task.gold_output = sb.run(task.gold_command)
        tasks.append(task)
    return tasks


# --------------------------------------------------------------------------- #
# Solving with Reflexion + execution-verified reward
# --------------------------------------------------------------------------- #


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in (text or "").strip().splitlines())


def functional_exec_match(submitted: str, gold_output: str) -> bool:
    return _normalize(submitted) == _normalize(gold_output)


def _initial_listing(sandbox: EphemeralSandbox) -> str:
    return sandbox.run("find . -type f | sed 's|^\\./||' | sort")


def solve_with_reflexion(
    task: SkillTask,
    generate_fn: GenerateFn,
    max_steps: int = 6,
    temperature: float = 0.7,
) -> (bool, Messages):
    """Run a ReAct/Reflexion episode. Returns (solved, full message trace)."""
    with EphemeralSandbox(task.files) as sb:
        listing = _initial_listing(sb)
        messages: Messages = [
            {"role": "system", "content": SKILL_SYSTEM_PROMPT},
            {"role": "user", "content": f"{task.instruction}\n\nFiles:\n{listing}"},
        ]
        for _ in range(max_steps):
            completion = generate_fn([messages], n=1, temperature=temperature)[0][0]
            messages.append({"role": "assistant", "content": completion})

            sub = _SUBMIT_RE.search(completion)
            if sub:
                if functional_exec_match(sub.group(1).strip(), task.gold_output):
                    return True, messages
                # Reflexion: tell the model it was wrong and let it retry.
                messages.append({"role": "user",
                                 "content": "Observation:\nIncorrect answer. Re-examine and try again."})
                continue

            ex = _EXEC_RE.search(completion)
            if ex:
                obs = sb.run(ex.group(1).strip())
                messages.append({"role": "user", "content": f"Observation:\n{obs}"})
            else:
                messages.append({"role": "user",
                                 "content": "Observation:\nNo valid action. Use execute[...] or submit[...]."})
    return False, messages


def collect_skill_samples(
    generate_fn: GenerateFn,
    n_tasks: int = 64,
    seed: int = 0,
    max_steps: int = 6,
    temperature: float = 0.7,
) -> List[Dict[str, object]]:
    """Self-instruct -> solve -> keep only solved episodes as {messages} samples."""
    samples: List[Dict[str, object]] = []
    for task in generate_tasks(n_tasks, seed):
        solved, messages = solve_with_reflexion(task, generate_fn, max_steps, temperature)
        if solved:
            samples.append({"messages": _drop_trailing_feedback(messages)})
    return samples


def _drop_trailing_feedback(messages: Messages) -> Messages:
    """Keep the conversation up to and including the winning assistant turn."""
    last_assistant = max((i for i, m in enumerate(messages) if m["role"] == "assistant"),
                         default=len(messages) - 1)
    return messages[: last_assistant + 1]


# --------------------------------------------------------------------------- #
# Offline selftest: real bash sandbox, scripted "model".
# --------------------------------------------------------------------------- #


def _selftest() -> None:
    # Safety policy.
    assert not is_safe_command("rm -rf /")
    assert not is_safe_command("cat ../secret")
    assert not is_safe_command("cat /etc/passwd")
    assert is_safe_command("find . -name '*.txt' | wc -l")

    # A concrete, verifiable task.
    files = {"a.txt": "x\n", "b.txt": "y\n", "sub/c.txt": "z\n", "d.log": "n\n"}
    task = SkillTask(
        instruction="How many .txt files? Answer with just the number.",
        files=files,
        gold_command="find . -type f -name '*.txt' | wc -l | tr -d ' '",
    )
    with EphemeralSandbox(task.files) as sb:
        task.gold_output = sb.run(task.gold_command)
    assert task.gold_output == "3", repr(task.gold_output)

    # Scripted solver: first execute the gold command, then submit the observation.
    def good_gen(prompts, n=1, temperature=1.0):
        msgs = prompts[0]
        n_assist = sum(1 for m in msgs if m["role"] == "assistant")
        if n_assist == 0:
            return [["Thought:\ncount txt files.\nAction:\nexecute[find . -type f -name '*.txt' | wc -l | tr -d ' ']"]]
        answer = msgs[-1]["content"].split("\n")[-1].strip()
        return [[f"Thought:\nthe count is {answer}.\nAction:\nsubmit[{answer}]"]]

    solved, trace = solve_with_reflexion(task, good_gen)
    assert solved and trace[-1]["role"] == "assistant" and "submit[3]" in trace[-1]["content"], trace

    # Wrong solver: always submits a bad answer -> Reflexion exhausts -> unsolved.
    def bad_gen(prompts, n=1, temperature=1.0):
        return [["Thought:\nguess.\nAction:\nsubmit[999]"]]

    solved_bad, _ = solve_with_reflexion(task, bad_gen, max_steps=3)
    assert not solved_bad

    # End-to-end collection keeps only solved tasks and yields {messages}.
    out = collect_skill_samples(good_gen, n_tasks=4, seed=1)
    assert out and all("messages" in s and s["messages"][-1]["role"] == "assistant" for s in out)
    print(f"skill_selfinstruct selftest OK ({len(out)}/4 tasks solved)")


if __name__ == "__main__":
    _selftest()
