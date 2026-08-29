"""Train mjlab-microduck on Modal GPUs, under a hard dollar cap.

    python modal_app.py plan                              # cost table, no Modal call
    modal run modal_app.py::probe --gpus L4,L40S          # measure s/iter per GPU
    MICRODUCK_GPU=L4 modal run --detach modal_app.py::main   # train under the cap

The `::main` is required, not decoration: two local entrypoints live here, so
a bare `modal run modal_app.py` cannot pick one.

The budget is enforced as the Modal function `timeout`: seconds = budget /
$-per-second for the exact machine requested. Modal kills the container at that
mark, so a run cannot outspend its cap no matter what the trainer does.
Checkpoints land in a Volume every `save_interval` iterations and the Volume
background-commits, so hitting the cap costs the tail of a run, never the run.

Modal 1.4 fixes `gpu` and `timeout` at decoration time and has no
`Function.with_options`, so the training machine is chosen by environment
variable at import:

    MICRODUCK_GPU=L40S MICRODUCK_BUDGET_USD=10 modal run --detach modal_app.py::main

The rate is computed against ASSUME_CPU/ASSUME_MEM_GIB ceilings rather than the
reservations below, because Modal bills max(requested, actually used).
Overstating usage shortens the timeout, which errs toward underspending.

What this does NOT cap: image builds (billed as CPU-only build compute, cents),
and the total across runs — the cap is per run. Two runs at $10 can spend $20.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).parent
APP_NAME = "microduck-rl"
REMOTE = "/root/repo"

# Modal list prices in USD/second, from modal.com/pricing (checked 2026-08-28).
# Keys are the exact strings Modal's `gpu=` accepts.
GPU_USD_S: dict[str, float] = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
}
CPU_USD_CORE_S = 0.0000131
MEM_USD_GIB_S = 0.00000222

# Reservations: floors that guarantee scheduling. Modal bills
# max(reservation, actual), so these are not what the budget math trusts.
CPU_RESERVE = 2.0
MEM_RESERVE_MIB = 8192

# Ceilings the budget math assumes we might actually consume. `probe` reports
# peak RSS so these can be checked against reality; keep them above it.
ASSUME_CPU = 8.0
ASSUME_MEM_GIB = 32.0

# Trim the timeout so clock skew and shutdown never nudge us over the cap.
SAFETY = 0.90

DEFAULT_TASK = "Mjlab-Velocity-Flat-MicroDuck"
DEFAULT_NUM_ENVS = 4096
MODAL_MAX_TIMEOUT_S = 24 * 3600


# Modal can satisfy a request with a larger card than asked for: probing
# gpu="A100-40GB" on 2026-08-29 landed an "NVIDIA A100 80GB PCIe". If that is
# billed at the 80 GB rate, pricing the cap at the 40 GB rate overshoots the
# budget by 13%. Price these at the dearest card the request can produce, so the
# cap holds whichever machine turns up.
GPU_BILLED_AS = {"A100-40GB": "A100-80GB", "A100": "A100-80GB"}


def usd_per_second(gpu: str, assume_cpu: float = ASSUME_CPU,
                   assume_mem_gib: float = ASSUME_MEM_GIB) -> float:
    """Worst-case billed rate for one container-second on this machine."""
    if gpu not in GPU_USD_S:
        raise ValueError(f"unknown gpu {gpu!r}; known: {', '.join(GPU_USD_S)}")
    gpu_price = GPU_USD_S[GPU_BILLED_AS.get(gpu, gpu)]
    return gpu_price + assume_cpu * CPU_USD_CORE_S + assume_mem_gib * MEM_USD_GIB_S


def budget_to_timeout(budget_usd: float, gpu: str) -> int:
    """Seconds of runtime `budget_usd` buys, floored into Modal's 24 h limit."""
    seconds = int(budget_usd / usd_per_second(gpu) * SAFETY)
    if seconds < 60:
        raise ValueError(
            f"${budget_usd:.2f} buys only {seconds}s on {gpu} — raise the budget "
            f"or pick a cheaper GPU"
        )
    return min(seconds, MODAL_MAX_TIMEOUT_S)


# The machine for `train`, chosen at import because Modal needs gpu/timeout at
# decoration time. Everything else is a normal CLI flag.
GPU = os.environ.get("MICRODUCK_GPU", "L4")
BUDGET_USD = float(os.environ.get("MICRODUCK_BUDGET_USD", "10"))
TRAIN_TIMEOUT = budget_to_timeout(BUDGET_USD, GPU)

# --- image ----------------------------------------------------------------
# `uv sync` runs at build time so the ~6 GB of torch + CUDA wheels bake into a
# cached layer. `git` is here because better-actuator-models installs from a
# git branch. Excluding .venv matters: the local one is 1.2 GB of macOS wheels.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("uv")
    .add_local_dir(
        REPO,
        REMOTE,
        copy=True,
        ignore=[
            ".venv", "**/.venv", ".git", "logs", "outputs", "wandb",
            "**/__pycache__", "**/*.pyc", "**/*.pt", "**/*.onnx",
            # This file. Modal mounts the entrypoint module itself, so copying
            # it in as well only means every edit here busts the uv sync layer
            # below and re-downloads 6 GB of wheels.
            "modal_app.py",
        ],
    )
    .run_commands(f"cd {REMOTE} && uv sync --frozen")
    .env({
        "UV_PROJECT_ENVIRONMENT": f"{REMOTE}/.venv",
        "PYTHONUNBUFFERED": "1",
        # Warp compiles kernels on first use — a fixed 1-3 min cold-start tax.
        # Caching them in a Volume makes every run after the first one cheaper.
        "WARP_CACHE_PATH": "/root/.cache/warp",
    })
)

app = modal.App(APP_NAME, image=image)

logs_volume = modal.Volume.from_name(f"{APP_NAME}-logs", create_if_missing=True)
warp_cache = modal.Volume.from_name(f"{APP_NAME}-warp-cache", create_if_missing=True)
VOLUMES = {f"{REMOTE}/logs": logs_volume, "/root/.cache/warp": warp_cache}

# --no-sync: the venv was built into the image; re-resolving at run time
# would need the network and could drift from the locked build.
UV_RUN = ["uv", "run", "--no-sync"]

ITER_RE = re.compile(r"Iteration time:\s*([0-9.]+)s")


def _run_train(argv: list[str], echo: bool = True) -> tuple[int, list[float]]:
    """Run the trainer, stream its output, and collect per-iteration seconds."""
    env = dict(os.environ)
    env.setdefault("WANDB_MODE", "online" if "WANDB_API_KEY" in env else "disabled")
    proc = subprocess.Popen(
        argv, cwd=REMOTE, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    iters: list[float] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        if echo:
            sys.stdout.write(line)
        m = ITER_RE.search(line)
        if m:
            iters.append(float(m.group(1)))
    return proc.wait(), iters


# The container has two Pythons: the image's system one (which has only `uv`)
# and the project venv at REMOTE/.venv where torch and warp actually live.
# Everything that touches a dependency has to cross into the venv, so this runs
# as a subprocess there rather than importing into the function body.
_REPORT_SRC = r"""
import json, subprocess
import torch, warp as wp
wp.init()
smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
     "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip()
print("REPORT_JSON:" + json.dumps({
    "nvidia_smi": smi,
    "torch_cuda": torch.cuda.is_available(),
    "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "warp_cuda": [str(d) for d in wp.get_cuda_devices()],
}))
"""


def _machine_report() -> dict:
    """GPU/CUDA sanity, read from inside the venv."""
    out = subprocess.run(
        UV_RUN + ["python", "-c", _REPORT_SRC],
        cwd=REMOTE, capture_output=True, text=True,
    )
    # Warp prints an init banner, so pick the tagged line rather than trusting
    # stdout to be nothing but JSON.
    for line in out.stdout.splitlines():
        if line.startswith("REPORT_JSON:"):
            return json.loads(line[len("REPORT_JSON:"):])
    return {"report_error": (out.stdout[-800:] + out.stderr[-800:]).strip()}


def _steady(iters: list[float]) -> float | None:
    """Mean iteration time with the warm-up dropped.

    The first iterations carry Warp kernel compilation and CUDA graph capture,
    which flatter or wreck the average depending on cache state.
    """
    tail = iters[5:] if len(iters) > 8 else iters
    return sum(tail) / len(tail) if tail else None


def _checkpoints(run_name: str = "") -> list[str]:
    """This run's checkpoints, oldest first.

    Ordered by mtime, not by name: `sorted()` on the filenames puts
    model_1000.pt before model_39.pt, so the "latest checkpoint" hint printed at
    the end of a run would name the wrong file.
    """
    logs = Path(REMOTE) / "logs"
    if not logs.exists():
        return []
    paths = [p for p in logs.rglob("model_*.pt") if not run_name or run_name in str(p)]
    paths.sort(key=lambda p: p.stat().st_mtime)
    return [str(p.relative_to(logs)) for p in paths]


def _probe_body(gpu: str, task: str, num_envs: int, iterations: int) -> dict:
    import resource

    t0 = time.time()
    report = _machine_report()
    print(json.dumps(report, indent=2), flush=True)

    argv = UV_RUN + ["train", task,
            "--env.scene.num-envs", str(num_envs),
            "--agent.max-iterations", str(iterations)]
    print(f"$ {shlex.join(argv)}", flush=True)
    code, iters = _run_train(argv, echo=False)

    wall = time.time() - t0
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "gpu": gpu,
        "exit_code": code,
        "num_envs": num_envs,
        "iterations_seen": len(iters),
        "s_per_iter": _steady(iters),
        "first_iter_s": iters[0] if iters else None,
        "startup_s": round(wall - sum(iters), 1),
        "wall_s": round(wall, 1),
        # These two are what ASSUME_CPU / ASSUME_MEM_GIB should sit above. Modal
        # bills max(reservation, actual), so guessing high here only shortens the
        # timeout — but guessing far too high leaves budget unspent.
        "peak_rss_gib": round(ru.ru_maxrss / (1024 ** 2), 2),
        "avg_cores": round((ru.ru_utime + ru.ru_stime) / wall, 2),
        **report,
    }


# --- probe ----------------------------------------------------------------
# Written out per GPU rather than generated: Modal needs `gpu` at decoration
# time, and one explicit function each beats a factory that has to be
# `serialized=True` to survive being defined in a local scope.
_PROBE_OPTS = dict(volumes=VOLUMES, cpu=CPU_RESERVE, memory=MEM_RESERVE_MIB,
                   timeout=900, retries=0, max_containers=1)


@app.function(gpu="T4", **_PROBE_OPTS)
def probe_t4(task: str, num_envs: int, iterations: int) -> dict:
    return _probe_body("T4", task, num_envs, iterations)


@app.function(gpu="L4", **_PROBE_OPTS)
def probe_l4(task: str, num_envs: int, iterations: int) -> dict:
    return _probe_body("L4", task, num_envs, iterations)


@app.function(gpu="A10", **_PROBE_OPTS)
def probe_a10(task: str, num_envs: int, iterations: int) -> dict:
    return _probe_body("A10", task, num_envs, iterations)


@app.function(gpu="L40S", **_PROBE_OPTS)
def probe_l40s(task: str, num_envs: int, iterations: int) -> dict:
    return _probe_body("L40S", task, num_envs, iterations)


@app.function(gpu="A100-40GB", **_PROBE_OPTS)
def probe_a100(task: str, num_envs: int, iterations: int) -> dict:
    return _probe_body("A100-40GB", task, num_envs, iterations)


@app.function(gpu="H100", **_PROBE_OPTS)
def probe_h100(task: str, num_envs: int, iterations: int) -> dict:
    return _probe_body("H100", task, num_envs, iterations)


PROBE_FNS = {"T4": probe_t4, "L4": probe_l4, "A10": probe_a10,
             "L40S": probe_l40s, "A100-40GB": probe_a100, "H100": probe_h100}


# --- train ----------------------------------------------------------------
@app.function(
    gpu=GPU,
    volumes=VOLUMES,
    cpu=CPU_RESERVE,
    memory=MEM_RESERVE_MIB,
    timeout=TRAIN_TIMEOUT,  # <- the budget, in seconds
    retries=0,              # a retry would silently double the spend
    max_containers=1,
)
def train_one(task: str, num_envs: int, extra: str, run_name: str) -> dict:
    t0 = time.time()
    print(json.dumps(_machine_report(), indent=2), flush=True)

    argv = UV_RUN + ["train", task,
                     "--env.scene.num-envs", str(num_envs),
                     "--agent.run-name", run_name] + shlex.split(extra)
    print(f"$ {shlex.join(argv)}", flush=True)
    code, iters = _run_train(argv)

    logs_volume.commit()
    ckpts = _checkpoints(run_name)
    return {
        "exit_code": code,
        "iterations": len(iters),
        "s_per_iter": _steady(iters),
        "wall_s": round(time.time() - t0, 1),
        "checkpoint_count": len(ckpts),
        "checkpoints": ckpts[-5:],
    }


# --- entrypoints ----------------------------------------------------------
@app.local_entrypoint()
def probe(gpus: str = "L4,L40S,A100-40GB", task: str = DEFAULT_TASK,
          num_envs: int = DEFAULT_NUM_ENVS, iterations: int = 40):
    """Measure s/iter on each GPU with a short real training run."""
    wanted = [g.strip() for g in gpus.split(",") if g.strip()]
    if unknown := [g for g in wanted if g not in PROBE_FNS]:
        sys.exit(f"no probe function for {unknown}; known: {list(PROBE_FNS)}")

    ceiling = sum(usd_per_second(g) * 300 for g in wanted)
    print(f"probing {wanted} at {num_envs} envs x {iterations} iters "
          f"(~${ceiling:.2f} if each takes 5 min)\n")

    results = []
    for g in wanted:
        print(f"=== {g} ===", flush=True)
        try:
            r = PROBE_FNS[g].remote(task, num_envs, iterations)
            print(json.dumps(r, indent=2), flush=True)
        except Exception as e:  # a GPU class can be unavailable, or OOM
            print(f"{g}: FAILED {type(e).__name__}: {e}\n")
            r = {"gpu": g, "error": f"{type(e).__name__}: {e}"}
        results.append(r)

    print("\n" + "=" * 70)
    print(f"{'GPU':<12} {'s/iter':>8} {'$/hr':>7} {'iters per $10':>14} {'peak RSS':>10}")
    print("-" * 70)
    best = None
    for r in results:
        if not r.get("s_per_iter"):
            print(f"{r['gpu']:<12} {'failed':>8}")
            continue
        per10 = int(budget_to_timeout(10, r["gpu"]) / r["s_per_iter"])
        print(f"{r['gpu']:<12} {r['s_per_iter']:>8.3f} {usd_per_second(r['gpu']) * 3600:>7.2f} "
              f"{per10:>14,} {r.get('peak_rss_gib', 0):>9.1f}G")
        if best is None or per10 > best[1]:
            best = (r["gpu"], per10)
    if best:
        print(f"\nBest iterations per dollar: {best[0]} ({best[1]:,} at $10)")
        spi = ",".join(f"{r['gpu']}={r['s_per_iter']:.3f}"
                       for r in results if r.get("s_per_iter"))
        print(f'  python modal_app.py plan --s-per-iter "{spi}"')


@app.local_entrypoint()
def main(task: str = DEFAULT_TASK, num_envs: int = DEFAULT_NUM_ENVS,
         extra: str = "", run_name: str = "", resume_run: str = "",
         resume_checkpoint: str = "model_.*.pt", dry_run: bool = False):
    """Train under the hard cap. GPU/budget come from MICRODUCK_GPU / MICRODUCK_BUDGET_USD.

    A budget caps one *segment*, not the training. mjlab resolves --agent.load-run
    against the log root, which here is the Volume the previous segment wrote to,
    so `--resume-run <dir>` continues where the last cap cut it off without any
    download round-trip. Three $10 runs are a $30 run in three pieces.
    """
    run_name = run_name or f"modal-{GPU.lower()}-{int(time.time())}"
    if resume_run:
        extra = (f"{extra} --agent.resume True --agent.load-run {resume_run} "
                 f"--agent.load-checkpoint {resume_checkpoint}").strip()
    rate = usd_per_second(GPU)
    print(f"task      {task}")
    print(f"machine   {GPU}, {CPU_RESERVE:g} cores / {MEM_RESERVE_MIB / 1024:g} GiB reserved")
    print(f"envs      {num_envs}")
    print(f"budget    ${BUDGET_USD:.2f} -> {TRAIN_TIMEOUT}s ({TRAIN_TIMEOUT / 3600:.1f}h) "
          f"at ${rate * 3600:.2f}/hr")
    print(f"run-name  {run_name}")
    print(f"extra     {extra or '(none)'}")
    print(f"logs      volume {APP_NAME}-logs -> {REMOTE}/logs")
    if dry_run:
        print("\n--dry-run: nothing launched, nothing spent.")
        return

    try:
        result = train_one.remote(task, num_envs, extra, run_name)
        print("\n" + json.dumps(result, indent=2))
        latest = (result.get("checkpoints") or [None])[-1]
    except modal.exception.FunctionTimeoutError:
        # The expected way a capped run ends, not an error. The container is
        # gone, so nothing was returned — but the Volume background-commits, so
        # the checkpoints are there and worth pointing at.
        print(f"\nBudget reached: Modal stopped the run at {TRAIN_TIMEOUT}s "
              f"(~${BUDGET_USD:.2f}). Checkpoints up to that point are in the Volume.")
        latest = None

    print(f"\nList what landed:\n  modal volume ls {APP_NAME}-logs "
          f"rsl_rl -R | grep {run_name}")
    if latest:
        name = Path(latest).name
        print(f"\nPull the last checkpoint:"
              f"\n  modal volume get {APP_NAME}-logs {latest} ./{name}"
              f"\nExport and drive it locally:"
              f"\n  uv run scripts/export.py {task} --checkpoint-file ./{name}"
              f"\n  ./sim.sh -- --walking output.onnx")
    print(f"\nContinue where this left off:"
          f"\n  MICRODUCK_GPU={GPU} modal run --detach modal_app.py::main "
          f"--task {task} --resume-run '.*{run_name}'")


def _plan(budget: float, s_per_iter: str) -> None:
    """Cost table. Pure arithmetic — imports modal but calls nothing."""
    measured = {}
    for part in filter(None, s_per_iter.split(",")):
        k, _, v = part.partition("=")
        measured[k.strip()] = float(v)

    print(f"\nBudget ${budget:.2f}, assuming {ASSUME_CPU:g} cores + {ASSUME_MEM_GIB:g} GiB "
          f"billed alongside the GPU, {1 - SAFETY:.0%} safety margin\n")
    head = f"{'GPU':<12} {'$/hr':>7} {'runtime':>9} {'timeout':>9}"
    if measured:
        head += f" {'s/iter':>8} {'iterations':>11}"
    print(head + "\n" + "-" * len(head))
    for gpu in GPU_USD_S:
        try:
            t = budget_to_timeout(budget, gpu)
        except ValueError:
            continue
        row = f"{gpu:<12} {usd_per_second(gpu) * 3600:>7.2f} {t / 3600:>8.1f}h {t:>8d}s"
        if measured:
            spi = measured.get(gpu)
            row += f" {spi:>8.2f} {int(t / spi):>11,}" if spi else f" {'-':>8} {'-':>11}"
        print(row)
    if not measured:
        print("\nNo throughput numbers yet — run `modal run modal_app.py::probe` first.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["plan"])
    ap.add_argument("--budget", type=float, default=BUDGET_USD)
    ap.add_argument("--s-per-iter", default="",
                    help='e.g. "L4=1.02,A100-40GB=0.31" from `probe`')
    a = ap.parse_args()
    _plan(a.budget, a.s_per_iter)
