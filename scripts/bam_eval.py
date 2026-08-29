"""Score a policy under the BAM actuator, against the same acceptance criteria.

`eval_sprint_jump.py` and everything built on it run the policy through
`infer_policy`, which models the XL330 as a MuJoCo position servo with the
current limit applied as a force clip. Its own comment calls that an
approximation, and the audit measured the size of it: **the same ONNX does
0.532 m/s there and 0.314 m/s under BAM at the same forced command.** Domain
randomisation accounts for 0.037 of that; the rest is the actuator. Every speed
number produced by the servo path is therefore optimistic by roughly a third.

Reimplementing BAM's voltage law, Stribeck and load-dependent friction in a raw
MuJoCo loop would be a second approximation to audit. This routes the rollout
through the mjlab training env instead, which already runs the real
`BamActuator` — so the actuator is not a model of the training actuator, it IS
the training actuator. Running many envs at once also gives parallel seeds for
free, so a rate comes with an interval rather than a single anecdote.

Two knobs matter for a fair comparison and both are explicit:

* `--force-cmd` holds the twist command constant. Without it the env samples
  commands at random and a mean forward speed averages toward zero no matter how
  good the policy is — a trap this script hit during the audit.
* `--no-dr` strips the randomisation events, isolating the actuator change. With
  DR on you get the harder, more realistic number.

    uv run scripts/bam_eval.py --policy p.onnx --behavior sprint --force-cmd 0.6,0,0
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv

sys.path.insert(0, str(Path(__file__).parent))
from robustness_eval import _mean_ci, _wilson  # noqa: E402

BEHAVIOR = {
    # behavior: (module, factory, factory-arg, default command)
    "sprint": ("microduck_sprint_env_cfg", "make_microduck_sprint_env_cfg", None,
               (0.6, 0.0, 0.0)),
    "spin2": ("microduck_tricks_env_cfg", "make_microduck_trick_env_cfg",
              "spin_two_leg", (0.0, 0.0, 3.0)),
    "oneleg": ("microduck_tricks_env_cfg", "make_microduck_trick_env_cfg",
               "one_leg_stand", (0.0, 0.0, 0.0)),
    "spin1": ("microduck_tricks_env_cfg", "make_microduck_trick_env_cfg",
              "spin_one_leg", (0.0, 0.0, 3.0)),
    "jump": ("microduck_jump_env_cfg", "make_microduck_jump_env_cfg", None,
             (0.0, 0.0, 0.0)),
    "headstand": ("microduck_headstand_env_cfg", "make_microduck_headstand_env_cfg",
                  None, (0.0, 0.0, 0.0)),
}

# Randomisation events, stripped by --no-dr to isolate the actuator change.
_DR_EVENT_HINTS = ("randomize", "push", "com", "mass", "friction", "armature",
                   "encoder", "imu", "damping", "kp", "kd")

# ...but NOT these. `expand_bam_friction_fields` matches "friction" while being
# required plumbing rather than randomisation: BAM writes per-env
# dof_frictionloss and refuses to run if the field was never expanded. AGENTS.md
# calls it out as mandatory for any standalone env cfg, and stripping it turns
# --no-dr into a hard RuntimeError instead of a cleaner comparison.
_DR_EVENT_KEEP = ("expand",)


def build_env(behavior: str, num_envs: int, no_dr: bool) -> ManagerBasedRlEnv:
    mod_name, fn_name, arg, _ = BEHAVIOR[behavior]
    mod = importlib.import_module(f"mjlab_microduck.tasks.{mod_name}")
    cfg = getattr(mod, fn_name)(arg) if arg else getattr(mod, fn_name)()
    cfg.scene.num_envs = num_envs
    if no_dr:
        removed = [n for n in cfg.events
                   if any(h in n.lower() for h in _DR_EVENT_HINTS)
                   and not any(k in n.lower() for k in _DR_EVENT_KEEP)]
        for name in removed:
            del cfg.events[name]
        # Curricula that widen a DR range hold the event's NAME and look it up
        # on the manager, so deleting an event without its curriculum raises
        # "Event term 'randomize_com' not found in active terms" at reset.
        for cname in [c for c, t in cfg.curriculum.items()
                      if any(str(v) in removed for v in (t.params or {}).values())
                      or any(h in c.lower() for h in _DR_EVENT_HINTS)]:
            del cfg.curriculum[cname]
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def _yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run(policy: str, behavior: str, envs: int, seconds: float,
        cmd: tuple[float, float, float] | None, no_dr: bool) -> dict:
    from mjlab_microduck.tasks import mdp as microduck_mdp

    # mjlab prints manager tables on every construction; a sweep builds one env
    # per row and the tables bury the results.
    with contextlib.redirect_stdout(io.StringIO()):
        env = build_env(behavior, envs, no_dr)
    sess = ort.InferenceSession(policy, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    robot = env.scene["robot"]

    obs_d, _ = env.reset()
    obs = obs_d["actor"] if isinstance(obs_d, dict) else obs_d

    n = envs
    steps = int(seconds / env.step_dt)
    fell = torch.zeros(n, dtype=torch.bool)
    net_yaw = torch.zeros(n)
    prev_yaw = _yaw(robot.data.root_link_quat_w)
    vx_sum = torch.zeros(n)
    tilt_max = torch.zeros(n)
    single = torch.zeros(n)
    airborne = torch.zeros(n)
    hold = torch.zeros(n)
    best_hold = torch.zeros(n)
    z0 = robot.data.root_link_pos_w[:, 2].clone()
    peak = z0.clone()

    for _ in range(steps):
        if cmd is not None:
            c = env.command_manager.get_command("twist")
            c[:, 0], c[:, 1], c[:, 2] = cmd
            obs_d = env.observation_manager.compute()
            obs = obs_d["actor"] if isinstance(obs_d, dict) else obs_d
        acts = np.concatenate([
            sess.run(None, {iname: obs[i:i + 1].cpu().numpy().astype(np.float32)})[0]
            for i in range(n)
        ])
        obs_d, *_ = env.step(torch.as_tensor(acts, dtype=torch.float32))
        obs = obs_d["actor"] if isinstance(obs_d, dict) else obs_d

        q = robot.data.root_link_quat_w
        cur = _yaw(q)
        d = (cur - prev_yaw + math.pi) % (2 * math.pi) - math.pi
        net_yaw += d
        prev_yaw = cur

        cos_tilt = (1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1, 1)
        tilt_max = torch.maximum(tilt_max, torch.rad2deg(torch.acos(cos_tilt)))

        vb = getattr(robot.data, "root_link_lin_vel_b", robot.data.root_link_lin_vel_w)
        vx_sum += vb[:, 0]
        z = robot.data.root_link_pos_w[:, 2]
        peak = torch.maximum(peak, z)
        fell |= z < 0.05

        l, r = microduck_mdp._feet_contact_pair(env, "feet_ground_contact")
        one = (l ^ r)
        single += one.float()
        airborne += (~l & ~r).float()
        hold = torch.where(one, hold + env.step_dt, torch.zeros_like(hold))
        best_hold = torch.maximum(best_hold, hold)

    return {
        "fell": fell.numpy(),
        "net_rev": (net_yaw / (2 * math.pi)).numpy(),
        "vx": (vx_sum / steps).numpy(),
        "tilt_max": tilt_max.numpy(),
        "single_frac": (single / steps).numpy(),
        "airborne_frac": (airborne / steps).numpy(),
        "best_hold": best_hold.numpy(),
        "peak_gain_cm": ((peak - z0) * 100).numpy(),
    }


CRITERIA = {
    "sprint": [("no fall", lambda r: ~r["fell"]),
               ("vx >= 0.30 m/s (BAM-calibrated)", lambda r: r["vx"] >= 0.30),
               ("max tilt <= 20 deg", lambda r: r["tilt_max"] <= 20.0)],
    "spin2": [("no fall", lambda r: ~r["fell"]),
              ("net rotation >= 1.5 rev", lambda r: np.abs(r["net_rev"]) >= 1.5),
              ("airborne <= 20%", lambda r: r["airborne_frac"] <= 0.20)],
    "oneleg": [("single-support hold >= 2.0 s", lambda r: r["best_hold"] >= 2.0),
               ("no fall", lambda r: ~r["fell"])],
    "spin1": [("single-support hold >= 2.0 s", lambda r: r["best_hold"] >= 2.0),
              ("no fall", lambda r: ~r["fell"])],
    "jump": [("peak gain >= 2.0 cm", lambda r: r["peak_gain_cm"] >= 2.0),
             ("max tilt <= 30 deg", lambda r: r["tilt_max"] <= 30.0)],
    "headstand": [("no fall", lambda r: ~r["fell"])],
}


def run_sweep(a) -> int:
    """DR off vs on, plus a command sweep for sprint.

    These are the axes mjlab exposes cleanly. Per-axis physical perturbation
    (friction, mass, current limit, sensor faults) stays on
    `robustness_eval.py`: mjlab simulates through a Warp model built at
    construction, so mutating `env.sim.mj_model` afterwards would not reliably
    reach the simulation and a sweep built on it would silently report the
    unperturbed number. The servo path's ABSOLUTE values are biased, but its
    comparisons ACROSS conditions are still the honest way to find which
    perturbation breaks a policy.
    """
    cmds = ([(0.3, 0, 0), (0.45, 0, 0), (0.6, 0, 0), (0.75, 0, 0)]
            if a.behavior == "sprint" else [BEHAVIOR[a.behavior][3]])
    print(f"\n{Path(a.policy).name}  behavior={a.behavior}  BAM  "
          f"{a.envs} envs x {a.seconds:.0f}s\n")
    print(f"{'DR':>4} {'cmd':>18} {'vx':>18} {'|net rev|':>10} {'no-fall':>8} {'tilt':>6}")
    print("-" * 70)
    for no_dr in (True, False):
        for cmd in cmds:
            r = run(a.policy, a.behavior, a.envs, a.seconds, cmd, no_dr)
            vx, lo, hi = _mean_ci(list(r["vx"]))
            nf = int((~r["fell"]).sum())
            print(f"{'off' if no_dr else 'on':>4} {str(cmd):>18} "
                  f"{vx:+.3f} [{lo:+.3f},{hi:+.3f}] {np.abs(r['net_rev']).mean():>10.2f} "
                  f"{nf:>4}/{len(r['fell']):<3} {r['tilt_max'].mean():>6.1f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--behavior", required=True, choices=list(BEHAVIOR))
    ap.add_argument("--envs", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--force-cmd", default=None, help="vx,vy,wz (default: behaviour's)")
    ap.add_argument("--sampled-cmd", action="store_true",
                    help="let the env sample commands instead of forcing one")
    ap.add_argument("--no-dr", action="store_true",
                    help="strip randomisation events to isolate the actuator")
    ap.add_argument("--sweep", action="store_true",
                    help="run DR off and DR on, and (for sprint) a command sweep")
    a = ap.parse_args()

    if a.sweep:
        return run_sweep(a)

    cmd = None
    if not a.sampled_cmd:
        cmd = (tuple(float(x) for x in a.force_cmd.split(","))
               if a.force_cmd else BEHAVIOR[a.behavior][3])

    r = run(a.policy, a.behavior, a.envs, a.seconds, cmd, a.no_dr)

    print(f"\n{Path(a.policy).name}  behavior={a.behavior}  BAM actuator  "
          f"{a.envs} envs x {a.seconds:.0f}s  cmd={cmd or 'sampled'}  "
          f"DR={'off' if a.no_dr else 'on'}\n")
    vx, lo, hi = _mean_ci(list(r["vx"]))
    rev, rlo, rhi = _mean_ci(list(np.abs(r["net_rev"])))
    print(f"  body vx        : {vx:+.3f} m/s  (95% CI {lo:+.3f} to {hi:+.3f})")
    print(f"  |net rotation| : {rev:.2f} rev   (95% CI {rlo:.2f} to {rhi:.2f})")
    print(f"  max tilt       : {r['tilt_max'].mean():.1f} deg")
    print(f"  best hold      : {r['best_hold'].mean():.2f} s "
          f"(max {r['best_hold'].max():.2f})")
    print(f"  peak gain      : {r['peak_gain_cm'].mean():.1f} cm")
    print()
    for label, test in CRITERIA[a.behavior]:
        ok = np.asarray(test(r), dtype=bool)
        k = int(ok.sum())
        wlo, whi = _wilson(k, len(ok))
        verdict = "PASS" if wlo >= 0.80 else "MARGINAL" if k / len(ok) >= 0.80 else "FAIL"
        print(f"  [{verdict:<8}] {label:<34} {k}/{len(ok)} = {k / len(ok) * 100:.0f}%"
              f"  (95% CI {wlo * 100:.0f}-{whi * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
