"""Multi-condition robustness evaluation with acceptance criteria and CIs.

`eval_sprint_jump.py` answers "what does this policy do" from ONE rollout under
nominal conditions. That is enough to catch a policy that cheats and useless for
deciding whether one is trustworthy: a single clean rollout cannot distinguish a
robust gait from one that works only because the floor friction happened to be
0.9 and nothing pushed it.

This sweeps the axes the training env randomizes — foot friction, mass, actuator
current (battery), IMU misalignment, encoder bias, sensor noise, pushes and
initial pose — at nominal, at the edge of the training range, and BEYOND it.
Inside-range performance says the policy learned its task; outside-range
behaviour is the only cheap evidence about sim-to-real.

Every behaviour has an explicit acceptance criterion (see ACCEPTANCE below) and
every rate is reported with a Wilson 95% interval, because "9/10 trials passed"
and "90% success" are very different claims at this sample size.

    uv run scripts/robustness_eval.py --policy runs/BEST/sprint-fastest.onnx \\
        --behavior sprint --trials 20
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_sprint_jump import (  # noqa: E402
    CONTROL_DECIMATION,
    TIMESTEP,
    _body_forward_speed,
    _foot_contacts,
    _reset_to_stance,
    _trunk_body_id,
    _trunk_tilt_deg,
    _yaw_rate,
    _yaw_angle,
    _build,
    _site_z,
)

# ── observation layout, as infer_policy assembles it ─────────────────────────
OBS_GYRO = slice(0, 3)
OBS_GRAVITY = slice(3, 6)
OBS_JOINT_POS = slice(6, 20)
OBS_JOINT_VEL = slice(20, 34)

# ── training ranges, from microduck_velocity_env_cfg ─────────────────────────
TRAIN = {
    "foot_friction": (0.7, 1.3),
    "mass_scale": (0.95, 1.05),
    "imu_tilt_deg": (0.0, 6.0),
    "encoder_bias": (-0.015, 0.015),
    "push_mps": (-0.3, 0.3),
    "current_a": (1.75, 1.75),   # nominal; battery sag is inside BAM, not here
}


@dataclass
class Condition:
    """One perturbation setting. `nominal` reproduces the single-rollout eval."""

    name: str
    foot_friction: float = 1.0
    mass_scale: float = 1.0
    current_a: float = 1.75
    imu_tilt_deg: float = 0.0
    encoder_bias: float = 0.0
    gyro_noise: float = 0.0
    joint_vel_noise: float = 0.0
    push_mps: float = 0.0
    init_joint_noise: float = 0.0
    init_tilt_deg: float = 0.0


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval — correct at small n and at rates near 0 or 1, where
    the normal approximation returns intervals outside [0, 1]."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mean_ci(xs: list[float], z: float = 1.96) -> tuple[float, float, float]:
    """(mean, lo, hi) with a normal 95% interval on the mean."""
    if not xs:
        return (float("nan"),) * 3
    a = np.asarray(xs, dtype=float)
    m = float(a.mean())
    if len(a) < 2:
        return (m, m, m)
    half = z * float(a.std(ddof=1)) / math.sqrt(len(a))
    return (m, m - half, m + half)


def _apply_condition(model, policy, cond: Condition, rng: np.random.Generator):
    """Perturb the model and wrap the observation builder for this trial."""
    # Foot friction: sliding component of the two foot geoms.
    for gname in ("left_foot_collision", "right_foot_collision"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid >= 0:
            model.geom_friction[gid, 0] = cond.foot_friction

    # Mass (and inertia with it, as the training DR does — scaling mass alone
    # would make a heavier robot implausibly easy to rotate).
    model.body_mass[:] = model.body_mass * cond.mass_scale
    model.body_inertia[:] = model.body_inertia * cond.mass_scale

    # Actuator current limit: the battery proxy. Torque = kt * I, so a sagging
    # pack is a lower force ceiling.
    if cond.current_a > 0:
        from bam.model import load_model

        kt = load_model(motor_name="xl330", model="m6").kt.value
        model.actuator_forcerange[:, 0] = -kt * cond.current_a
        model.actuator_forcerange[:, 1] = kt * cond.current_a
        model.actuator_forcelimited[:] = 1

    # Sensor corruption, applied where the robot would actually see it: a
    # constant per-trial encoder offset, a fixed IMU mounting tilt, and
    # zero-mean noise on gyro and joint velocity.
    bias = rng.normal(0.0, cond.encoder_bias, 14) if cond.encoder_bias else np.zeros(14)
    tilt = math.radians(cond.imu_tilt_deg)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-9
    original = policy.get_observations

    def noisy():
        obs = original().copy()
        if cond.imu_tilt_deg:
            g = obs[OBS_GRAVITY]
            k = axis
            obs[OBS_GRAVITY] = (
                g * math.cos(tilt)
                + np.cross(k, g) * math.sin(tilt)
                + k * np.dot(k, g) * (1 - math.cos(tilt))
            )
        obs[OBS_JOINT_POS] += bias
        if cond.gyro_noise:
            obs[OBS_GYRO] += rng.normal(0.0, cond.gyro_noise, 3)
        if cond.joint_vel_noise:
            obs[OBS_JOINT_VEL] += rng.normal(0.0, cond.joint_vel_noise, 14)
        return obs.astype(np.float32)

    policy.get_observations = noisy


def _perturb_start(model, data, policy, cond: Condition, rng: np.random.Generator):
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = model.jnt_qposadr[fj]
    if cond.init_joint_noise:
        for qi in policy.joint_qpos_indices:
            data.qpos[qi] += rng.normal(0.0, cond.init_joint_noise)
    if cond.init_tilt_deg:
        a = math.radians(rng.uniform(-cond.init_tilt_deg, cond.init_tilt_deg))
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax) + 1e-9
        data.qpos[adr + 3:adr + 7] = [
            math.cos(a / 2), *(ax * math.sin(a / 2))
        ]
    mujoco.mj_forward(model, data)


def rollout(policy_path: str, behavior: str, cond: Condition, seed: int,
            seconds: float, cmd: np.ndarray) -> dict:
    """One trial. Returns raw per-trial measurements; scoring happens above."""
    rng = np.random.default_rng(seed)
    # PolicyInference prints a load banner; at hundreds of rollouts that is
    # noise that hides the table.
    with contextlib.redirect_stdout(io.StringIO()):
        model, data, policy = _build(policy_path)
    _apply_condition(model, policy, cond, rng)
    trunk = _trunk_body_id(model)
    _reset_to_stance(model, data, policy)
    _perturb_start(model, data, policy, cond, rng)

    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    vadr = model.jnt_dofadr[fj]
    # Jump measures from t=0 (its optimum jumps out of the reset transient);
    # everything else settles first so t=0 is a real stance.
    settle = 0.0 if behavior == "jump" else 1.0
    for k in range(int(settle / TIMESTEP)):
        if k % CONTROL_DECIMATION == 0:
            policy.vel_cmd[:] = 0.0
            policy._update_command()
            policy.apply_action(policy.infer())
        mujoco.mj_step(model, data)

    policy.vel_cmd[:] = cmd.astype(np.float32)
    stance_z = float(data.xpos[trunk][2])
    head0 = _site_z(model, data, "mouth_tip")

    speeds, yaws, tilts, invs = [], [], [], []
    net_yaw = 0.0
    prev_yaw = _yaw_angle(model, data)
    single = both = airborne = 0
    hold = best_hold = 0.0
    inv_hold = best_inv_hold = 0.0
    peak = stance_z
    flights = 0
    was_air = False
    fell_at = None
    push_every = int(4.0 / TIMESTEP)
    n = int(seconds / TIMESTEP)

    for k in range(n):
        if k % CONTROL_DECIMATION == 0:
            policy._update_command()
            policy.apply_action(policy.infer())
        if cond.push_mps and k and k % push_every == 0:
            d = rng.normal(size=2)
            d /= np.linalg.norm(d) + 1e-9
            data.qvel[vadr:vadr + 2] += d * abs(cond.push_mps)
        mujoco.mj_step(model, data)

        z = float(data.xpos[trunk][2])
        peak = max(peak, z)
        tilt = _trunk_tilt_deg(model, data)
        tilts.append(tilt)
        speeds.append(_body_forward_speed(model, data))
        yaws.append(_yaw_rate(model, data))
        cur_yaw = _yaw_angle(model, data)
        net_yaw += (cur_yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
        prev_yaw = cur_yaw

        l, r = _foot_contacts(model, data)
        if l ^ r:
            single += 1
            hold += TIMESTEP
            best_hold = max(best_hold, hold)
        else:
            hold = 0.0
            both += 1 if (l and r) else 0
            if not (l or r):
                airborne += 1
        air = not (l or r)
        if was_air and not air:
            flights += 1
        was_air = air

        head_z = _site_z(model, data, "mouth_tip")
        feet_z = (_site_z(model, data, "left_foot") + _site_z(model, data, "right_foot")) / 2
        inv = feet_z - head_z
        invs.append(inv)
        if inv > 0.05 and head_z < 0.06:
            inv_hold += TIMESTEP
            best_inv_hold = max(best_inv_hold, inv_hold)
        else:
            inv_hold = 0.0

        if z < 0.05 and fell_at is None:
            fell_at = k * TIMESTEP

    steady = slice(int(1.0 / TIMESTEP), None)
    return {
        "fell": fell_at is not None,
        "fell_at": fell_at,
        "vx": float(np.mean(speeds[steady] or speeds)),
        "yaw": float(np.mean(yaws[steady] or yaws)),
        "abs_yaw": float(np.mean(np.abs(yaws[steady] or yaws))),
        "net_rev": net_yaw / (2 * math.pi),
        "net_yaw_rate": net_yaw / max(seconds, 1e-9),
        "max_tilt": float(max(tilts)),
        "mean_tilt": float(np.mean(tilts)),
        "end_tilt": float(tilts[-1]),
        "single_frac": single / n,
        "double_frac": both / n,
        "airborne_frac": airborne / n,
        "best_hold_s": best_hold,
        "peak_gain_cm": (peak - stance_z) * 100,
        "flights": flights,
        "best_inv_hold_s": best_inv_hold,
        "max_inversion": float(max(invs)),
        "stance_head_z": head0,
    }


# ── acceptance criteria: measurable, per behaviour ───────────────────────────
@dataclass
class Criterion:
    key: str
    label: str
    test: object
    kind: str = "rate"  # "rate" = fraction of trials passing


ACCEPTANCE: dict[str, list[Criterion]] = {
    "sprint": [
        Criterion("upright", "does not fall in 10 s", lambda r: not r["fell"]),
        Criterion("speed", "body vx >= 0.45 m/s at cmd 0.6", lambda r: r["vx"] >= 0.45),
        Criterion("straight", "|yaw| <= 0.15 rad/s at zero yaw cmd",
                  lambda r: abs(r["yaw"]) <= 0.15),
        Criterion("tilt", "max trunk tilt <= 20 deg", lambda r: r["max_tilt"] <= 20.0),
    ],
    "spin2": [
        Criterion("upright", "does not fall in 12 s", lambda r: not r["fell"]),
        # NET rotation, not |rate|: the run-1 spin passed a |rate| criterion at
        # 6.5 rad/s while turning 0.04 revolutions in 10 s.
        Criterion("net_spin", "net rotation >= 1.5 rev in 12 s",
                  lambda r: abs(r["net_rev"]) >= 1.5),
        # WAS "double-support >= 80%". That criterion was wrong on physics, not
        # on the policy: a biped cannot rotate with both feet planted without
        # slipping, so it has to step. The run-1 spin passed it at 94% precisely
        # BECAUSE it was not rotating. What the criterion was really guarding
        # against is hopping, so it guards against that directly now.
        Criterion("grounded", "airborne <= 20% (not hopping)",
                  lambda r: r["airborne_frac"] <= 0.20),
        Criterion("tilt", "max trunk tilt <= 15 deg", lambda r: r["max_tilt"] <= 15.0),
    ],
    "headstand": [
        Criterion("inverts", "reaches inversion > 0.05 m", lambda r: r["max_inversion"] > 0.05),
        Criterion("holds", "continuous headstand >= 5 s", lambda r: r["best_inv_hold_s"] >= 5.0),
    ],
    "jump": [
        Criterion("height", "peak gain >= 2.0 cm", lambda r: r["peak_gain_cm"] >= 2.0),
        Criterion("repeats", ">= 2 distinct flights", lambda r: r["flights"] >= 2),
        Criterion("upright_end", "ends upright (tilt < 30 deg)", lambda r: r["end_tilt"] < 30.0),
    ],
    "oneleg": [
        Criterion("hold", "single-support hold >= 2.0 s", lambda r: r["best_hold_s"] >= 2.0),
        Criterion("upright", "does not fall", lambda r: not r["fell"]),
    ],
}

DEFAULT_CMD = {
    "sprint": np.array([0.6, 0.0, 0.0]),
    "spin2": np.array([0.0, 0.0, 3.0]),
    "headstand": np.array([0.0, 0.0, 0.0]),
    "jump": np.array([0.0, 0.0, 0.0]),
    "oneleg": np.array([0.0, 0.0, 0.0]),
}
DEFAULT_SECONDS = {"sprint": 10.0, "spin2": 12.0, "headstand": 12.0,
                   "jump": 12.0, "oneleg": 12.0}


def conditions(scope: str) -> list[Condition]:
    """nominal -> inside the training envelope -> beyond it."""
    base = [Condition("nominal")]
    inside = [
        Condition("friction_low_train", foot_friction=0.7),
        Condition("friction_high_train", foot_friction=1.3),
        Condition("mass_light_train", mass_scale=0.95),
        Condition("mass_heavy_train", mass_scale=1.05),
        Condition("imu_tilt_6deg", imu_tilt_deg=6.0),
        Condition("encoder_bias_train", encoder_bias=0.015),
        Condition("push_0.3", push_mps=0.3),
        Condition("sensor_noise", gyro_noise=0.05, joint_vel_noise=0.5),
        Condition("start_perturbed", init_joint_noise=0.03, init_tilt_deg=5.0),
    ]
    beyond = [
        Condition("friction_0.4_BEYOND", foot_friction=0.4),
        Condition("friction_1.8_BEYOND", foot_friction=1.8),
        Condition("mass_1.15_BEYOND", mass_scale=1.15),
        Condition("battery_sag_1.2A", current_a=1.2),
        Condition("battery_weak_1.45A", current_a=1.45),
        Condition("imu_tilt_10deg_BEYOND", imu_tilt_deg=10.0),
        Condition("push_0.6_BEYOND", push_mps=0.6),
        Condition("start_tilt_12deg_BEYOND", init_joint_noise=0.05, init_tilt_deg=12.0),
        Condition("combined_worst", foot_friction=0.6, mass_scale=1.10, current_a=1.45,
                  imu_tilt_deg=8.0, encoder_bias=0.02, gyro_noise=0.05,
                  joint_vel_noise=0.5, push_mps=0.4, init_joint_noise=0.04,
                  init_tilt_deg=8.0),
    ]
    return {"nominal": base, "inside": base + inside,
            "all": base + inside + beyond}[scope]


def _servo_path_warning() -> None:
    """Say, every time, that this path is the biased one.

    Measured against the BAM actuator at a matched command: this path reports
    0.532 m/s where BAM gives 0.314 for the same ONNX, and it scored the fixed
    spin at 73% no-fall where BAM gives 100%. It is wrong in DIFFERENT
    DIRECTIONS for different behaviours, so there is no correction factor —
    only a reason to quote `bam_eval.py` instead.

    This path still earns its keep for what BAM cannot do cheaply: recording
    video, and sweeping friction / mass / current / sensor faults per trial.
    """
    print(
        "\n  NOTE: position-servo actuator (infer_policy's approximation).\n"
        "  Speeds here run ~36% HIGH and spin stability ~25% LOW versus BAM.\n"
        "  For numbers to quote, use: uv run scripts/bam_eval.py\n",
        file=sys.stderr,
    )


def main() -> int:
    _servo_path_warning()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--behavior", required=True, choices=list(ACCEPTANCE))
    ap.add_argument("--trials", type=int, default=8, help="seeds per condition")
    ap.add_argument("--scope", default="all", choices=["nominal", "inside", "all"])
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--cmd", default=None, help="vx,vy,wz override")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    secs = a.seconds or DEFAULT_SECONDS[a.behavior]
    cmd = (np.array([float(x) for x in a.cmd.split(",")]) if a.cmd
           else DEFAULT_CMD[a.behavior])
    crits = ACCEPTANCE[a.behavior]

    print(f"\n{Path(a.policy).name}  behavior={a.behavior}  "
          f"{a.trials} seeds x {len(conditions(a.scope))} conditions  cmd={cmd}\n")
    hdr = f"{'condition':<26} {'n':>3} " + " ".join(f"{c.key:>12}" for c in crits) \
          + f" {'vx':>7} {'|yaw|':>7} {'tilt':>6}"
    print(hdr + "\n" + "-" * len(hdr))

    all_rows, per_cond = [], {}
    for cond in conditions(a.scope):
        rs = [rollout(a.policy, a.behavior, cond, 1000 + i, secs, cmd)
              for i in range(a.trials)]
        all_rows.extend(rs)
        per_cond[cond.name] = rs
        cells = []
        for c in crits:
            k = sum(1 for r in rs if c.test(r))
            lo, hi = _wilson(k, len(rs))
            cells.append(f"{k}/{len(rs)} {int(lo * 100):>2}-{int(hi * 100):<2}")
        vx, _, _ = _mean_ci([r["vx"] for r in rs])
        ay, _, _ = _mean_ci([r["abs_yaw"] for r in rs])
        mt, _, _ = _mean_ci([r["max_tilt"] for r in rs])
        print(f"{cond.name:<26} {len(rs):>3} " + " ".join(f"{c:>12}" for c in cells)
              + f" {vx:>7.3f} {ay:>7.2f} {mt:>6.1f}")

    print("\n" + "=" * len(hdr))
    print("OVERALL across every condition:")
    for c in crits:
        k = sum(1 for r in all_rows if c.test(r))
        lo, hi = _wilson(k, len(all_rows))
        verdict = "PASS" if lo >= 0.80 else "MARGINAL" if k / len(all_rows) >= 0.80 else "FAIL"
        print(f"  [{verdict:<8}] {c.label:<40} {k}/{len(all_rows)} "
              f"= {k / len(all_rows) * 100:.0f}%  (95% CI {lo * 100:.0f}-{hi * 100:.0f}%)")

    # A headstand puts the trunk on the floor by design, so the z-threshold
    # fall detector fires on every successful trial. Reporting it there would
    # be a false alarm, which is why the headstand criteria omit it entirely.
    falls = [] if a.behavior == "headstand" else [r for r in all_rows if r["fell"]]
    if a.behavior == "headstand":
        print("\n  (fall detection is z-based and meaningless inverted — omitted)")
    if falls:
        t = [r["fell_at"] for r in falls]
        print(f"\n  falls: {len(falls)}/{len(all_rows)}, "
              f"earliest {min(t):.1f}s, median {float(np.median(t)):.1f}s")
    worst = [] if a.behavior == "headstand" else sorted(per_cond.items(),
                   key=lambda kv: sum(1 for r in kv[1] if r["fell"]), reverse=True)[:3]
    if worst:
        print("  worst conditions by falls: "
              + ", ".join(f"{k}({sum(1 for r in v if r['fell'])}/{len(v)})" for k, v in worst))

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(
            {"policy": a.policy, "behavior": a.behavior,
             "per_condition": {k: v for k, v in per_cond.items()}}, indent=1))
        print(f"\n  raw trials -> {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
