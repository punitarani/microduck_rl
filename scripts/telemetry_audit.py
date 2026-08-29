"""Per-policy telemetry: the failure modes that a success rate hides.

A policy can pass every acceptance criterion and still be unshippable. It can
saturate its servos, chatter at the control rate, slip its feet, load one leg
twice as hard as the other, or land with impacts the hardware will not survive.
None of that shows up in "did it fall" — and all of it shows up on a real robot.

What is measured, and why each one matters on hardware:

- **joint saturation**: fraction of control steps where a commanded target sits
  within 5% of the joint's mechanical limit. The playbook has a whole note about
  joints parking on limits; a saturated joint has no authority left to react.
- **action jitter**: RMS change in commanded target per control step. This is
  what the action-rate penalty exists to suppress; on an XL330 it is heat and
  gear wear, and it was 7.04 reward-units of cost in the jump.
- **foot slip**: tangential foot speed while that foot is loaded. Slip means the
  gait is relying on friction the real floor may not have.
- **left/right asymmetry**: relative difference in per-leg loading and joint
  travel. A gait that is 30% asymmetric in sim usually walks in a circle in
  reality, which is exactly the sprint's measured defect.
- **impact spikes**: peak |a_z| of the trunk. The proxy for landing violence.
- **actuator effort**: fraction of the current limit actually used, which says
  how much margin a weaker battery would eat.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_sprint_jump import (  # noqa: E402
    CONTROL_DECIMATION,
    TIMESTEP,
    _build,
    _reset_to_stance,
    _trunk_body_id,
)

LEFT_JOINTS = slice(0, 5)    # hip_yaw, hip_roll, hip_pitch, knee, ankle
RIGHT_JOINTS = slice(9, 14)


def audit(policy_path: str, cmd, seconds: float, settle: float = 1.0) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        model, data, policy = _build(policy_path)
    trunk = _trunk_body_id(model)
    _reset_to_stance(model, data, policy)

    lo = model.jnt_range[:, 0].copy()
    hi = model.jnt_range[:, 1].copy()
    qadr = np.array(policy.joint_qpos_indices)
    # Map each actuated joint to its range via the qpos address.
    jrange = []
    for qi in qadr:
        j = int(np.argmax(model.jnt_qposadr == qi))
        jrange.append((lo[j], hi[j]))
    jrange = np.array(jrange)
    span = np.maximum(jrange[:, 1] - jrange[:, 0], 1e-6)

    for k in range(int(settle / TIMESTEP)):
        if k % CONTROL_DECIMATION == 0:
            policy.vel_cmd[:] = 0.0
            policy._update_command()
            policy.apply_action(policy.infer())
        mujoco.mj_step(model, data)

    policy.vel_cmd[:] = np.asarray(cmd, dtype=np.float32)
    prev_ctrl = data.ctrl.copy()
    prev_vz = 0.0
    sat = jit = 0
    ctrl_steps = 0
    slips, az, effort = [], [], []
    load_l, load_r = [], []
    travel_l, travel_r = [], []
    wrench = np.zeros(6)
    flim = model.actuator_forcerange[:, 1].copy()

    for k in range(int(seconds / TIMESTEP)):
        if k % CONTROL_DECIMATION == 0:
            policy._update_command()
            policy.apply_action(policy.infer())
            c = data.ctrl.copy()
            near = ((c - jrange[:, 0]) < 0.05 * span) | ((jrange[:, 1] - c) < 0.05 * span)
            sat += int(near.sum())
            jit += float(np.sqrt(np.mean((c - prev_ctrl) ** 2)))
            prev_ctrl = c
            ctrl_steps += 1
            travel_l.append(np.abs(c[LEFT_JOINTS]).sum())
            travel_r.append(np.abs(c[RIGHT_JOINTS]).sum())
        mujoco.mj_step(model, data)

        vz = float(data.cvel[trunk][5])
        az.append(abs(vz - prev_vz) / TIMESTEP)
        prev_vz = vz
        effort.append(float(np.mean(np.abs(data.actuator_force) / np.maximum(flim, 1e-9))))

        L = R = 0.0
        for i in range(data.ncon):
            con = data.contact[i]
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                     for g in (con.geom1, con.geom2)]
            side = ("L" if any("left_foot" in n for n in names)
                    else "R" if any("right_foot" in n for n in names) else None)
            if side is None:
                continue
            mujoco.mj_contactForce(model, data, i, wrench)
            f = abs(float(wrench[0]))
            if side == "L":
                L += f
            else:
                R += f
            if f > 0.4:  # loaded: slip here is real slip, not a swinging foot
                slips.append(float(np.linalg.norm(wrench[1:3])))
        load_l.append(L)
        load_r.append(R)

    n_ctrl = max(ctrl_steps, 1)
    ml, mr = float(np.mean(load_l)), float(np.mean(load_r))
    tl, tr = float(np.mean(travel_l)), float(np.mean(travel_r))
    return {
        "joint_saturation_pct": 100.0 * sat / (n_ctrl * 14),
        "action_jitter_rms": jit / n_ctrl,
        "foot_slip_mean_N": float(np.mean(slips)) if slips else 0.0,
        "load_asym_pct": 100.0 * abs(ml - mr) / max(ml + mr, 1e-9),
        "travel_asym_pct": 100.0 * abs(tl - tr) / max(tl + tr, 1e-9),
        "impact_peak_az": float(np.percentile(az, 99.5)),
        "effort_mean_pct": 100.0 * float(np.mean(effort)),
        "effort_p95_pct": 100.0 * float(np.percentile(effort, 95)),
    }


POLICIES = [
    ("sprint-fastest", "runs/BEST/sprint-fastest.onnx", (0.6, 0, 0), 10.0),
    ("sprint-steerable", "runs/BEST/sprint-steerable.onnx", (0.6, 0, 0), 10.0),
    ("spin-two-leg", "runs/BEST/spin-two-leg.onnx", (0, 0, 3.0), 10.0),
    ("headstand(v1 tripod)", "runs/BEST/headstand.onnx", (0, 0, 0), 10.0),
    ("jump-highest", "runs/BEST/jump-highest-unstable.onnx", (0, 0, 0), 10.0),
    ("jump-stable", "runs/BEST/jump-stable-shallow.onnx", (0, 0, 0), 10.0),
    ("one-leg-stand", "runs/BEST/one-leg-stand.onnx", (0, 0, 0), 10.0),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()

    cols = ["joint_saturation_pct", "action_jitter_rms", "foot_slip_mean_N",
            "load_asym_pct", "travel_asym_pct", "impact_peak_az",
            "effort_mean_pct", "effort_p95_pct"]
    short = ["sat%", "jitter", "slip N", "loadAsym%", "travAsym%", "peak|az|",
             "effort%", "effP95%"]
    print(f"\n{'policy':<22} " + " ".join(f"{s:>9}" for s in short))
    print("-" * (22 + 10 * len(short)))
    for name, path, cmd, secs in POLICIES:
        if a.only and a.only not in name:
            continue
        if not Path(path).exists():
            print(f"{name:<22} (missing)")
            continue
        r = audit(path, cmd, secs)
        print(f"{name:<22} " + " ".join(f"{r[c]:>9.2f}" for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
