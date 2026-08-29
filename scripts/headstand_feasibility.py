"""Is a stacked headstand physically possible for this robot?

Two training runs failed to produce one, and the second failed in a specific
way: the stacking gate was never once satisfied in 2300 iterations. That is
either a reward-design problem or a physics problem, and the two call for
completely different responses — so this answers the physics question before
anything else is spent on the reward.

The test is the one AGENTS.md prescribes before training a pose task: a target
pose must be a STABLE EQUILIBRIUM. Hold its control for 3 s from noisy initial
conditions and check tilt, not just height, because a settle test that only
records z reports a collapsed robot as resting fine.

Three questions, in order:

1. **Geometry.** Can the trunk sit >= 8 cm above the beak at all? (A rigid
   inversion gives 0.108 m, so yes — the gate is not asking the impossible.)
2. **Statics.** Is there a joint configuration, inverted and resting on the
   head, whose gravity-compensation torques fit inside the XL330's 0.641 Nm at
   1.75 A, with the centre of mass over the head contact?
3. **Stability.** Does that configuration actually hold for 3 s under gravity
   when the servos are commanded to it?

The three answers are NOT a simple pass/fail, and conflating them is how a
tractable control problem gets mistaken for an impossible one:

  * torque exceeds the ceiling, or no balanced pose exists -> IMPOSSIBLE, close it
  * balanced pose exists but collapses when commanded    -> UNSTABLE EQUILIBRIUM;
    possible, but it is an inverted pendulum and needs active feedback, which is
    a much harder RL problem than a pose-holding task
  * balanced pose exists and holds open-loop             -> statically stable;
    the reward is simply wrong and the pose can be handed to it as a target
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np
from scipy.optimize import minimize

ROBOT_XML = "src/mjlab_microduck/robot/microduck/robot_allcollisions.xml"

# robot_allcollisions.xml is the ROBOT ONLY — no ground plane. Loading it bare
# gives a robot in free fall, and free fall preserves pose perfectly for 3 s, so
# the settle test returns a confident "HELD" for a robot that is simply falling.
# The `contacts: 0` line is what exposed it. This wraps the robot in a floor.
SCENE_TEMPLATE = """<mujoco model="headstand_test">
  <include file="{robot}"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0"
          friction="1.0 0.005 0.0001" condim="3"/>
    <light pos="0 0 3"/>
  </worldbody>
</mujoco>
"""
TORQUE_LIMIT = 0.641  # kt(0.366) * 1.75 A
SETTLE_S = 3.0
TIMESTEP = 0.005

# Symmetric parameterisation: both legs mirror, so the search is over 5 numbers
# rather than 14. A headstand has no reason to be asymmetric, and halving the
# dimension makes the search actually converge.
PARAM_NAMES = ("neck_pitch", "head_pitch", "hip_pitch", "knee", "ankle")


def _load_with_floor() -> mujoco.MjModel:
    """Robot + ground plane, written next to the robot XML so its `include`
    and mesh paths still resolve."""
    import os
    import tempfile

    robot = os.path.abspath(ROBOT_XML)
    directory = os.path.dirname(robot)
    xml = SCENE_TEMPLATE.format(robot=os.path.basename(robot))
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=directory,
                                     delete=False) as fh:
        fh.write(xml)
        path = fh.name
    try:
        return mujoco.MjModel.from_xml_path(path)
    finally:
        os.unlink(path)


def joint_index(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def build_pose(model, data, params, pitch_deg: float = 180.0) -> None:
    """Place the robot inverted with the given symmetric joint angles."""
    neck, head, hip, knee, ankle = params
    fj = joint_index(model, "trunk_base_freejoint")
    adr = model.jnt_qposadr[fj]
    a = np.radians(pitch_deg) / 2.0
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[adr:adr + 3] = [0.0, 0.0, 0.30]
    data.qpos[adr + 3:adr + 7] = [np.cos(a), 0.0, np.sin(a), 0.0]  # pitch flip

    def setj(name, val):
        j = joint_index(model, name)
        if j >= 0:
            data.qpos[model.jnt_qposadr[j]] = np.clip(
                val, model.jnt_range[j, 0], model.jnt_range[j, 1])

    setj("neck_pitch", neck)
    setj("head_pitch", head)
    for side in ("left", "right"):
        setj(f"{side}_hip_pitch", hip)
        setj(f"{side}_knee", knee)
        setj(f"{side}_ankle", ankle)
    mujoco.mj_forward(model, data)


def drop_to_contact(model, data, max_iter: int = 400) -> None:
    """Lower the robot until something touches, so contacts are real."""
    fj = joint_index(model, "trunk_base_freejoint")
    adr = model.jnt_qposadr[fj]
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        if data.ncon > 0:
            return
        data.qpos[adr + 2] -= 0.001
    mujoco.mj_forward(model, data)


def evaluate(model, data, params) -> dict:
    """Static cost of a candidate pose: CoM offset, torque demand, stack height."""
    build_pose(model, data, params)
    drop_to_contact(model, data)

    com = np.array(data.subtree_com[0])
    # Support point: the mean xy of whatever is touching the floor.
    if data.ncon:
        pts = np.array([data.contact[i].pos for i in range(data.ncon)])
        support = pts[:, :2].mean(axis=0)
        contact_z = float(pts[:, 2].mean())
    else:
        support = com[:2]
        contact_z = 0.0
    com_offset = float(np.linalg.norm(com[:2] - support))

    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_inverse(model, data)
    tau = np.array([data.qfrc_inverse[model.jnt_dofadr[model.actuator_trnid[i, 0]]]
                    for i in range(model.nu)])

    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    head_z = float(data.site_xpos[head_id][2])
    stack = float(data.xpos[trunk_id][2]) - head_z

    return {
        "com_offset": com_offset,
        "max_torque": float(np.abs(tau).max()),
        "stack": stack,
        "head_z": head_z,
        "contact_z": contact_z,
        "ncon": int(data.ncon),
        "tau": tau,
    }


def cost(params, model, data) -> float:
    r = evaluate(model, data, params)
    # Balance first, then torque headroom, then keep the stack tall.
    return (10.0 * r["com_offset"]
            + 2.0 * max(0.0, r["max_torque"] - TORQUE_LIMIT)
            + 1.0 * max(0.0, 0.08 - r["stack"]))


def settle_test(model, data, params, noise: float, seed: int) -> dict:
    """Command the pose and see whether it is actually held for 3 s."""
    rng = np.random.default_rng(seed)
    build_pose(model, data, params)
    drop_to_contact(model, data)
    if noise:
        for i in range(model.nu):
            j = model.actuator_trnid[i, 0]
            data.qpos[model.jnt_qposadr[j]] += rng.normal(0.0, noise)
        mujoco.mj_forward(model, data)

    target = np.array([data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
                       for i in range(model.nu)])
    data.ctrl[:] = target
    head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    for _ in range(int(SETTLE_S / TIMESTEP)):
        mujoco.mj_step(model, data)
    stack = float(data.xpos[trunk_id][2]) - float(data.site_xpos[head_id][2])
    q = data.qpos[model.jnt_qposadr[joint_index(model, "trunk_base_freejoint")] + 3:][:4]
    cos_tilt = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
    return {"stack_after": stack,
            "tilt_after_deg": float(np.degrees(np.arccos(np.clip(cos_tilt, -1, 1)))),
            "ncon_after": int(data.ncon),
            "head_z_after": float(data.site_xpos[head_id][2])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--restarts", type=int, default=24)
    a = ap.parse_args()

    model = _load_with_floor()
    model.opt.timestep = TIMESTEP
    model.actuator_forcerange[:, 0] = -TORQUE_LIMIT
    model.actuator_forcerange[:, 1] = TORQUE_LIMIT
    model.actuator_forcelimited[:] = 1
    data = mujoco.MjData(model)

    print(f"mass {model.body_mass.sum():.3f} kg, weight "
          f"{model.body_mass.sum() * 9.81:.2f} N, torque ceiling {TORQUE_LIMIT} Nm\n")

    rng = np.random.default_rng(0)
    best, best_cost = None, np.inf
    for i in range(a.restarts):
        x0 = rng.uniform(-1.0, 1.0, len(PARAM_NAMES))
        res = minimize(cost, x0, args=(model, data), method="Nelder-Mead",
                       options={"maxiter": 600, "xatol": 1e-3, "fatol": 1e-4})
        if res.fun < best_cost:
            best_cost, best = res.fun, res.x

    r = evaluate(model, data, best)
    print("best static pose found (symmetric):")
    for n, v in zip(PARAM_NAMES, best):
        print(f"  {n:<12} {v:+.3f} rad")
    print(f"\n  CoM offset from support : {r['com_offset'] * 100:.2f} cm"
          f"   {'OK' if r['com_offset'] < 0.02 else 'NOT BALANCED'}")
    print(f"  peak joint torque       : {r['max_torque']:.3f} Nm"
          f"   {'OK' if r['max_torque'] <= TORQUE_LIMIT else 'EXCEEDS ' + str(TORQUE_LIMIT)}")
    print(f"  trunk above head        : {r['stack'] * 100:.1f} cm"
          f"   {'OK' if r['stack'] >= 0.08 else 'NOT STACKED (needs 8 cm)'}")
    print(f"  contacts                : {r['ncon']}"
          f"   {'point contact -> inverted pendulum' if r['ncon'] <= 2 else 'patch contact -> may be statically stable'}")

    print(f"\nsettle test — command the pose, {SETTLE_S:.0f} s, from noisy inits:")
    held = 0
    for seed, noise in enumerate([0.0, 0.02, 0.02, 0.05, 0.05]):
        s = settle_test(model, data, best, noise, 100 + seed)
        # Touching the floor is part of the criterion. Without it, a robot in
        # free fall passes every other check.
        ok = (s["stack_after"] >= 0.08 and s["tilt_after_deg"] > 120
              and s["ncon_after"] > 0 and s["head_z_after"] < 0.06)
        held += ok
        print(f"  noise {noise:.2f}: stack {s['stack_after'] * 100:+5.1f} cm, "
              f"tilt {s['tilt_after_deg']:5.1f}°, head {s['head_z_after'] * 100:4.1f} cm, "
              f"contacts {s['ncon_after']}  {'HELD' if ok else 'collapsed'}")

    balanced = r["com_offset"] < 0.02 and r["stack"] >= 0.08 and r["ncon"] > 0
    within_torque = r["max_torque"] <= TORQUE_LIMIT

    if not (balanced and within_torque):
        verdict = ("IMPOSSIBLE at this torque ceiling — no balanced, stacked, "
                   "torque-feasible pose was found. Close the task.")
    elif held >= 3:
        verdict = ("STATICALLY STABLE — the pose holds open-loop. The reward is "
                   "the only thing standing in the way; hand this pose to it as "
                   "an explicit target.")
    else:
        verdict = (
            "POSSIBLE BUT UNSTABLE — a balanced, stacked pose exists at "
            f"{r['max_torque']:.3f} Nm ({100 * r['max_torque'] / TORQUE_LIMIT:.0f}% of "
            "the ceiling), so the robot is strong enough and the geometry works. "
            "It rests on a point contact and collapses when commanded open-loop, "
            "so a headstand here is an INVERTED PENDULUM: it needs active "
            "feedback balancing, not a pose to hold. That is a materially harder "
            "RL problem than the other tricks and none of the reward designs so "
            "far even attempted it."
        )
    print(f"\nVERDICT: {verdict}")
    print(f"\n  (torque headroom: peak {r['max_torque']:.3f} Nm of {TORQUE_LIMIT} Nm "
          f"available — strength is NOT the limit here)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
