"""Headless CPU-MuJoCo evaluation for the sprint and jump policies.

`play` needs a GPU and a viewer; this needs neither, so a policy can be scored
on a laptop straight after `modal volume get`. It drives the ONNX through
`infer_policy.PolicyInference` — the same observation builder the real runtime
uses — rather than rebuilding the 61D layout, so a mismatch here would be a
mismatch on the robot too.

    uv run scripts/eval_sprint_jump.py sprint --policy sprint.onnx
    uv run scripts/eval_sprint_jump.py jump   --policy jump.onnx

Sprint sweeps commanded forward speed and reports what the robot ACHIEVES at
each command, which is the only honest way to answer "how fast": the command is
what you asked for, the measured displacement is what you got, and the gap
between them is where the ceiling is.

Jump reports peak trunk height above its own resting stance and total flight
time, flight being no-foot-contact.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from infer_policy import MICRODUCK_XML, PolicyInference  # noqa: E402

CONTROL_DECIMATION = 4
TIMESTEP = 0.005
DEFAULT_CURRENT_LIMIT = 1.75  # XL330 firmware saturation, as infer_policy applies

RENDER_FPS = 50
RENDER_EVERY = int(1.0 / (RENDER_FPS * TIMESTEP))  # physics steps per frame
RENDER_SIZE = (480, 640)  # h, w


class Recorder:
    """Optional mp4 of the rollout, from a camera that tracks the trunk.

    The scene's only camera is the robot's own POV, which is useless for judging
    a gait or a jump, so this drives a free camera locked to the trunk instead.
    A no-op when path is None, so the measurement path is identical whether or
    not anything is being recorded.
    """

    def __init__(self, model, path: str | None, trunk_id: int):
        self.path = path
        if path is None:
            return
        import imageio.v2 as iio

        self.renderer = mujoco.Renderer(model, height=RENDER_SIZE[0], width=RENDER_SIZE[1])
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = trunk_id
        self.cam.distance = 0.9
        self.cam.azimuth = 130.0
        self.cam.elevation = -12.0
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = iio.get_writer(path, fps=RENDER_FPS, macro_block_size=1)
        self.frames = 0

    def maybe_capture(self, data, step: int) -> None:
        if self.path is None or step % RENDER_EVERY:
            return
        self.renderer.update_scene(data, camera=self.cam)
        self.writer.append_data(self.renderer.render())
        self.frames += 1

    def close(self) -> None:
        if self.path is None:
            return
        self.writer.close()
        print(f"\nrecording: {self.path} ({self.frames} frames, "
              f"{self.frames / RENDER_FPS:.1f}s)")


def _build(policy_path: str, current_limit: float = DEFAULT_CURRENT_LIMIT):
    model = mujoco.MjModel.from_xml_path(MICRODUCK_XML)
    model.opt.timestep = TIMESTEP
    if current_limit > 0:
        from bam.model import load_model

        kt = load_model(motor_name="xl330", model="m6").kt.value
        model.actuator_forcerange[:, 0] = -kt * current_limit
        model.actuator_forcerange[:, 1] = kt * current_limit
        model.actuator_forcelimited[:] = 1
    data = mujoco.MjData(model)
    policy = PolicyInference(
        model, data,
        walking_onnx_path=policy_path,
        use_projected_gravity=True,
        new_cmd_obs=True,
    )
    return model, data, policy


def _reset_to_stance(model, data, policy) -> None:
    """Place the robot standing, exactly as infer_policy.py does.

    `mj_resetData` alone drops it at qpos0, which is not a stance — the vendored
    walking policy scored 0.002 m/s that way, which is a broken harness rather
    than a broken policy.
    """
    mujoco.mj_resetData(model, data)
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = model.jnt_qposadr[fj]
    data.qpos[adr + 0] = 0.0
    data.qpos[adr + 1] = 0.0
    data.qpos[adr + 2] = 0.125
    data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    data.ctrl[:] = policy.default_pose
    policy.last_action[:] = 0.0
    mujoco.mj_forward(model, data)


def _trunk_body_id(model) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")


def _trunk_tilt_deg(model, data) -> float:
    """Angle between the trunk's up-axis and world up."""
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = model.jnt_qposadr[fj]
    q = data.qpos[adr + 3:adr + 7]
    cos_tilt = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
    return float(np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0))))


def _yaw_angle(model, data) -> float:
    """Heading, from the trunk quaternion. Integrating the DIFFERENCE of this is
    the only honest way to count revolutions."""
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr = model.jnt_qposadr[fj]
    w, x, y, z = data.qpos[adr + 3:adr + 7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _yaw_rate(model, data) -> float:
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr_v = model.jnt_dofadr[fj]
    return float(data.qvel[adr_v + 5])


def _body_forward_speed(model, data) -> float:
    """Trunk forward speed in the BODY frame.

    The twist command is body-frame vx, so this is the like-for-like
    comparison. World-x displacement is not: a policy that veers covers real
    ground while its x-displacement understates it, which made the vendored
    walker look like it moved at 0.4x its command.
    """
    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    adr_q = model.jnt_qposadr[fj]
    adr_v = model.jnt_dofadr[fj]
    v_world = data.qvel[adr_v:adr_v + 3].copy()
    quat = data.qpos[adr_q + 3:adr_q + 7].copy()
    v_body = np.zeros(3)
    mujoco.mju_rotVecQuat(v_body, v_world, np.array([quat[0], -quat[1], -quat[2], -quat[3]]))
    return float(v_body[0])


def _feet_in_contact(model, data) -> bool:
    """True if any geom whose name mentions a foot is touching anything."""
    for i in range(data.ncon):
        c = data.contact[i]
        for g in (c.geom1, c.geom2):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if "foot" in name or "ankle" in name or "sole" in name:
                return True
    return False


# A foot resting with almost no load is not bearing weight. The training
# contact sensor reduces by netforce, so it calls such a foot "off" while raw
# geometric contact calls it "down" — which made a weight-shift read as a
# one-leg stand in training and a two-foot stand here. Ignore contacts under a
# few percent of body weight (~0.8 kg) so both agree on what "standing on it"
# means.
FOOT_LOAD_THRESHOLD_N = 0.4


def _foot_contacts(model, data, min_force: float = FOOT_LOAD_THRESHOLD_N):
    """(left_down, right_down), by LOAD not mere touch."""
    load = {"left": 0.0, "right": 0.0}
    wrench = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                 for g in (c.geom1, c.geom2)]
        side = ("left" if any("left_foot" in n for n in names)
                else "right" if any("right_foot" in n for n in names) else None)
        if side is None:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        load[side] += abs(float(wrench[0]))  # normal component
    return load["left"] >= min_force, load["right"] >= min_force


def _site_z(model, data, name: str) -> float:
    return float(data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)][2])


def run_headstand(policy_path: str, seconds: float, record: str | None = None) -> int:
    """Score a headstand as feet-above-head, the shape only a headstand has.

    Standing measures about -0.21 (mouth 0.233 up, feet on the floor); a clean
    headstand is positive. Trunk tilt alone would call a robot lying on its back
    'inverted', which is why inversion is measured from geometry instead.
    """
    model, data, policy = _build(policy_path)
    trunk = _trunk_body_id(model)
    rec = Recorder(model, record, trunk)
    _reset_to_stance(model, data, policy)
    policy.vel_cmd[:] = 0.0

    inversions, head_zs = [], []
    hold = best_hold = 0.0
    for k in range(int(seconds / TIMESTEP)):
        if k % CONTROL_DECIMATION == 0:
            policy._update_command()
            policy.apply_action(policy.infer())
        mujoco.mj_step(model, data)
        rec.maybe_capture(data, k)
        head_z = _site_z(model, data, "mouth_tip")
        feet_z = (_site_z(model, data, "left_foot") + _site_z(model, data, "right_foot")) / 2
        inv = feet_z - head_z
        inversions.append(inv)
        head_zs.append(head_z)
        if inv > 0.05 and head_z < 0.06:
            hold += TIMESTEP
            best_hold = max(best_hold, hold)
        else:
            hold = 0.0

    inv = np.array(inversions)
    print(f"inversion (feet - head), standing is about -0.21 m")
    print(f"  final                : {inv[-1]:+.3f} m")
    print(f"  best                 : {inv.max():+.3f} m")
    print(f"  mean over last 3 s   : {inv[-int(3 / TIMESTEP):].mean():+.3f} m")
    print(f"head height final      : {head_zs[-1] * 100:.1f} cm")
    print(f"time inverted (>0.05)  : {(inv > 0.05).mean() * 100:.0f}% of rollout")
    print(f"longest headstand hold : {best_hold:.2f}s")
    print(f"VERDICT                : "
          f"{'HEADSTAND held' if best_hold >= 1.0 else 'passes through inversion' if inv.max() > 0.05 else 'never inverts'}")
    rec.close()
    return 0


def run_trick(policy_path: str, kind: str, yaw_cmd: float, seconds: float,
              record: str | None = None) -> int:
    """Score a balance or spin trick from a real rollout."""
    model, data, policy = _build(policy_path)
    trunk = _trunk_body_id(model)
    rec = Recorder(model, record, trunk)
    _reset_to_stance(model, data, policy)
    _settle(model, data, policy, 0.8)

    policy.vel_cmd[:] = np.array([0.0, 0.0, yaw_cmd], dtype=np.float32)
    single = both = airborne = 0
    run_len = best_run = 0
    # NET rotation, from the integrated yaw ANGLE. Summing |yaw rate| counts a
    # back-and-forth shake as progress: the run-1 spin scored 12.5 "revolutions"
    # that way while its actual heading moved 0.04 of one.
    net_yaw = 0.0
    prev_yaw = _yaw_angle(model, data)
    yaws, tilts = [], []
    steps = int(seconds / TIMESTEP)
    for k in range(steps):
        if k % CONTROL_DECIMATION == 0:
            policy._update_command()
            policy.apply_action(policy.infer())
        mujoco.mj_step(model, data)
        rec.maybe_capture(data, k)
        l, r = _foot_contacts(model, data)
        if l ^ r:
            single += 1
            run_len += 1
            best_run = max(best_run, run_len)
        else:
            run_len = 0
            both += 1 if (l and r) else 0
            airborne += 1 if not (l or r) else 0
        wz = _yaw_rate(model, data)
        yaws.append(wz)
        cur_yaw = _yaw_angle(model, data)
        d_yaw = (cur_yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
        net_yaw += d_yaw
        prev_yaw = cur_yaw
        tilts.append(_trunk_tilt_deg(model, data))
        if data.xpos[trunk][2] < 0.05:
            print(f"FELL at t={k * TIMESTEP:.1f}s")
            break

    n = max(len(yaws), 1)
    print(f"duration held          : {n * TIMESTEP:.1f}s of {seconds:.0f}s")
    print(f"single-support         : {single / n * 100:.0f}% of steps")
    print(f"  longest unbroken     : {best_run * TIMESTEP:.2f}s")
    print(f"double-support         : {both / n * 100:.0f}%")
    print(f"airborne               : {airborne / n * 100:.0f}%")
    print(f"mean |yaw rate|        : {float(np.mean(np.abs(yaws))):.2f} rad/s "
          f"(magnitude only — oscillation inflates this)")
    print(f"mean SIGNED yaw rate   : {float(np.mean(yaws)):+.2f} rad/s")
    print(f"NET rotation           : {net_yaw / (2 * math.pi):+.2f} revolutions "
          f"{'(SPINNING)' if abs(net_yaw) > 2 * math.pi else '(NOT actually rotating)'}")
    print(f"mean / max trunk tilt  : {float(np.mean(tilts)):.0f}° / {max(tilts):.0f}°")
    print(f"upright at end         : "
          f"{'yes' if _trunk_tilt_deg(model, data) < 30 else 'NO — ended fallen'}")
    rec.close()
    return 0


def _settle(model, data, policy, seconds: float = 1.5) -> None:
    """Let the policy stand before measuring, so t=0 is a real stance."""
    policy.vel_cmd[:] = 0.0
    for k in range(int(seconds / TIMESTEP)):
        if k % CONTROL_DECIMATION == 0:
            policy._update_command()
            policy.apply_action(policy.infer())
        mujoco.mj_step(model, data)


def run_sprint(policy_path: str, commands, seconds: float, record: str | None = None) -> int:
    model, data, policy = _build(policy_path)
    trunk = _trunk_body_id(model)
    rec = Recorder(model, record, trunk)
    frame = 0
    print(f"{'cmd vx':>7} {'body vx':>9} {'ratio':>6} {'path m/s':>9} {'fell':>5}")
    print("-" * 42)
    best = 0.0
    for cmd in commands:
        _reset_to_stance(model, data, policy)
        _settle(model, data, policy)

        policy.vel_cmd[:] = np.array([cmd, 0.0, 0.0], dtype=np.float32)
        speeds, path, prev, fell = [], 0.0, data.xpos[trunk].copy(), False
        for k in range(int(seconds / TIMESTEP)):
            if k % CONTROL_DECIMATION == 0:
                policy._update_command()
                policy.apply_action(policy.infer())
            mujoco.mj_step(model, data)
            rec.maybe_capture(data, frame)
            frame += 1
            speeds.append(_body_forward_speed(model, data))
            here = data.xpos[trunk].copy()
            path += float(np.linalg.norm(here[:2] - prev[:2]))
            prev = here
            if here[2] < 0.05:
                fell = True
                break
        elapsed = len(speeds) * TIMESTEP
        # Trim the first 1 s: the policy is still accelerating out of stance.
        steady = speeds[int(1.0 / TIMESTEP):] or speeds
        achieved = float(np.mean(steady))
        ratio = achieved / cmd if cmd else float("nan")
        print(f"{cmd:>7.2f} {achieved:>9.3f} {ratio:>6.2f} {path / elapsed:>9.3f} "
              f"{'YES' if fell else '':>5}")
        if not fell:
            best = max(best, achieved)
    print(f"\nBest sustained body-frame forward speed: {best:.3f} m/s")
    rec.close()
    return 0


def run_turn(policy_path: str, vx: float, yaw_cmds, seconds: float,
             record: str | None = None) -> int:
    """Hold a forward command while sweeping yaw: can it turn AND keep speed?

    A top speed that only exists in a straight line is a much weaker result than
    it looks, and the straight-line sweep cannot tell the difference. This is
    the same measurement with the yaw command switched on.
    """
    model, data, policy = _build(policy_path)
    trunk = _trunk_body_id(model)
    rec = Recorder(model, record, trunk)
    frame = 0
    print(f"holding cmd vx={vx:.2f} while sweeping yaw\n")
    print(f"{'cmd yaw':>8} {'body vx':>9} {'yaw rate':>9} {'ratio':>6} {'max tilt':>9} {'fell':>5}")
    print("-" * 52)
    for yaw in yaw_cmds:
        _reset_to_stance(model, data, policy)
        _settle(model, data, policy)
        policy.vel_cmd[:] = np.array([vx, 0.0, yaw], dtype=np.float32)
        speeds, yaws, tilt, fell = [], [], 0.0, False
        for k in range(int(seconds / TIMESTEP)):
            if k % CONTROL_DECIMATION == 0:
                policy._update_command()
                policy.apply_action(policy.infer())
            mujoco.mj_step(model, data)
            rec.maybe_capture(data, frame); frame += 1
            speeds.append(_body_forward_speed(model, data))
            yaws.append(_yaw_rate(model, data))
            tilt = max(tilt, _trunk_tilt_deg(model, data))
            if data.xpos[trunk][2] < 0.05:
                fell = True
                break
        steady = slice(int(1.0 / TIMESTEP), None)
        vx_m = float(np.mean(speeds[steady] or speeds))
        yaw_m = float(np.mean(yaws[steady] or yaws))
        ratio = yaw_m / yaw if yaw else float("nan")
        print(f"{yaw:>8.2f} {vx_m:>9.3f} {yaw_m:>9.3f} {ratio:>6.2f} {tilt:>8.1f}° "
              f"{'YES' if fell else '':>5}")
    rec.close()
    return 0


def run_jump(policy_path: str, seconds: float, record: str | None = None) -> int:
    model, data, policy = _build(policy_path)
    trunk = _trunk_body_id(model)
    rec = Recorder(model, record, trunk)
    _reset_to_stance(model, data, policy)
    # NO settle here, unlike sprint. An episode-max reward pays only for beating
    # the record, so the optimal policy jumps ONCE and then stands — settling
    # first skips the only jump and measures the stance afterwards, which
    # reported 0.0 cm for a policy that had just cleared 1.1 cm.
    stance_z = float(data.xpos[trunk][2])
    late_peak = stance_z  # peak after the opening jump, to tell one-shot apart

    peak, airborne_steps, flights, in_flight, this_flight = stance_z, 0, [], False, 0
    max_tilt = 0.0
    policy.vel_cmd[:] = 0.0
    for k in range(int(seconds / TIMESTEP)):
        if k % CONTROL_DECIMATION == 0:
            policy._update_command()
            policy.apply_action(policy.infer())
        mujoco.mj_step(model, data)
        rec.maybe_capture(data, k)
        z = float(data.xpos[trunk][2])
        peak = max(peak, z)
        if k * TIMESTEP > 3.0:
            late_peak = max(late_peak, z)
        max_tilt = max(max_tilt, _trunk_tilt_deg(model, data))
        if not _feet_in_contact(model, data):
            airborne_steps += 1
            this_flight += 1
            in_flight = True
        elif in_flight:
            flights.append(this_flight * TIMESTEP)
            this_flight, in_flight = 0, False

    print(f"resting stance trunk z : {stance_z * 100:.1f} cm")
    print(f"peak trunk z           : {peak * 100:.1f} cm")
    print(f"PEAK HEIGHT GAINED     : {(peak - stance_z) * 100:.1f} cm")
    # A policy that only jumps out of the reset transient is not deployable:
    # on the robot you press a button and expect a jump, not a reboot.
    print(f"  best jump after t=3s : {(late_peak - stance_z) * 100:.1f} cm "
          f"({'repeats' if late_peak - stance_z > 0.005 else 'ONE-SHOT — only jumps at reset'})")
    print(f"airborne fraction      : {airborne_steps / (seconds / TIMESTEP) * 100:.1f}%")
    print(f"distinct flights       : {len(flights)}")
    # A high number that ends on its face is not a jump. Tilt and end-state say
    # whether the height was earned or was the first half of a fall.
    print(f"max trunk tilt         : {max_tilt:.0f}°")
    print(f"upright at end         : "
          f"{'yes' if _trunk_tilt_deg(model, data) < 30 else 'NO — ended fallen'}")
    if flights:
        print(f"longest flight         : {max(flights) * 1000:.0f} ms")
        print(f"median flight          : {statistics.median(flights) * 1000:.0f} ms")
    else:
        print("longest flight         : none — never left the ground")
    rec.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["sprint", "jump", "turn", "trick", "headstand"])
    ap.add_argument("--policy", required=True, help="ONNX from scripts/export.py")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--commands", default="0.2,0.4,0.6,0.8,1.0,1.2",
                    help="sprint only: forward speeds to sweep")
    ap.add_argument("--record", default=None, help="write an mp4 of the rollout here")
    ap.add_argument("--vx", type=float, default=0.6, help="turn mode: forward command held")
    ap.add_argument("--yaws", default="-1.5,-0.75,0.0,0.75,1.5",
                    help="turn mode: yaw commands to sweep")
    ap.add_argument("--kind", default="balance", help="trick mode: label only")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="trick mode: yaw command (0 for a balance trick)")
    a = ap.parse_args()

    if a.mode == "sprint":
        return run_sprint(a.policy, [float(c) for c in a.commands.split(",")],
                          a.seconds, a.record)
    if a.mode == "headstand":
        return run_headstand(a.policy, a.seconds, a.record)
    if a.mode == "trick":
        return run_trick(a.policy, a.kind, a.yaw, a.seconds, a.record)
    if a.mode == "turn":
        return run_turn(a.policy, a.vx, [float(y) for y in a.yaws.split(",")],
                        a.seconds, a.record)
    return run_jump(a.policy, a.seconds, a.record)


if __name__ == "__main__":
    raise SystemExit(main())
