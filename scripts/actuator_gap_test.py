"""Does a policy behave the same under BAM as under the deployment actuator?

Training uses the BAM voltage-control model of the XL330. `infer_policy` — the
deployment rehearsal, and every eval built on it — uses MuJoCo position servos
with the current limit applied as a force clip, which its own comment
acknowledges is an approximation.

That gap was the standing explanation for the one-leg policy scoring 95% of its
hold target in training and 0% single support when exported, but it was never
tested. This runs the SAME exported ONNX in the mjlab training env (real BAM
actuators, full DR) and reports the same behavioural metric the deployment eval
reports, so the two numbers are directly comparable.

Three outcomes, three different conclusions:

  * holds under BAM, not under the servo  -> the actuator gap is real, and the
    deployment rehearsal is the thing to fix
  * fails under both                      -> the training REWARD was being
    satisfied by something other than the behaviour, and the policy never
    learned it
  * holds under both                      -> the deployment eval has a bug

    uv run scripts/actuator_gap_test.py --policy p.onnx --task Mjlab-OneLegStand-Flat-MicroDuck
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv

TASK_FACTORY = {
    "Mjlab-OneLegStand-Flat-MicroDuck": ("microduck_tricks_env_cfg",
                                         "make_microduck_trick_env_cfg", "one_leg_stand"),
    "Mjlab-SpinOneLeg-Flat-MicroDuck": ("microduck_tricks_env_cfg",
                                        "make_microduck_trick_env_cfg", "spin_one_leg"),
    "Mjlab-SpinTwoLeg-Flat-MicroDuck": ("microduck_tricks_env_cfg",
                                        "make_microduck_trick_env_cfg", "spin_two_leg"),
    "Mjlab-Jump-Flat-MicroDuck": ("microduck_jump_env_cfg",
                                  "make_microduck_jump_env_cfg", None),
    "Mjlab-Sprint-Flat-MicroDuck": ("microduck_sprint_env_cfg",
                                    "make_microduck_sprint_env_cfg", None),
}


def build_env(task: str, num_envs: int):
    import importlib

    mod_name, fn_name, arg = TASK_FACTORY[task]
    mod = importlib.import_module(f"mjlab_microduck.tasks.{mod_name}")
    fn = getattr(mod, fn_name)
    cfg = fn(arg) if arg else fn()
    cfg.scene.num_envs = num_envs
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--task", required=True, choices=list(TASK_FACTORY))
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--force-cmd", default=None,
                    help="vx,vy,wz held constant. REQUIRED for a fair sprint/spin "
                         "comparison: the training env samples commands at random, "
                         "so an unforced mean averages toward zero no matter how "
                         "good the policy is.")
    a = ap.parse_args()
    forced = ([float(x) for x in a.force_cmd.split(",")] if a.force_cmd else None)

    env = build_env(a.task, a.envs)
    sess = ort.InferenceSession(a.policy, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    obs_dict, _ = env.reset()
    obs = obs_dict["actor"] if isinstance(obs_dict, dict) else obs_dict
    print(f"env actor obs: {tuple(obs.shape)}  policy expects "
          f"{sess.get_inputs()[0].shape}")

    from mjlab_microduck.tasks import mdp as microduck_mdp

    steps = int(a.seconds / env.step_dt)
    single = torch.zeros(a.envs)
    hold = torch.zeros(a.envs)
    best = torch.zeros(a.envs)
    yaws, vxs, tilts = [], [], []
    robot = env.scene["robot"]
    for _ in range(steps):
        if forced is not None:
            cmd = env.command_manager.get_command("twist")
            cmd[:, 0], cmd[:, 1], cmd[:, 2] = forced[0], forced[1], forced[2]
            obs_dict = env.observation_manager.compute()
            obs = obs_dict["actor"] if isinstance(obs_dict, dict) else obs_dict
        acts = np.concatenate([
            sess.run(None, {iname: obs[i:i + 1].cpu().numpy().astype(np.float32)})[0]
            for i in range(obs.shape[0])
        ])
        obs_dict, *_ = env.step(torch.as_tensor(acts, dtype=torch.float32))
        obs = obs_dict["actor"] if isinstance(obs_dict, dict) else obs_dict
        # Behaviour-appropriate telemetry, so sprint and spin can be compared
        # across actuator models too — not just the one-leg hold.
        yaws.append(robot.data.root_link_ang_vel_w[:, 2].abs().mean().item())
        q = robot.data.root_link_quat_w
        cos_tilt = (1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1, 1)
        tilts.append(torch.rad2deg(torch.acos(cos_tilt)).mean().item())
        vb = robot.data.root_link_lin_vel_b if hasattr(robot.data, "root_link_lin_vel_b") \
            else robot.data.root_link_lin_vel_w
        vxs.append(vb[:, 0].mean().item())
        l, r = microduck_mdp._feet_contact_pair(env, "feet_ground_contact")
        one = (l ^ r).float()
        single += one
        hold = torch.where(one.bool(), hold + env.step_dt, torch.zeros_like(hold))
        best = torch.maximum(best, hold)

    frac = (single / steps)
    print(f"\nIN THE TRAINING ENV (BAM actuators, full DR), {a.envs} envs x {a.seconds:.0f}s:")
    print(f"  single-support fraction : mean {frac.mean():.2f}  "
          f"min {frac.min():.2f}  max {frac.max():.2f}")
    print(f"  longest unbroken hold   : mean {best.mean():.2f}s  "
          f"min {best.min():.2f}s  max {best.max():.2f}s")
    print(f"  envs holding >= 2.0s    : {int((best >= 2.0).sum())}/{a.envs}")
    import numpy as _np
    print(f"  mean |yaw rate|         : {_np.mean(yaws):.2f} rad/s")
    print(f"  mean forward vx (body)  : {_np.mean(vxs):.3f} m/s")
    print(f"  mean trunk tilt         : {_np.mean(tilts):.1f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
