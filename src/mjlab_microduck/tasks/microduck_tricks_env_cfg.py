"""Microduck TRICKS — one-leg balance and spins.

Three tricks from one factory, all built on the velocity recipe so the robot,
61D obs contract, DR and NaN guards come along unchanged and any of them can be
hot-swapped into the runtime:

- `one_leg_stand`  hold a stance on exactly one foot
- `spin_two_leg`   spin in place on both feet, tracking a fast yaw command
- `spin_one_leg`   spin in place on one foot — the hard one

Two design points carry most of the weight.

**Holding one foot, not merely having one foot down.** Requiring "exactly one
foot" seemed to rule out the alternating cheat. It does not — swapping feet
rapidly satisfies exactly-one at EVERY instant. Run 1 scored 88% single support
with a longest unbroken hold of 0.21 s and 2.6 revolutions of unasked-for spin:
a hopping pirouette wearing a balance task's numbers. So the one-leg gate is
`sustained_single_support`, which tracks WHICH foot bears weight and resets its
clock on any switch, so only genuinely holding accumulates.

The trunk-height floor stays for a different cheat: a robot folded onto one knee
also has exactly one foot down, and without a floor that collapse is cheaper
than balancing.

**Spins are rewarded on a CAPPED yaw rate.** Uncapped |omega_z| is a jackpot —
the fastest way to spin a 25 cm biped is to fall over, and AGENTS.md notes this
robot tumbles at 3.5-5.5 rad/s naturally, so an uncapped term pays most for
exactly the failure mode. Capping makes "spin and stay up" the argmax.

The gait recipe comes out for all three: `air_time` pays for alternating
single-support, which is a walk and would be earned by stepping in place rather
than by balancing or spinning.
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

FEET_CONTACT_SENSOR = "feet_ground_contact"

# Gait shaping that would be earned by stepping in place instead of by the trick.
_WALKING_GAIT_TERMS = ("air_time", "foot_clearance", "foot_swing_height")

# Stand tall enough that a collapse onto one knee cannot satisfy a support gate.
# Stance trunk z is ~0.119 measured; 0.10 leaves room for a real crouch while
# excluding a fold.
SUPPORT_MIN_HEIGHT = 0.10

# Hold this long on one foot to earn full credit. Run 1's best unbroken hold was
# 0.21 s, so 2 s is a real target rather than a formality.
ONE_LEG_HOLD_TARGET_S = 2.0

# Run 1 never produced a hold: 0/16 envs reached 2 s under BAM in the training
# env, and 0% single-support when exported. The reward was not wrong, it was
# OUTBID. At weight 6 and a measured hold value near 0.2, holding paid ~1.2
# while a plain two-foot stance collected upright (2.0) plus both tracking terms
# (2.0) for free. Standing still was the better deal, and the policy took it.
# At 20 a sustained hold is worth more than everything a stance earns.
ONE_LEG_SUPPORT_WEIGHT = 20.0

# Brisk but holdable. Above ~5 rad/s this robot is tumbling, not spinning.
SPIN_TARGET_RATE = 4.0

# The walking recipe records that a ramp to ang +/-2.0 "outpaced the robot's
# capability", so asking for +/-4.0 from step 0 would repeat exactly the mistake
# the sprint task was written to avoid. Ramp it, and let the run show where the
# spin rate actually tops out.
SPIN_RATE_STAGES = [
    {"step": 0,         "ang_vel_z": (-1.5, 1.5)},
    {"step": 400 * 24,  "ang_vel_z": (-2.5, 2.5)},
    {"step": 900 * 24,  "ang_vel_z": (-3.2, 3.2)},
    {"step": 1500 * 24, "ang_vel_z": (-4.0, 4.0)},
]

# Near-zero twist for the balance trick, never exactly zero: a command slot that
# is never non-zero has dead weights forever, and these share the 61D contract.
STILL_TWIST = ((-0.05, 0.05), (-0.05, 0.05), (-0.1, 0.1))

TRICKS = ("one_leg_stand", "spin_two_leg", "spin_one_leg")


def make_microduck_trick_env_cfg(trick: str, play: bool = False) -> ManagerBasedRlEnvCfg:
    if trick not in TRICKS:
        raise ValueError(f"unknown trick {trick!r}; known: {', '.join(TRICKS)}")

    cfg = make_microduck_velocity_env_cfg(play=play)
    trunk = SceneEntityCfg("robot", body_names=("trunk_base",))
    spinning = trick.startswith("spin")
    one_leg = trick in ("one_leg_stand", "spin_one_leg")

    for name in _WALKING_GAIT_TERMS:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # A trick is a deviation from the home stance, so the pose term must not
    # pin it there — same reason as the jump task.
    cfg.rewards["pose"].weight = 0.1
    # Motion-blockers stay low: these are dynamic, and for the spins the
    # angular terms penalise the trick itself.
    cfg.rewards["body_ang_vel"].weight = -0.005 if spinning else -0.02
    cfg.rewards["angular_momentum"].weight = -0.002 if spinning else -0.01
    cfg.rewards["action_rate_l2"].weight = -0.005  # ramped after discovery
    cfg.rewards["head_pose_tracking"].weight = 0.3

    command = cfg.commands["twist"]
    command.rel_turn_in_place_envs = 0.0
    command.rel_standing_envs = 0.0
    if spinning:
        # Spin in place: no translation, fast yaw. The tracking reward then
        # does the steering and the capped-rate term supplies the ambition.
        command.ranges.lin_vel_x = (-0.05, 0.05)
        command.ranges.lin_vel_y = (-0.05, 0.05)
        command.ranges.ang_vel_z = SPIN_RATE_STAGES[0]["ang_vel_z"]
        cfg.rewards["track_angular_velocity"].weight = 3.0
        cfg.rewards["track_linear_velocity"].weight = 1.0
    else:
        command.ranges.lin_vel_x, command.ranges.lin_vel_y, command.ranges.ang_vel_z = STILL_TWIST
        cfg.rewards["track_linear_velocity"].weight = 1.0
        cfg.rewards["track_angular_velocity"].weight = 1.0

    support_params = {
        "sensor_name": FEET_CONTACT_SENSOR,
        "asset_cfg": trunk,
        "min_height": SUPPORT_MIN_HEIGHT,
    }
    if one_leg:
        support_params["hold_target_s"] = ONE_LEG_HOLD_TARGET_S
    cfg.rewards["support"] = RewardTermCfg(
        func=microduck_mdp.sustained_single_support if one_leg
        else microduck_mdp.double_support_reward,
        # The one-leg gate is the whole trick and is hard to satisfy, so it
        # carries the stack. The two-leg gate is nearly free while standing, so
        # it is only a guard against the spin becoming a fall.
        weight=ONE_LEG_SUPPORT_WEIGHT if one_leg else 1.0,
        params=support_params,
    )

    if spinning:
        cfg.rewards["spin_rate"] = RewardTermCfg(
            func=microduck_mdp.yaw_rate_capped,
            weight=3.0,
            params={"asset_cfg": trunk, "target_rate": SPIN_TARGET_RATE},
        )
        cfg.rewards["spin_progress"] = RewardTermCfg(
            func=microduck_mdp.spin_progress,
            weight=1.0,
            params={"asset_cfg": trunk},
        )

    if spinning:
        if play:
            command.ranges.ang_vel_z = SPIN_RATE_STAGES[-1]["ang_vel_z"]
        else:
            cfg.curriculum["spin_rate_range"] = CurriculumTermCfg(
                func=microduck_mdp.twist_command_range_curriculum,
                params={"command_name": "twist", "range_stages": SPIN_RATE_STAGES},
            )

    if not play:
        # Smoothness only after the skill exists — an attempt-tax during
        # discovery makes "do nothing" win.
        cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                # Pushed out from 600/1200: run 1's support metric peaked around
                # iteration 660 and had collapsed by 1490, which is exactly where
                # this tax was ramping. An attempt-tax arriving while the skill
                # is still forming makes standing still win.
                "weight_stages": [
                    {"step": 0, "weight": -0.005},
                    {"step": 1500 * 24, "weight": -0.02},
                    {"step": 3000 * 24, "weight": -0.05},
                ],
            },
        )

    return cfg


def _rl_cfg(name: str, iters: int):
    return dataclasses.replace(
        MicroduckRlCfg, experiment_name=name, run_name=name, max_iterations=iters
    )


# Episodic tricks are ~1000 iterations at 4096 envs per the playbook; the
# one-legged spin gets more because it is two skills at once.
MicroduckOneLegStandRlCfg = _rl_cfg("one_leg_stand", 4_000)
MicroduckSpinTwoLegRlCfg = _rl_cfg("spin_two_leg", 4_000)
MicroduckSpinOneLegRlCfg = _rl_cfg("spin_one_leg", 6_000)
