"""Microduck HEADSTAND — invert and hold on the head.

The one trick here that cannot use the walking robot. `robot_walk.xml` has
exactly two colliding geoms, both feet: the trunk and head pass through the
floor, which is fine when falling is cheap and fatal when the whole task is to
balance on your head. This swaps in the all-collisions model the standup family
uses (11 colliding geoms), which is a one-line entity swap rather than a fork.

Measured as a STACK — head on the floor, trunk well above it, feet above that.
Trunk tilt alone is useless (a robot on its back is "inverted" by any tilt
measure), and feet-above-head plus a low head is not enough either: run 1
satisfied exactly that by faceplanting with its legs sprawled up, resting 2.91 N
on its head and 4.29 N across BOTH HIPS with the trunk 3.4 cm off the floor. It
scored a 11.7 s "headstand". The stacking gate is what tells a stand from a
tripod, and it was added only after the video was watched.

Two terms, mirroring what the other tricks needed:

- `headstand_progress` is potential-based on feet-above-head. Standing is about
  -0.19 and a clean headstand about +0.17, so it is a dense gradient across the
  entire maneuver. That matters more here than anywhere else, because a
  headstand has no natural partial credit — every intermediate pose is just
  falling over, and without shaping there is nothing to climb.
- `headstand_hold` pays for SUSTAINED inversion, because the one-leg tricks
  already demonstrated that an instantaneous gate gets satisfied by passing
  through it rather than by holding it.

**This is a BALANCING task, not a pose task**, and getting that wrong cost two
runs. `scripts/headstand_feasibility.py` settled the physics: a balanced,
stacked, torque-feasible pose exists (CoM offset 0.00 cm, 13.7 cm stack, peak
0.136 Nm against a 0.641 Nm ceiling — the robot has 4.7x the strength it needs),
but it rests on ONE contact point and collapses in every open-loop settle trial.
It is an inverted pendulum.

Runs 1 and 2 both paid for BEING in the posture. Nothing paid for the corrective
control that keeps you there, and the posture is not reachable by accident, so
`headstand_hold` went unsatisfied for 2300 straight iterations. Two changes
follow from that:

- `headstand_balance_progress` pays for every metre of lean REMOVED, which is
  dense from the first frame and is what a pendulum controller actually does.
- `reset_to_headstand` spawns half of all episodes already balanced, using the
  solved equilibrium with noise on top. This is AGENTS.md's reverse-curriculum
  prescription for "learns the start, never the last mile": the policy currently
  never visits the balance state, so it gets no on-policy data there. Entering a
  headstand from standing is a separate and much harder problem; the remaining
  half of episodes still train it.
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# Gait shaping is meaningless upside down.
_WALKING_GAIT_TERMS = ("air_time", "foot_clearance", "foot_swing_height")

HEAD_SITE = "mouth_tip"
FOOT_SITES = ("left_foot", "right_foot")

# Standing measures about -0.19; a clean headstand about +0.17.
HEADSTAND_MIN_INVERSION = 0.05
HEADSTAND_MAX_HEAD_HEIGHT = 0.06
# Run 1 satisfied "feet above head, head on floor" by faceplanting with its legs
# sprawled up: 2.91 N on the head and 4.29 N across both hips, trunk only 3.4 cm
# off the floor. A real headstand STACKS head -> trunk -> feet, so the trunk has
# to be well clear of the head. 8 cm rejects that tripod and admits a stack.
HEADSTAND_MIN_TRUNK_ABOVE_HEAD = 0.08
HEADSTAND_STACK_WEIGHT = 60.0
HEADSTAND_HOLD_TARGET_S = 2.0
HEADSTAND_PROGRESS_WEIGHT = 60.0
HEADSTAND_HOLD_WEIGHT = 8.0
HEADSTAND_METRIC_WEIGHT = 0.02  # readable in logs; a weight-0 metric logs 0.0000
HEADSTAND_BALANCE_WEIGHT = 120.0  # metres of lean removed; the dense signal
HEADSTAND_SPAWN_FRACTION = 0.5


def make_microduck_headstand_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play)

    # The whole reason this task cannot use the walking recipe's robot.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}

    head = SceneEntityCfg("robot", site_names=(HEAD_SITE,))
    feet = SceneEntityCfg("robot", site_names=FOOT_SITES)
    geom = {"head_cfg": head, "feet_cfg": feet}

    for name in _WALKING_GAIT_TERMS:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # Every one of these fights being upside down.
    cfg.rewards["upright"].weight = 0.0     # upright is the opposite of the goal
    cfg.rewards["pose"].weight = 0.05
    cfg.rewards["track_linear_velocity"].weight = 0.2
    cfg.rewards["track_angular_velocity"].weight = 0.2
    cfg.rewards["head_pose_tracking"].weight = 0.1
    cfg.rewards["body_ang_vel"].weight = -0.005
    cfg.rewards["angular_momentum"].weight = -0.002
    cfg.rewards["action_rate_l2"].weight = -0.005

    command = cfg.commands["twist"]
    command.rel_turn_in_place_envs = 0.0
    command.rel_standing_envs = 0.0
    command.ranges.lin_vel_x = (-0.05, 0.05)
    command.ranges.lin_vel_y = (-0.05, 0.05)
    command.ranges.ang_vel_z = (-0.1, 0.1)

    cfg.rewards["headstand_progress"] = RewardTermCfg(
        func=microduck_mdp.headstand_progress,
        weight=HEADSTAND_PROGRESS_WEIGHT,
        params=geom,
    )
    cfg.rewards["headstand_hold"] = RewardTermCfg(
        func=microduck_mdp.headstand_hold,
        weight=HEADSTAND_HOLD_WEIGHT,
        params={
            **geom,
            "min_inversion": HEADSTAND_MIN_INVERSION,
            "max_head_height": HEADSTAND_MAX_HEAD_HEIGHT,
            "min_trunk_above_head": HEADSTAND_MIN_TRUNK_ABOVE_HEAD,
            "hold_target_s": HEADSTAND_HOLD_TARGET_S,
        },
    )
    # Shape the axis the tripod cheated on, so there is a gradient out of it
    # rather than a gate the policy can sit just outside.
    cfg.rewards["headstand_stack"] = RewardTermCfg(
        func=microduck_mdp.headstand_stack_progress,
        weight=HEADSTAND_STACK_WEIGHT,
        params=geom,
    )
    # The term both earlier designs lacked: pay for CORRECTING a lean.
    cfg.rewards["headstand_balance"] = RewardTermCfg(
        func=microduck_mdp.headstand_balance_progress,
        weight=HEADSTAND_BALANCE_WEIGHT,
        params={"head_cfg": head},
    )
    cfg.rewards["balance_err_m"] = RewardTermCfg(
        func=microduck_mdp.headstand_balance_metric,
        weight=HEADSTAND_METRIC_WEIGHT,
        params={"head_cfg": head},
    )
    cfg.rewards["inversion_m"] = RewardTermCfg(
        func=microduck_mdp.headstand_inversion_metric,
        weight=HEADSTAND_METRIC_WEIGHT,
        params=geom,
    )

    # `fell_over` terminates on trunk tilt, which a headstand deliberately
    # violates — leaving it in would end the episode at the moment of success.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]

    if not play:
        # Reverse curriculum: start half the episodes inside the balance state,
        # which the policy otherwise never reaches and so never learns to hold.
        cfg.events["spawn_headstand"] = EventTermCfg(
            func=microduck_mdp.reset_to_headstand,
            mode="reset",
            params={"fraction": HEADSTAND_SPAWN_FRACTION,
                    "joint_noise": 0.03, "tilt_noise_deg": 4.0},
        )
        cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.005},
                    {"step": 1000 * 24, "weight": -0.02},
                ],
            },
        )

    return cfg


MicroduckHeadstandRlCfg = dataclasses.replace(
    MicroduckRlCfg, experiment_name="headstand", run_name="headstand",
    max_iterations=4_000,
)
