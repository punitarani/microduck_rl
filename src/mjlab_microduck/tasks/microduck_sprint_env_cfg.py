"""Microduck SPRINT — straight-line top speed.

The walking recipe fixes `lin_vel_x` at ±0.4 on purpose: its comment records
that a ramp to wider ranges "outpaced the robot's capability" and tracked a
reward decline after iter 1000. So 0.4 m/s is roughly where walking was known
to work, NOT a measured ceiling — and finding that ceiling is this task's whole
objective.

Approach: reuse the velocity env wholesale (robot, 61D obs, DR, noise, delays,
rewards, sim2real stack) and change only the forward command — a staged ramp.
The lateral and turn envelope keeps the walking recipe's coverage on purpose:
narrowing it to buy more forward experience produced a policy that ran in a
circle (see the SPRINT_ANG_VEL_Z comment), and a top speed you can only hold in
a straight line is a much weaker result than it looks.

Why a ramp rather than just commanding 1.0 m/s: an unreachable command is not a
harder task, it is a different and worse one. The tracking reward saturates at
"as close as physics allows", the gradient toward *faster* vanishes, and the
policy learns to lean at a wall it cannot pass. Ramping keeps the command near
the frontier where the gradient is informative.

Reading the run: `Curriculum/sprint_speed` is the current max forward command.
Watch measured speed per checkpoint, not reward — run 1's reward stayed within
7% while its actual top speed fell by 28%, so reward alone would not have shown
the ceiling. The checkpoint before the decline is the one to keep.

Deploys exactly like the walking policy: same 61D obs, same 14 actions, so the
runtime can hot-swap it into the walk slot.
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# Straight-line focus, but NOT at the cost of being able to steer.
#
# Run 1 used ang +/-0.8 with a 0.05 turn-in-place bucket, reasoning that a
# sprint task should spend its experience going forward. Measured result: the
# policy could not turn. At iteration 750 it drifted LEFT at 0.275 rad/s under a
# zero yaw command and tracked +0.75 backwards; by iteration 1500 the bias had
# flipped to +0.338 rad/s right. A bias that changes sign between checkpoints is
# not a structural artefact, it is a behaviour with too little data to converge
# — commanded to run straight, it ran in a circle.
#
# The walking recipe's own values are restored: ang +/-1.0 is the number its
# comment calls "the big change — it makes turning learnable", and the
# turn-in-place bucket goes back to the velocity recipe's fraction, because the
# playbook is explicit that rare command regions need an explicit bucket or they
# never train.
SPRINT_LIN_VEL_Y = (-0.25, 0.25)
SPRINT_ANG_VEL_Z = (-1.0, 1.0)
SPRINT_TURN_IN_PLACE_FRACTION = 0.15

# Forward-speed ramp, TRUNCATED at the measured ceiling.
#
# Run 1 ramped to 1.05 and measured what happens past the robot's limit:
#
#   iter  750 -> 0.617 m/s
#   iter 1500 -> 0.758 m/s   <- peak
#   iter 2000 -> 0.679 m/s
#   iter 2750 -> 0.549 m/s
#
# Speed peaked at 1500 and then FELL as the ramp climbed past what the robot can
# do, with mean reward falling alongside it (122.9 -> 116.7 -> 114.5). This is
# the decline the walking recipe's own comment recorded when its ramp "outpaced
# the robot's capability" — the same wall, found again from the other side.
#
# So the ramp now stops at 0.75, just under the measured 0.758 ceiling, and the
# stages are compressed: the first run covered 0.45 -> its peak inside 1500
# iterations, so spending 3000 getting there was slack, not caution.
SPRINT_SPEED_STAGES = [
    {"step": 0,         "lin_vel_x": (-0.30, 0.45)},
    {"step": 400 * 24,  "lin_vel_x": (-0.30, 0.55)},
    {"step": 900 * 24,  "lin_vel_x": (-0.25, 0.65)},
    {"step": 1500 * 24, "lin_vel_x": (-0.25, 0.75)},
]


def make_microduck_sprint_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Velocity env with a forward-speed curriculum instead of fixed ranges."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    command = cfg.commands["twist"]
    # Start at stage 0 rather than the walking recipe's ±0.4, so the curriculum
    # owns this number from step 0 and the config cannot disagree with itself.
    command.ranges.lin_vel_x = SPRINT_SPEED_STAGES[0]["lin_vel_x"]
    command.ranges.lin_vel_y = SPRINT_LIN_VEL_Y
    command.ranges.ang_vel_z = SPRINT_ANG_VEL_Z
    command.rel_turn_in_place_envs = SPRINT_TURN_IN_PLACE_FRACTION

    # In play mode the curriculum never advances (no training steps), so pin the
    # command to the top stage — otherwise `play` shows a 0.45 m/s walk and the
    # sprint is invisible.
    if play:
        command.ranges.lin_vel_x = SPRINT_SPEED_STAGES[-1]["lin_vel_x"]
    else:
        cfg.curriculum["sprint_speed"] = CurriculumTermCfg(
            func=microduck_mdp.twist_command_range_curriculum,
            params={
                "command_name": "twist",
                "range_stages": SPRINT_SPEED_STAGES,
            },
        )

    return cfg


# Same PPO hyperparameters as walking; the task differs only in what it is
# asked to do. max_iterations covers the full ramp (7 stages x 1000) plus room
# to consolidate the last one.
MicroduckSprintRlCfg = dataclasses.replace(
    MicroduckRlCfg,
    experiment_name="sprint",
    run_name="sprint",
    max_iterations=12_000,
)
