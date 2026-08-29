"""Microduck JUMP — maximise peak trunk height in a genuine flight phase.

There is no jump task upstream, so this is built the way the playbook says to
build a dynamic maneuver: reuse the velocity recipe wholesale (robot, 61D obs,
DR, noise, delays, NaN guards, sim2real stack) and change only the command
distribution and the reward recipe.

The objective is `jump_record`: Δ(episode-best flight height), where height only
counts while BOTH feet are off the ground and the trunk is upright. Paying only
for a new personal best makes "as high as possible" the literal argmax — a
second identical hop pays exactly zero, so there is nothing to farm and no
jackpot to rate-limit. `jump_flight` pays for being airborne at all, because the
record term is silent until the robot has already left the ground once, which is
not something PPO stumbles into; the curriculum decays it once takeoff exists so
the record term decides how high.

What had to be turned DOWN, and why — these are the terms that make an excellent
walker and a hopeless jumper:

- `pose` (1.0 → 0.1): rewards staying near the home stance. A jump is a deep
  crouch followed by a violent extension, i.e. maximal deviation from home.
- `air_time` (3.0, REMOVED): the walking gait's per-foot air-time reward pays
  for alternating single-support. That is a walk, and it is a cheaper way to
  earn air-time reward than a jump. `jump_flight` replaces it and requires
  BOTH feet off.
- `foot_clearance` (-2.0) and `foot_swing_height` (-0.25), REMOVED: swing-height
  shaping for a gait, which taxes the tuck and the extension.
- `body_ang_vel` / `angular_momentum` halved: motion-blockers penalise what a
  dynamic motion physically requires; the playbook says keep them low here.
- `action_rate` starts near zero and ramps: any attempt-tax active while a hard
  skill is being explored makes "do nothing" win. Smoothness comes AFTER
  discovery.

What is deliberately NOT delayed is the landing-impact cost. Roulade's run-1
lesson was that a violent solution discovered under zero impact cost gets locked
in, so `gentle_landing` is active from step 0. It is self-negating (returns
-|a_z|) and therefore takes a POSITIVE weight — a negative one would pay for
violence.

The twist command is squeezed to near zero (jump in place) but never to exactly
zero: a command slot that is never non-zero has dead weights forever, and this
policy shares the 61D obs contract with the walker so its command neurons have
to stay alive for the runtime to hot-swap it.
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

# Gait-shaping terms that describe a walk, not a jump.
_WALKING_GAIT_TERMS = ("air_time", "foot_clearance", "foot_swing_height")

# Jump in place. Non-zero so the command neurons stay alive (61D obs contract).
JUMP_LIN_VEL_X = (-0.05, 0.05)
JUMP_LIN_VEL_Y = (-0.05, 0.05)
JUMP_ANG_VEL_Z = (-0.1, 0.1)

# Both feet off the ground AND within this tilt to count as flight: a tumble
# also leaves the ground, and a face-first launch is not a jump.
JUMP_TILT_GATE_DEG = 30.0

FEET_CONTACT_SENSOR = "feet_ground_contact"

# Weights are calibrated against observed Episode_Reward magnitudes in the
# smoke test, not guessed — the playbook's "compare reward mass, not weights".
# Sized against the standing stack rather than guessed. `upright` pays weight
# 2.0 every step, so ~1 s of merely standing there is worth ~100 raw units. A
# record term in METRES at weight 500 makes a whole 5 cm jump worth 25 — about
# 1% of an episode of standing, i.e. invisible. At 3000 a 5 cm jump is worth
# ~150, which competes. This is the playbook's "compare reward mass, not
# weights": the same weight means nothing without the stack it sits in.
# Run 2 measured the cost of a pure episode-max objective: the policy jumped
# ONCE out of the reset transient (0.9 cm) and then stood still for the rest of
# the episode, because a repeat jump of the same height pays nothing while still
# costing impact and action-rate. So the record term keeps a reduced weight (it
# is still what drives height records) and the per-flight payout below supplies
# the reason to keep jumping at all.
JUMP_RECORD_WEIGHT = 1200.0  # metres of new record; 5 cm -> 60
JUMP_FLIGHT_PAYOUT_WEIGHT = 3000.0  # peak^2/0.05 per landing; 4 cm -> ~96
JUMP_PAYOUT_REF_HEIGHT = 0.05
JUMP_FLIGHT_WEIGHT = 1.0     # per airborne step, decayed by curriculum

# Run 1 measured 75 flights in 15 s with a 20 ms median — 5 Hz vibration, not
# jumping. Paying per airborne STEP means N twitches earn as much as one real
# flight of equal total airtime, and twitches are far easier to find; at 9%
# airborne the flight bonus was worth ~180 against the record term's ~45, so
# micro-hopping WAS the reward-maximising behaviour. Requiring 1 cm of real
# clearance makes a twitch pay exactly zero. 1 cm and not 2: the run had already
# reached 1.5 cm, so this is a threshold it can clear today rather than one that
# switches the discovery aid off entirely.
JUMP_FLIGHT_MIN_HEIGHT = 0.01
# Sized to be READABLE, which 0 and 1e-4 both failed to be: the trainer prints
# Episode_Reward to four decimals, so a 5 cm jump at 1e-4 still shows 0.0000.
# At 0.02 it prints, and its whole-episode contribution (~1.0) is under 1% of
# the record term's ~150 for the same jump — visible, not steerable.
JUMP_PEAK_METRIC_WEIGHT = 0.02


def make_microduck_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Velocity env with a jump objective replacing the gait objective."""
    cfg = make_microduck_velocity_env_cfg(play=play)

    # The walking robot has exactly two colliding geoms and both are feet, so a
    # bad landing has nothing to catch it: the trunk passes through the floor and
    # the robot keeps tipping. Measured consequence — 0/152 trials ended upright
    # across every condition, and a fallen robot reads as 91% "airborne" because
    # its feet are simply off the ground. Every task in this repo that meets the
    # ground (standup, roulade, sitstand) uses the all-collisions model; the jump
    # needed it too and inherited the walking one by default.
    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}

    trunk = SceneEntityCfg("robot", body_names=("trunk_base",))

    # --- command: hold station, jump in place ---
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = JUMP_LIN_VEL_X
    command.ranges.lin_vel_y = JUMP_LIN_VEL_Y
    command.ranges.ang_vel_z = JUMP_ANG_VEL_Z
    command.rel_turn_in_place_envs = 0.0
    command.rel_standing_envs = 0.0  # every env is a jumping env

    # --- drop the gait recipe ---
    for name in _WALKING_GAIT_TERMS:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # --- turn down what fights a crouch-and-extend ---
    # ...but NOT upright. Run 3 paid per flight and got what it paid for: 2 cm
    # of height with 23-100 flights per rollout and a trunk tilt of 122-169°,
    # i.e. flailing. Squaring the payout makes aggression profitable, so the
    # counterweight has to be raised with it or "jump" degenerates into "throw
    # yourself upward". Landing impact is raised for the same reason.
    cfg.rewards["upright"].weight = 4.0
    cfg.rewards["pose"].weight = 0.1
    cfg.rewards["body_ang_vel"].weight = -0.02
    cfg.rewards["angular_momentum"].weight = -0.01
    cfg.rewards["action_rate_l2"].weight = -0.005  # ramped up after discovery
    # Station-keeping only; at 2.0 the tracking terms dominate the jump.
    cfg.rewards["track_linear_velocity"].weight = 0.5
    cfg.rewards["track_angular_velocity"].weight = 0.5
    # The head is 38% of body mass — its swing is a jump technique, not noise.
    cfg.rewards["head_pose_tracking"].weight = 0.3

    # --- the objective ---
    cfg.rewards["jump_record"] = RewardTermCfg(
        func=microduck_mdp.jump_record_progress,
        weight=JUMP_RECORD_WEIGHT,
        params={
            "sensor_name": FEET_CONTACT_SENSOR,
            "asset_cfg": trunk,
            "tilt_gate_deg": JUMP_TILT_GATE_DEG,
        },
    )
    cfg.rewards["jump_flight"] = RewardTermCfg(
        func=microduck_mdp.jump_flight_bonus,
        weight=JUMP_FLIGHT_WEIGHT,
        params={
            "sensor_name": FEET_CONTACT_SENSOR,
            "asset_cfg": trunk,
            "tilt_gate_deg": JUMP_TILT_GATE_DEG,
            "min_height": JUMP_FLIGHT_MIN_HEIGHT,
        },
    )
    # Metric, not objective. The weight is 1e-4 rather than 0 because
    # Episode_Reward logs the WEIGHTED value, so a weight-0 term reads 0.0000
    # no matter how high the robot jumps — which is exactly what it did on the
    # first run, leaving the height curve invisible for the whole session.
    # Recover metres as Episode_Reward/jump_peak_m / 1e-4. At this weight its
    # contribution is ~5e-6 against a 3000-weight objective: unfarmable noise.
    cfg.rewards["jump_peak_m"] = RewardTermCfg(
        func=microduck_mdp.jump_peak_height,
        weight=JUMP_PEAK_METRIC_WEIGHT,
        params={
            "sensor_name": FEET_CONTACT_SENSOR,
            "asset_cfg": trunk,
            "tilt_gate_deg": JUMP_TILT_GATE_DEG,
        },
    )
    # Paid once per flight, on landing, as peak^2 — the reason to jump AGAIN.
    # Squared so height stays the point: one 4 cm jump beats two 2 cm ones.
    cfg.rewards["jump_payout"] = RewardTermCfg(
        func=microduck_mdp.jump_flight_payout,
        weight=JUMP_FLIGHT_PAYOUT_WEIGHT,
        params={
            "sensor_name": FEET_CONTACT_SENSOR,
            "asset_cfg": trunk,
            "tilt_gate_deg": JUMP_TILT_GATE_DEG,
            "ref_height": JUMP_PAYOUT_REF_HEIGHT,
        },
    )
    # Self-negating (-|a_z|) → POSITIVE weight. Active from step 0 so a violent
    # landing never becomes the locked-in solution (roulade run-1 lesson).
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.01,
        params={"asset_cfg": trunk},
    )

    if not play:
        # Smoothness AFTER the skill exists, never during discovery.
        cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.005},
                    {"step": 800 * 24, "weight": -0.02},
                    {"step": 1600 * 24, "weight": -0.05},
                    {"step": 2400 * 24, "weight": -0.10},
                ],
            },
        )
        # Hand the objective over: once takeoff is reliable, stop paying for
        # merely being airborne so height is the only thing left to earn.
        cfg.curriculum["jump_flight_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "jump_flight",
                # Handed over far earlier than run 1's 1200/2400: the record
                # term has to be the dominant signal before the flight bonus has
                # time to teach a habit.
                "weight_stages": [
                    {"step": 0, "weight": JUMP_FLIGHT_WEIGHT},
                    {"step": 800 * 24, "weight": 0.5},
                    {"step": 1600 * 24, "weight": 0.2},
                ],
            },
        )

    return cfg


# Episodic trick budget is ~1000 iters at 4096 envs per the playbook; allow
# room for the two curricula to land and consolidate.
MicroduckJumpRlCfg = dataclasses.replace(
    MicroduckRlCfg,
    experiment_name="jump",
    run_name="jump",
    max_iterations=6_000,
)
