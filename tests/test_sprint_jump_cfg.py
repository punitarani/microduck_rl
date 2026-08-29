"""Config invariants for the sprint and jump tasks.

These lock in the things that are silently wrong rather than loudly broken: a
penalty whose sign got flipped pays for the violation, a gait term left in the
jump stack makes walking cheaper than jumping, and a command slot pinned to
exactly zero kills its input neurons for the whole 61D contract.
"""

import math

import torch

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    JUMP_TILT_GATE_DEG,
    make_microduck_jump_env_cfg,
)
from mjlab_microduck.tasks.microduck_sprint_env_cfg import (
    SPRINT_SPEED_STAGES,
    make_microduck_sprint_env_cfg,
)

# ── sprint ───────────────────────────────────────────────────────────────────


def test_sprint_starts_at_stage_zero_not_the_walking_range():
    """The curriculum owns lin_vel_x from step 0, so the config cannot disagree
    with the curriculum about the starting speed."""
    cfg = make_microduck_sprint_env_cfg()
    assert cfg.commands["twist"].ranges.lin_vel_x == SPRINT_SPEED_STAGES[0]["lin_vel_x"]
    assert "sprint_speed" in cfg.curriculum


def test_sprint_speed_stages_are_monotonic_and_start_reachable():
    """Stage 0 must be at/above the walking recipe's proven 0.4, and each stage
    must be faster than the last — a non-monotonic ramp would quietly ask the
    policy to slow down mid-run."""
    tops = [s["lin_vel_x"][1] for s in SPRINT_SPEED_STAGES]
    steps = [s["step"] for s in SPRINT_SPEED_STAGES]
    assert tops[0] >= 0.4
    assert tops == sorted(tops) and len(set(tops)) == len(tops)
    assert steps == sorted(steps) and steps[0] == 0
    for s in SPRINT_SPEED_STAGES:
        lo, hi = s["lin_vel_x"]
        assert lo < 0 < hi, "keep a backward component so the slot stays alive"


def test_sprint_keeps_lateral_and_turn_alive():
    """Narrowed for straight-line focus, never zeroed: a command input that is
    never non-zero has dead weights forever."""
    r = make_microduck_sprint_env_cfg().commands["twist"].ranges
    for lo, hi in (r.lin_vel_y, r.ang_vel_z):
        assert lo < 0 < hi


def test_sprint_play_pins_the_top_stage():
    """`play` runs no curriculum, so without this the viewer shows a 0.45 m/s
    walk and the sprint is invisible."""
    cfg = make_microduck_sprint_env_cfg(play=True)
    assert cfg.commands["twist"].ranges.lin_vel_x == SPRINT_SPEED_STAGES[-1]["lin_vel_x"]
    assert "sprint_speed" not in cfg.curriculum


# ── jump ─────────────────────────────────────────────────────────────────────


def test_jump_drops_the_walking_gait_recipe():
    """air_time pays for alternating single-support — a walk, and a cheaper way
    to earn air-time reward than a jump."""
    r = make_microduck_jump_env_cfg().rewards
    for name in ("air_time", "foot_clearance", "foot_swing_height"):
        assert name not in r, f"{name} rewards a gait, not a jump"
    assert "jump_record" in r and "jump_flight" in r


def test_jump_objective_outweighs_standing_still():
    """A 5 cm jump must be worth more than a second of merely standing upright,
    or height is invisible next to the inherited stack. Both objective terms
    count: the record drives the height ceiling, the payout drives repetition,
    and it is their SUM the policy is choosing against standing."""
    r = make_microduck_jump_env_cfg().rewards
    five_cm = 0.05
    record = five_cm * r["jump_record"].weight
    payout = (five_cm ** 2 / r["jump_payout"].params["ref_height"]) * r["jump_payout"].weight
    one_second_upright = 50 * r["upright"].weight
    assert record + payout > one_second_upright


def test_jump_impact_penalty_has_the_self_negating_sign():
    """trunk_vertical_accel_penalty returns -|a_z|, so a NEGATIVE weight would
    double-negate into paying for violent landings."""
    r = make_microduck_jump_env_cfg().rewards
    assert r["gentle_landing"].weight > 0


def test_jump_does_not_pin_the_home_pose():
    """A jump is a deep crouch and a violent extension — maximal deviation from
    home. The walking recipe's pose weight fights exactly that."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    walk = make_microduck_velocity_env_cfg().rewards["pose"].weight
    jump = make_microduck_jump_env_cfg().rewards["pose"].weight
    assert jump < walk


def test_jump_keeps_command_slots_alive():
    """Jump in place, but never at exactly zero — this policy shares the 61D obs
    contract with the walker."""
    r = make_microduck_jump_env_cfg().commands["twist"].ranges
    for lo, hi in (r.lin_vel_x, r.lin_vel_y, r.ang_vel_z):
        assert lo < 0 < hi


def test_jump_smoothness_is_introduced_after_discovery():
    """Any attempt-tax active while a hard skill is being explored makes
    'do nothing' win, so action_rate must start near zero and grow."""
    cfg = make_microduck_jump_env_cfg()
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[0]["step"] == 0
    assert abs(stages[0]["weight"]) < abs(stages[-1]["weight"])
    assert all(s["weight"] <= 0 for s in stages)


def test_jump_peak_metric_is_readable_but_not_an_objective():
    """Weight 0 would log 0.0000 no matter how high the robot jumps, because
    Episode_Reward logs the WEIGHTED value. It must be non-zero to be readable,
    and negligible against the objective so it cannot be farmed."""
    r = make_microduck_jump_env_cfg().rewards
    metric = r["jump_peak_m"].weight
    five_cm = 0.05
    # Printed to 4 decimals, so it must clear 1e-4 for a realistic jump.
    assert metric * five_cm >= 1e-4, "invisible at the trainer's print precision"
    # ...and stay under a few percent of the objective for the same jump.
    metric_episode = metric * five_cm * 1000  # paid every step once set
    assert metric_episode < 0.05 * r["jump_record"].weight * 0.05


# ── the reward functions themselves ──────────────────────────────────────────


class _FakeAsset:
    def __init__(self, data):
        self.data = data


class _Data:
    def __init__(self, z, quat):
        self.root_link_pos_w = torch.tensor([[0.0, 0.0, v] for v in z])
        self.root_link_quat_w = quat


class _FakeScene:
    def __init__(self, asset, n):
        self._asset = asset
        self.sensors = {}
        self.terrain = type("T", (), {"env_origins": torch.zeros(n, 3)})()

    def __getitem__(self, _):
        return self._asset


class _FakeEnv:
    """Minimal stand-in: the jump rewards touch only scene/asset/episode state."""

    def __init__(self, z, airborne, tilt_deg=0.0):
        n = len(z)
        half = math.radians(tilt_deg) / 2.0
        quat = torch.tensor([[math.cos(half), math.sin(half), 0.0, 0.0]] * n)
        self.num_envs = n
        self.device = "cpu"
        self.scene = _FakeScene(_FakeAsset(_Data(z, quat)), n)
        self.episode_length_buf = torch.full((n,), 10)
        self.step_dt = 0.02  # 50 Hz control, as the envs run
        self._airborne = torch.tensor(airborne)


def _patch_contact(monkeypatch, env):
    monkeypatch.setattr(
        microduck_mdp, "_sensor_any_contact", lambda e, name: ~env._airborne
    )


def test_jump_record_pays_only_for_beating_the_record(monkeypatch):
    """The anti-farm property: a second identical hop pays exactly zero."""
    env = _FakeEnv(z=[0.167], airborne=[True])
    _patch_contact(monkeypatch, env)
    kw = dict(sensor_name="feet", tilt_gate_deg=JUMP_TILT_GATE_DEG)

    first = microduck_mdp.jump_record_progress(env, **kw)
    assert math.isclose(first.item(), 0.05, abs_tol=1e-6)  # 0.167 - 0.117

    repeat = microduck_mdp.jump_record_progress(env, **kw)
    assert repeat.item() == 0.0, "repeating a hop must pay nothing"

    env.scene._asset.data.root_link_pos_w[0, 2] = 0.187
    higher = microduck_mdp.jump_record_progress(env, **kw)
    assert math.isclose(higher.item(), 0.02, abs_tol=1e-6), "only the increment"


def test_jump_record_ignores_standing_tall(monkeypatch):
    """Standing on tiptoes is the cheap non-jump way to raise trunk z; the
    flight gate is what makes it worth nothing."""
    env = _FakeEnv(z=[0.30], airborne=[False])
    _patch_contact(monkeypatch, env)
    reward = microduck_mdp.jump_record_progress(
        env, sensor_name="feet", tilt_gate_deg=JUMP_TILT_GATE_DEG
    )
    assert reward.item() == 0.0


def test_jump_record_ignores_a_tumble(monkeypatch):
    """A tumble also leaves the ground. Without the tilt gate it would score as
    the best jump of the episode."""
    env = _FakeEnv(z=[0.30], airborne=[True], tilt_deg=80.0)
    _patch_contact(monkeypatch, env)
    reward = microduck_mdp.jump_record_progress(
        env, sensor_name="feet", tilt_gate_deg=JUMP_TILT_GATE_DEG
    )
    assert reward.item() == 0.0


def test_jump_record_resets_between_episodes(monkeypatch):
    """A fresh episode must start with no record, or the first real jump of
    episode two pays nothing."""
    env = _FakeEnv(z=[0.167], airborne=[True])
    _patch_contact(monkeypatch, env)
    kw = dict(sensor_name="feet", tilt_gate_deg=JUMP_TILT_GATE_DEG)
    microduck_mdp.jump_record_progress(env, **kw)

    env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    again = microduck_mdp.jump_record_progress(env, **kw)
    assert math.isclose(again.item(), 0.05, abs_tol=1e-6)


def test_jump_flight_bonus_min_height_rejects_a_twitch(monkeypatch):
    """The unguarded flight bonus pays per airborne STEP, so N short hops earn
    as much as one real flight — a mid-run harvest measured 75 flights in 15 s
    with a 20 ms median. min_height makes a twitch worth exactly nothing."""
    twitch = _FakeEnv(z=[0.120], airborne=[True])   # 3 mm off stance
    real = _FakeEnv(z=[0.150], airborne=[True])     # 3.3 cm off stance
    for env in (twitch, real):
        _patch_contact(monkeypatch, env)

    kw = dict(sensor_name="feet", tilt_gate_deg=JUMP_TILT_GATE_DEG)
    assert microduck_mdp.jump_flight_bonus(twitch, **kw).item() == 1.0, "ungated pays"
    assert microduck_mdp.jump_flight_bonus(twitch, min_height=0.02, **kw).item() == 0.0
    assert microduck_mdp.jump_flight_bonus(real, min_height=0.02, **kw).item() == 1.0


# ── tricks ───────────────────────────────────────────────────────────────────

from mjlab_microduck.tasks.microduck_tricks_env_cfg import (  # noqa: E402
    SPIN_RATE_STAGES,
    SUPPORT_MIN_HEIGHT,
    TRICKS,
    make_microduck_trick_env_cfg,
)


def test_sprint_keeps_the_walking_recipes_turn_coverage():
    """Narrowing this to buy forward experience produced a policy that ran in a
    circle: zero yaw command, 0.28-0.34 rad/s of drift. +/-1.0 is the value the
    walking recipe calls 'the big change - it makes turning learnable'."""
    cfg = make_microduck_sprint_env_cfg()
    assert cfg.commands["twist"].ranges.ang_vel_z == (-1.0, 1.0)
    assert cfg.commands["twist"].rel_turn_in_place_envs >= 0.15


def test_every_trick_builds_and_drops_the_gait_recipe():
    for trick in TRICKS:
        r = make_microduck_trick_env_cfg(trick).rewards
        for name in ("air_time", "foot_clearance", "foot_swing_height"):
            assert name not in r, f"{trick} kept {name}, which pays for a walk"
        assert "support" in r


def test_one_leg_tricks_require_a_SUSTAINED_hold():
    """'Exactly one foot' is still satisfied by alternating feet at every
    instant, which is what run 1 actually learned. Only the sustained version
    tracks which foot bears weight and resets on a switch."""
    for trick in ("one_leg_stand", "spin_one_leg"):
        r = make_microduck_trick_env_cfg(trick).rewards["support"]
        assert r.func is microduck_mdp.sustained_single_support
        assert r.params["hold_target_s"] >= 1.0
    two = make_microduck_trick_env_cfg("spin_two_leg").rewards["support"].func
    assert two is microduck_mdp.double_support_reward


def test_support_gate_has_a_height_floor():
    """A robot folded onto one knee also has exactly one foot down; without a
    floor that collapse is cheaper than balancing."""
    for trick in TRICKS:
        p = make_microduck_trick_env_cfg(trick).rewards["support"].params
        assert p["min_height"] >= SUPPORT_MIN_HEIGHT > 0.0


def test_spin_rate_is_capped_and_ramped():
    """Uncapped |omega_z| pays most for falling over, which is how a 25 cm biped
    spins fastest. And +/-4.0 from step 0 repeats the mistake the walking recipe
    recorded when a ramp to +/-2.0 outpaced the robot."""
    cfg = make_microduck_trick_env_cfg("spin_two_leg")
    assert cfg.rewards["spin_rate"].params["target_rate"] > 0
    assert "spin_rate_range" in cfg.curriculum
    tops = [s["ang_vel_z"][1] for s in SPIN_RATE_STAGES]
    assert tops == sorted(tops) and tops[0] < tops[-1]
    assert cfg.commands["twist"].ranges.ang_vel_z == SPIN_RATE_STAGES[0]["ang_vel_z"]


def test_spin_play_pins_the_top_rate():
    cfg = make_microduck_trick_env_cfg("spin_two_leg", play=True)
    assert cfg.commands["twist"].ranges.ang_vel_z == SPIN_RATE_STAGES[-1]["ang_vel_z"]


def test_unknown_trick_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        make_microduck_trick_env_cfg("moonwalk")


def test_jump_payout_pays_per_flight_and_favours_height(monkeypatch):
    """A pure episode-max objective made the policy jump once and then stand
    still. The payout pays on each LANDING, and squares peak so trading height
    for repetitions loses."""
    env = _FakeEnv(z=[0.167], airborne=[True])  # 5 cm up
    _patch_contact(monkeypatch, env)
    kw = dict(sensor_name="feet", tilt_gate_deg=JUMP_TILT_GATE_DEG, ref_height=0.05)

    assert microduck_mdp.jump_flight_payout(env, **kw).item() == 0.0, "nothing mid-flight"
    env._airborne = torch.tensor([False])          # land
    paid = microduck_mdp.jump_flight_payout(env, **kw).item()
    assert math.isclose(paid, 0.05 ** 2 / 0.05, abs_tol=1e-6)  # float32
    assert microduck_mdp.jump_flight_payout(env, **kw).item() == 0.0, "paid once only"


def test_jump_payout_is_superlinear_in_height(monkeypatch):
    """One 4 cm jump must beat two 2 cm jumps, or frequency wins over height."""
    def one_jump(z):
        env = _FakeEnv(z=[z], airborne=[True])
        _patch_contact(monkeypatch, env)
        kw = dict(sensor_name="feet", tilt_gate_deg=JUMP_TILT_GATE_DEG, ref_height=0.05)
        microduck_mdp.jump_flight_payout(env, **kw)
        env._airborne = torch.tensor([False])
        return microduck_mdp.jump_flight_payout(env, **kw).item()

    assert one_jump(0.117 + 0.04) > 2 * one_jump(0.117 + 0.02)


def _patch_feet(monkeypatch, seq):
    """Drive _feet_contact_pair from a scripted (left, right) sequence."""
    state = {"i": 0}

    def fake(env, name):
        l, r = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        return torch.tensor([l]), torch.tensor([r])

    monkeypatch.setattr(microduck_mdp, "_feet_contact_pair", fake)


def test_sustained_support_rejects_alternating_feet(monkeypatch):
    """The cheat that 'exactly one foot' did NOT prevent: swapping feet
    satisfies exactly-one at every instant. Run 1 scored 88% single support
    with a 0.21 s longest hold — a hopping spin wearing a balance task's
    numbers. Alternation must reset the clock to ~0."""
    env = _FakeEnv(z=[0.119], airborne=[False])
    _patch_feet(monkeypatch, [(True, False), (False, True)] * 10)
    kw = dict(sensor_name="feet", tilt_gate_deg=25.0, min_height=0.10, hold_target_s=2.0)
    rewards = [microduck_mdp.sustained_single_support(env, **kw).item() for _ in range(20)]
    assert max(rewards) <= env.step_dt / 2.0 + 1e-6, "alternating must not accumulate"


def test_sustained_support_rewards_holding_one_foot(monkeypatch):
    """Holding the SAME foot accumulates toward full credit."""
    env = _FakeEnv(z=[0.119], airborne=[False])
    _patch_feet(monkeypatch, [(True, False)] * 200)
    kw = dict(sensor_name="feet", tilt_gate_deg=25.0, min_height=0.10, hold_target_s=2.0)
    rewards = [microduck_mdp.sustained_single_support(env, **kw).item() for _ in range(200)]
    assert rewards[0] < rewards[50] < rewards[-1], "a longer hold must pay more"
    assert rewards[-1] == 1.0, "reaching the target pays full credit"


def test_sustained_support_ignores_double_support(monkeypatch):
    env = _FakeEnv(z=[0.119], airborne=[False])
    _patch_feet(monkeypatch, [(True, True)] * 20)
    kw = dict(sensor_name="feet", tilt_gate_deg=25.0, min_height=0.10, hold_target_s=2.0)
    assert all(microduck_mdp.sustained_single_support(env, **kw).item() == 0.0
               for _ in range(20))


# ── headstand ────────────────────────────────────────────────────────────────

from mjlab_microduck.tasks.microduck_headstand_env_cfg import (  # noqa: E402
    make_microduck_headstand_env_cfg,
)


def test_headstand_uses_the_all_collisions_robot():
    """robot_walk.xml has exactly two colliding geoms, both feet — the head
    passes through the floor, which is fatal when the task is to balance on it."""
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG

    cfg = make_microduck_headstand_env_cfg()
    assert cfg.scene.entities["robot"] is MICRODUCK_STANDUP_ROBOT_CFG


def test_headstand_drops_upright_and_the_fall_termination():
    """`upright` rewards the opposite of the goal, and `fell_over` terminates on
    trunk tilt — which a headstand deliberately violates, so leaving it in ends
    the episode at the moment of success."""
    cfg = make_microduck_headstand_env_cfg()
    assert cfg.rewards["upright"].weight == 0.0
    assert "fell_over" not in cfg.terminations


def test_headstand_hold_needs_both_gates(monkeypatch):
    """Feet-above-head alone is satisfied mid-somersault; the head must also be
    near the floor for it to be a stand rather than a moment of a tumble."""
    calls = {}

    def fake_geom(env, head_cfg, feet_cfg):
        return torch.tensor([calls["inv"]]), torch.tensor([calls["head_z"]])

    monkeypatch.setattr(microduck_mdp, "_headstand_geometry", fake_geom)
    env = _FakeEnv(z=[0.1], airborne=[False])
    kw = dict(head_cfg=None, feet_cfg=None, min_inversion=0.05,
              max_head_height=0.06, hold_target_s=2.0)

    calls.update(inv=0.15, head_z=0.20)  # inverted but head high: a somersault
    assert microduck_mdp.headstand_hold(env, **kw).item() == 0.0
    calls.update(inv=-0.20, head_z=0.03)  # head low but upright: just standing
    assert microduck_mdp.headstand_hold(env, **kw).item() == 0.0
    calls.update(inv=0.15, head_z=0.03)  # both: a real headstand
    assert microduck_mdp.headstand_hold(env, **kw).item() > 0.0


def test_jump_keeps_upright_pressure_against_a_squared_payout():
    """A squared per-flight payout makes aggression profitable; run 3 produced
    2 cm of height with up to 100 flights and 169° of tilt. The upright term has
    to outweigh the walking recipe's default, not be turned down with the rest."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    jump = make_microduck_jump_env_cfg().rewards
    walk = make_microduck_velocity_env_cfg().rewards
    assert jump["upright"].weight > walk["upright"].weight
    assert jump["gentle_landing"].weight > 0
