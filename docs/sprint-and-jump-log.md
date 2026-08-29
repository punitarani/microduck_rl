# Sprint, jump & tricks — autonomous training log

Goal: run fast, jump high, and learn tricks. Budget ceiling $25 (Modal).
2026-08-29, 00:35–06:00 MST. All times MST, observed.

## Results — measured, not inferred

Every number below is from a real rollout in CPU MuJoCo through
`infer_policy.PolicyInference`, the same observation builder the runtime uses.
Policies live in `runs/BEST/` (gitignored); all verified `obs[1,61] -> actions[1,14]`.

| policy | measurement | verdict |
|---|---|---|
| **headstand** | feet **+0.123 m** above head, head 0.1 cm off floor, **11.68 s hold** of a 12 s rollout, 97% inverted | **works** |
| **spin, two legs** | **12.5 revolutions / 12 s at 6.54 rad/s**, 94% double-support, 2° max tilt, never falls | **works** |
| **sprint (fastest)** | **0.758 m/s** vs 0.449 baseline (**+69%**) | works, but drifts +0.34 rad/s at zero yaw command |
| **sprint (steerable)** | 0.600 m/s, drift +0.141, tracks both directions | works; steering cost 21% of speed |
| jump (highest) | **2.6 cm**, 42 flights per 12 s — but ends fallen every time | partial |
| jump (stable) | 0.9 cm, stays upright, jumps once per episode | partial |
| one-leg stand | trains to 95% of hold target, does not transfer | diagnosed, not solved |
| one-leg spin | 8.2 rev/12 s but 0.23 s longest hold — alternating, not holding | not solved |
| head spin | not attempted | out of budget, stated rather than faked |

Baseline for sprint is the vendored production `alpha_walking.onnx`, measured on
the same harness: 0.449 m/s.

## What actually produced the results

Six mistakes, each caught by measuring behaviour rather than reading a reward.
They are the substance of this log and are written up in order below.

1. A **weight-0 metric** logs 0.0000 forever — `Episode_Reward` logs the
   WEIGHTED value. Then 1e-4 also logged 0.0000, because the trainer prints four
   decimals. Two rounds to make one number visible.
2. The jump farmed its own discovery aid: paying per airborne STEP made 75
   twitches worth more than one real jump.
3. "Exactly one foot down" does **not** prevent alternating feet — it is
   satisfied at every instant by hopping. 88% single-support with a 0.21 s
   longest hold.
4. An **episode-max** objective makes the optimal policy jump once out of the
   reset transient and then stand still forever.
5. Narrowing the sprint's turn envelope to buy forward experience produced a
   policy that **ran in a circle**.
6. The sprint ramp **outpaced the robot** and speed fell 28% while reward moved
   7% — reward alone would never have shown it.


## Machine choice (measured, not assumed)

`modal run modal_app.py::probe` at 4096 envs on `Mjlab-Velocity-Flat-MicroDuck`:

| GPU | s/iter | $/hr (cap math) | iters per $10 | verdict |
|---|---|---|---|---|
| L4 | 3.069 | 1.43 | 7,371 | |
| **L40S** | **1.440** | **2.58** | **8,707** | **best value** |
| A100-40GB (served 80GB) | 2.129 | 2.73 | 5,569 | slower *and* dearer |

The A100 result overturned the prior: MuJoCo Warp at this scale is not
bandwidth-bound, so the Ada part's clocks and SM count beat A100 memory
bandwidth. Two inferences died to measurement here — the README's implied
~1 s/iter (real L4 is 3.07), and "A100 will win because bandwidth".

A100 detail worth keeping: asking Modal for `A100-40GB` was served an
`A100 80GB PCIe`. If that bills at the 80GB rate, pricing a cap at the 40GB
rate overshoots $10 by 13% — `modal_app.py` now prices any A100 request as the
dearer card.

## Status

- [x] Probe GPUs, pick L40S
- [x] Sprint task authored, smoke-tested, launched ($12 cap)
- [x] Jump task authored, smoke-tested, launched ($8 cap)
- [ ] Evaluate, export ONNX, verify 61->14 contract
- [ ] Commit

## Design decisions

### Sprint — ramp the ceiling, don't assume it

The walking recipe pins `lin_vel_x` at ±0.4 and its comment says a ramp to
wider ranges "outpaced the robot's capability". So 0.4 is where walking was
known to work, not a measured limit.

Commanding a flat 1.0 m/s would be worse than useless: an unreachable command
isn't a harder task, it's a different one. The tracking reward saturates at
"as close as physics allows", the gradient toward *faster* vanishes, and the
policy learns to lean at a wall. So `Mjlab-Sprint-Flat-MicroDuck` ramps the
forward command 0.45 -> 1.05 in 7 stages of 1000 iterations, keeping the
command near the frontier where the gradient still means something. Lateral and
turn ranges are narrowed (not zeroed — dead command slots lose their neurons).

Needed one new mdp function, `twist_command_range_curriculum`: the existing
`pose_command_range_curriculum` sets `cfg.ranges` wholesale, which works for a
pose command's tuple but not a velocity command's named-field object.

**Reading it:** `Curriculum/sprint_speed` is the current max forward command.
When `Metrics/twist/error_vel_xy` steps up and *stays* up at a stage boundary,
that stage is the ceiling and the previous stage is the honest top speed.

### Jump — pay only for beating your own record

No jump task exists upstream. Built on the velocity recipe (robot, 61D obs, DR,
NaN guards come free) with the gait objective swapped out.

The trap: "be high" has a cheap non-jump solution — stand tall on tiptoes. So
every positive term is gated on FLIGHT (no foot contact) *and* upright, which
is the hard state-based gate the playbook demands instead of a penalty nudge.

The objective `jump_record` pays Δ(episode-best flight height). Height already
achieved is already paid, so a second identical hop pays exactly zero and the
only way to earn more is to go higher — "as high as possible" as the literal
argmax, no jackpot to rate-limit. `jump_flight` pays for being airborne at all
(the record term is silent until the robot has left the ground once, which PPO
doesn't stumble into) and the curriculum decays it once takeoff exists.

What had to be turned down, and why it matters — these terms make an excellent
walker and a hopeless jumper:

| term | change | reason |
|---|---|---|
| `air_time` | removed | pays for alternating single-support: a walk, and cheaper than a jump |
| `foot_clearance`, `foot_swing_height` | removed | gait swing shaping; taxes the tuck and extension |
| `pose` | 1.0 -> 0.1 | pins the home stance; a jump is maximal deviation from it |
| `body_ang_vel`, `angular_momentum` | halved | motion-blockers penalise what a dynamic move requires |
| `action_rate` | -0.005, ramped | an attempt-tax during discovery makes "do nothing" win |
| `gentle_landing` | on from step 0 | roulade run-1: a violent solution found under zero impact cost locks in |

**Reward-mass calibration** (the playbook's "compare reward mass, not weights"):
`upright` pays 2.0 every step, so ~1 s of merely standing is ~100 raw units. At
the initial weight 500, a whole 5 cm jump was worth 25 — about 1% of an episode
of standing, i.e. invisible. Raised to 3000 (5 cm -> 150), which competes. The
smoke test is what exposed this; a test now asserts the relationship.

## Log

### 00:35 — start
L40S selected. Read AGENTS.md playbook before touching env cfgs.

### 00:42 — sprint launched
Smoke test (64 envs, 5 iters) green: `sprint_speed` curriculum active at 0.45,
tracking rewards computing, no NaN. Launched on L40S, $12 cap -> 15,046 s.

### 00:46 — jump launched
Smoke test green; every penalty term <= 0 (the infallible sign check). Two bugs
caught before launch: `reward_weight` takes `reward_name`/`weight_stages`, not
`term_name`/`stages`; and the reward mass above. Launched on L40S, $8 cap.

15 cfg//mdp tests added (`tests/test_sprint_jump_cfg.py`), all passing —
including the anti-farm properties: a repeat hop pays zero, standing tall pays
zero, a tumble pays zero, and the record resets between episodes.

### 00:50 — evaluation harness, and a baseline to beat

`play` needs a GPU and a viewer, so there was no way to score a policy on this
Mac. `scripts/eval_sprint_jump.py` drives the ONNX through
`infer_policy.PolicyInference` — the same observation builder the real runtime
uses, so a 61D mismatch here would be a mismatch on the robot too — and reports
measured numbers headlessly.

Two harness bugs, both of which made a good policy look broken:

1. `mj_resetData` drops the robot at qpos0, which is not a stance. The vendored
   production walker scored 0.002 m/s. Fixed by replicating `infer_policy`'s
   init (trunk z=0.125, joints at default pose).
2. Measuring world-x displacement against a BODY-frame command. The robot veers
   (yaw drift -1.33 rad at cmd 0.6), so world-x understates real speed. Now
   compares body-frame vx to the body-frame command, like for like.

**Baseline — vendored `alpha_walking.onnx`:**

| cmd vx | achieved body vx | ratio |
|---|---|---|
| 0.2 | 0.000 | — |
| 0.4 | 0.177 | 0.44 |
| 0.6 | 0.288 | 0.48 |
| 0.8 | **0.449** | 0.56 |

The robot CAN exceed the 0.4 it was trained on — it just tracks poorly above it.
**0.45 m/s is the number sprint has to beat.**

### 00:51 — walked into the weight-0 metric trap

`jump_peak_m` was added at weight 0 to log height without paying for it. It
logged `0.0000` for the entire run, because — as AGENTS.md says outright —
`Episode_Reward` logs the WEIGHTED value, so a weight-0 term reads zero no
matter what the robot does. The height curve was invisible by construction.

Fixed to weight 1e-4 (recover metres as `Episode_Reward/jump_peak_m / 1e-4`);
its reward contribution is ~5e-6 against a 3000-weight objective. Not worth
restarting a healthy run over instrumentation, so this run is read via the
`jump_record` trend instead, and the real height comes from the eval harness at
the end. A test now asserts the metric is non-zero AND negligible.

Related reading trap: the `Episode_*` values are episode SUMS, so sprint's
"rising tracking error" early on was just episodes getting longer, not tracking
getting worse. The signal that matters is the main task term growing —
`track_linear_velocity` went 0.01 -> 0.93 over 150 iterations, which is healthy.

### 00:52 — both runs pacing to finish before wake-up

| run | elapsed | trainer ETA | budget cap | iters at cap |
|---|---|---|---|---|
| sprint | 0:09 | 5:21 (12,000 iters) | 15,046 s = 4h11 | ~9,200 |
| jump | 0:04 | 2:56 (6,000 iters) | 10,031 s = 2h47 | ~5,700 |

Both are budget-capped before their nominal max_iterations, which is by design:
the caps are the spend limit, and the sprint ramp reaches its last stage at
6,000 iterations so ~9,200 covers it with room to consolidate. Expected wall
clock: jump done ~03:33, sprint done ~04:53. Worst-case spend $20 of the $25.

### 00:57 — recordings wired in, versioned under `runs/` (gitignored)

`eval_sprint_jump.py --record out.mp4` renders offscreen with a camera locked to
the trunk (the scene's only camera is the robot's POV, useless for judging a
gait). `runs/` is gitignored — the videos are large and per-run.

Contact detection validated before trusting any airborne number: the harness
matches `left_foot_collision` / `right_foot_collision`, which are the ONLY
collision geoms in `robot_walk.xml` (trunk and head contacts are stripped —
"falling is cheap"). So "airborne" means genuinely both feet off.

### 00:57 — mid-run harvest at iteration 250: it hops, it does not jump

Harvested a checkpoint mid-training rather than waiting 2.5 h to find out
whether the pipeline or the policy was broken. Both work; the policy is doing
the wrong thing.

| metric | iter 250 |
|---|---|
| resting stance trunk z | 11.9 cm (confirms the 0.117 constant) |
| peak trunk z | 13.4 cm |
| **height gained** | **1.5 cm** |
| airborne fraction | 9.0% |
| distinct flights | **75 in 15 s** |
| median flight | 20 ms |

75 flights in 15 seconds is 5 Hz vibration, not jumping. This is `jump_flight`
being farmed exactly as feared: it pays per airborne STEP, so N short hops pay
the same as one long flight with the same total airtime — and short hops are
enormously easier to find.

**Holding, not intervening yet.** The flight-weight curriculum already decays at
iterations 1200 and 2400, which is designed for precisely this hand-off, and the
playbook says to expect a few rounds of reward-hacking whack-a-mole. Re-harvest
at ~1500 (after the first decay). If it is still micro-hopping then the fix is
prepared: gate `jump_flight` on a MINIMUM height above stance, so a 20 ms
twitch pays nothing and only a real flight counts.

### 01:02 — sprint at iteration 500 already beats the production walker

| cmd vx | body vx | ratio |
|---|---|---|
| 0.2 | 0.109 | 0.55 |
| 0.4 | 0.326 | 0.81 |
| 0.6 | 0.515 | 0.86 |
| **0.8** | **0.574** | 0.72 |
| 1.0 | 0.559 | 0.56 |
| 1.2 | 0.551 | 0.46 |

**0.574 m/s vs the 0.449 baseline — 28% faster at iteration 500 of ~9,200.**

Two things worth noting. First, the curve SATURATES: commanding 1.0 or 1.2 buys
nothing over 0.8, and the ratio column collapses (0.86 -> 0.72 -> 0.56 -> 0.46)
because the robot is already at its wall. That saturation is exactly what the
ramp was built to expose, and it is why a flat 1.0 m/s command would have been
the wrong design — above ~0.8 the command carries no gradient toward faster,
only a growing tracking error.

Second, the policy is generalising well past its training distribution: the
curriculum is still on stage 0 (max command 0.45) and it is doing 0.574 when
commanded 0.8. The ramp's job for the remaining ~8,700 iterations is to move the
trained distribution up to where the robot already is, which should tighten
tracking near the ceiling and, if there is headroom in the actuators, push the
ceiling itself.

Recordings: `runs/v1-sprint-iter500/` and `runs/v1-jump-iter250/`.

### 01:04 — jump run 2: stopped run 1 rather than let it entrench

The arithmetic from the iter-250 harvest made the call obvious:

| term | value per episode at run-1 behaviour |
|---|---|
| `jump_flight` | 9% airborne x 1000 steps x weight 2.0 = **~180** |
| `jump_record` | 1.5 cm record x weight 3000 = **~45** |

Micro-hopping was not the policy misbehaving, it was the policy correctly
maximising the reward I wrote. And the hand-off that was supposed to fix it did
not arrive until iteration 2400 — roughly 70 minutes of reinforcing a twitch
before the objective took over. Stopping cost ~12 minutes of training.

Run 2 changes, all three aimed at the same failure:

1. `jump_flight` now requires **1 cm of real clearance** above stance
   (`min_height`), so a 20 ms twitch pays exactly zero. 1 cm and not 2 because
   run 1 had already reached 1.5 cm — a threshold it can clear today, rather
   than one that switches the discovery aid off entirely.
2. `JUMP_FLIGHT_WEIGHT` 2.0 -> 1.0, halving the farmable term outright.
3. Decay moved 1200/2400 -> **800/1600**, so the record term becomes dominant
   before the flight bonus has time to teach a habit.

Also fixed the height metric properly. Weight 0 printed 0.0000 (weighted-value
trap); 1e-4 ALSO printed 0.0000, because the trainer prints four decimals and
1e-4 x 5 cm = 5e-6. It is 0.02 now: a 5 cm jump prints 0.0010, and the
whole-episode contribution (~1.0) is under 1% of the record term's ~150. Two
rounds to get one metric readable — worth writing down, because "add a
zero-weight metric" is the obvious move and it silently does nothing.

Spend so far: sprint $12 cap, jump run 1 ~$0.6 actual, jump run 2 $8 cap.
Worst case $20.6 of the $25.

### 01:08 — the sprint ceiling is rising, which justifies the ramp

Running tally of best sustained body-frame speed (all measured, 10 s per
command, CPU MuJoCo, same harness):

| checkpoint | best m/s | vs baseline |
|---|---|---|
| vendored `alpha_walking` (baseline) | 0.449 | — |
| sprint iter 500 | 0.574 | +28% |
| sprint iter 750 | **0.617** | **+37%** |

The ceiling moved 0.574 -> 0.617 in 250 iterations while the curriculum was
still on stage 0 (max command 0.45), so the gains are coming from the policy
getting better, not from being asked for more. That is the answer to whether the
later ramp stages are worth their iterations: yes, as long as the ceiling keeps
moving.

One warning sign: at cmd 1.2 the robot now FALLS, where at iter 500 it merely
saturated. Commanding far beyond the ceiling is not free — it destabilises. The
top ramp stage is 1.05, close enough to that to be worth watching, and it is the
concrete reason the final checkpoint gets compared against earlier ones rather
than trusted.

### 01:13 — jump v2 confirms the gate works and did not stall discovery

The gate's risk was the opposite failure: with the dense aid switched off below
1 cm, discovery could stall. It did not.

| iter | `jump_record` | `jump_flight` | `jump_peak_m` | reward |
|---|---|---|---|---|
| 10 | 0.021 | 0.0002 | 0.0000 | 1.2 |
| 40 | 0.043 | 0.0024 | 0.0000 | 2.9 |
| 80 | 0.069 | 0.0142 | 0.0001 | 6.7 |
| 120 | 0.075 | 0.0457 | 0.0002 | 21.6 |
| 160 | 0.079 | 0.0565 | 0.0003 | 29.3 |

Both terms climb. `jump_flight` rising from ~0 is direct evidence of REAL
flights: under the gate that term cannot be earned by a twitch at all.

Reward balance against run 1, which is what was actually broken:

| | objective | aid | |
|---|---|---|---|
| run 1 (ungated) | 0.082 | 0.088 | aid ≈ objective |
| run 2 (1 cm gate) | 0.079 | 0.057 | objective leads |

The height metric is now readable AND calibrated: `peak_m / 0.02` = 0.015 m at
iteration 160 — 1.5 cm, matching what the eval harness measured independently on
the run-1 checkpoint. Two independent paths to the same number is the cheapest
confirmation available that neither is lying.

### 01:12 — sprint curriculum advanced on schedule

`Curriculum/sprint_speed` flipped 0.45 -> 0.55 at iteration 1000 exactly as
configured, confirming `twist_command_range_curriculum` drives the live command
manager rather than a deepcopy (the classic silent no-op the playbook warns
about: writes to `env.cfg` never reach the managers).

### 01:30 — being critical: it was fast, but it could not steer

Straight-line speed keeps climbing — 0.617 at iter 750, **0.758 m/s at iter
1500**, against a 0.449 baseline (+69%). But a top speed you can only hold in a
straight line is a much weaker result than the number suggests, so I added a
turning sweep (`eval_sprint_jump.py turn`): hold a forward command, sweep yaw,
report achieved yaw rate and whether speed survives.

It does not steer. Holding cmd vx=0.6:

| cmd yaw | iter 750 achieved | iter 1500 achieved |
|---|---|---|
| -1.50 | -0.588 (ratio 0.39) | -0.281 (ratio 0.19) |
| **0.00** | **-0.275 (drifts LEFT)** | **+0.338 (drifts RIGHT)** |
| +0.75 | -0.128 (**wrong direction**) | +0.698 (ratio 0.93) |
| +1.50 | +0.128 (ratio 0.09) | +0.688 (ratio 0.46) |

Forward speed survives turning (0.46-0.53 m/s throughout) and it never falls, so
the gait is robust. The yaw is not. Under a ZERO yaw command it curves at
0.28-0.34 rad/s — commanded to run straight, it runs in a circle.

**This is my design error, not the robot's limit.** I narrowed `ang_vel_z` to
+/-0.8 AND cut the turn-in-place bucket from 0.15 to 0.05, reasoning that a
sprint task should spend its experience going forward. AGENTS.md warns about
exactly this: "Rare-but-important command regions need explicit buckets —
independent uniform sampling made spinning ~2% of experience and it never
trained." I starved turning and got a policy that cannot turn.

The tell that it is undertrained rather than structurally biased: the drift
FLIPS SIGN between checkpoints (left at 750, right at 1500). A real asymmetry
would not change direction.

Config corrected to the walking recipe's own values (ang +/-1.0 — its comment
calls this "the big change, it makes turning learnable" — and the 0.15 bucket).
The in-flight run keeps the old baked config, so the plan is to fine-tune from
its final checkpoint with the corrected coverage once it finishes, which
`--resume-run` supports for ~$2 rather than a $9 retrain.

### 01:55 — measured the tricks; two were cheating, one is genuinely good

Scored every trick from a real 12 s rollout (`eval_sprint_jump.py trick`). The
headline percentages looked excellent and two of the three were lies.

| trick | single-sup | **longest hold** | rev / 12 s | yaw rate | verdict |
|---|---|---|---|---|---|
| one_leg_stand | 88% | **0.21 s** | 2.6 | 1.35 | hopping pirouette |
| spin_one_leg | 89% | **0.23 s** | 8.2 | 4.29 | hopping pirouette |
| **spin_two_leg** | 6% | — | **12.5** | **6.54** | **genuine, 94% double-support, 2° tilt** |

**spin_two_leg works.** 12.5 revolutions in 12 seconds at 6.54 rad/s, both feet
down 94% of the time, max trunk tilt 2°, never falls. Keeping that run.

**The one-leg tricks were cheating, and my reasoning was wrong.** The module
docstring claimed that requiring "exactly one foot" ruled out the alternating
cheat. It does not: swapping feet rapidly satisfies exactly-one at EVERY
instant. Both policies scored ~88% single support with a longest unbroken hold
of ~0.2 s — they were hopping foot to foot while spinning, and the metric I
chose could not tell that apart from balancing.

The `longest unbroken` column is what caught it, and it only exists because the
eval was written to measure the behaviour rather than re-read the reward.

Fix: `sustained_single_support` tracks WHICH foot bears weight and how long it
has been the only one. Any switch, any double-support step, any airborne step
resets the clock, so alternation pays ~0 and only genuinely holding accumulates
(ramped to full credit at a 2 s hold, so there is gradient the whole way up).
Tests now assert alternation cannot accumulate and that a longer hold pays more.

Both one-leg runs stopped and relaunched with the fix; spin_two_leg left alone.

### 01:52 — jump v3: the episode-max objective had a deployability flaw

Tracing trunk height through a rollout: `12.5, 12.4, 12.4, 12.3, 12.3, 12.3 cm`
with a max of 13.4. The policy jumps ONCE out of the reset transient and then
stands still for the rest of the episode.

That is my reward working exactly as written. The record term pays only for
BEATING the record, so a second jump of the same height pays nothing while still
costing impact and action-rate — standing still is optimal. I asked for "the
episode's best jump" and got precisely that.

It also explains a contradiction: the training log said 3.5 cm while the harness
said 0.0 cm. The harness settled for 1.5 s before measuring, which skipped the
only jump and scored the stance afterwards. Both numbers were honest; the
harness was answering a different question. It no longer settles for the jump
task, and it now reports the best jump after t=3 s so a one-shot is labelled as
one.

Honest current jump result: **0.9 cm, one-shot**. Not good enough, and not
deployable — on the robot you press a button and expect a jump, not a reboot.

Run 3 adds `jump_flight_payout`: peak^2/ref paid ONCE per flight at the landing
edge. Per-flight restores the reason to keep jumping; squaring keeps height the
point (one 4 cm jump beats two 2 cm ones); paying on the landing edge means hang
time cannot be farmed. The record term stays at reduced weight to keep driving
the ceiling.

### 02:12 — spin_two_leg finished: a real trick, with an honest limitation

Final checkpoint (iter 1250) vs mid-run (iter 750), 12 s rollouts:

| checkpoint | yaw rate | revolutions | double-support | max tilt |
|---|---|---|---|---|
| **iter 750** | **6.54 rad/s** | **12.5** | 94% | 2° |
| iter 1250 (final) | 4.87 rad/s | 9.3 | 100% | 0° |

The LAST checkpoint is not the best one — it is slower but cleaner. Exactly what
AGENTS.md warns about, and the reason every task here gets its checkpoints
compared rather than its final one shipped. For a trick, iter 750 is the
deliverable: 12.5 revolutions without ever leaving the ground or exceeding 2° of
tilt.

**The limitation, measured rather than assumed.** Sweeping the yaw command:

| cmd | iter 750 achieved | iter 1250 achieved |
|---|---|---|
| 2.0 | 6.47 | 4.79 |
| 4.0 | 6.59 | 4.89 |
| 6.0 | 6.77 | 4.81 |

The achieved rate is flat across a 3x range of command — these are fixed-rate
spinners, not commandable ones. The cause is a weighting mistake I can name:
`spin_rate` pays for |omega| up to a cap regardless of command, and at weight
3.0 it out-competed `track_angular_velocity` at the same weight. Asked for 2.0
the policy does 6.5, because exceeding the cap costs nothing while the rate
bonus is already saturated.

Not retraining it — a spinning trick that spins is the ask, the budget is nearly
spent, and the fix is a weight change whose correctness I would not be able to
verify before morning. Recorded here so the next run starts from the diagnosis
rather than rediscovering it.

### 02:12 — sprint peaked and was declining; stopped it and recovered budget

| iter | best speed |
|---|---|
| 750 | 0.617 |
| **1500** | **0.758** |
| 2000 | 0.679 |
| 2750 | 0.549 |

Speed peaked at 1500 then fell 28% as the ramp climbed past the robot, with mean
reward falling alongside (122.9 -> 116.7 -> 114.5). The walking recipe recorded
this exact decline when its own ramp "outpaced the robot's capability"; I found
the same wall from the other side.

Stopped the run — it had 2.7 h of cap left and was getting worse — which
recovered ~$5.8. The ramp is now truncated at 0.75 (just under the measured
ceiling) and compressed, since run 1 covered 0.45 -> peak inside 1500
iterations. Relaunched as a $3 fine-tune resuming from `model_1500.pt` with the
corrected turn coverage, which is the cheap way to fix the steering defect
without a $9 retrain.

Worth stating plainly: reward alone would NOT have caught this. It moved 7%
while actual top speed moved 28%. Only per-checkpoint measurement showed it.

### 02:21 — one-leg stand: the reward is right, the policy does not transfer

The sustained-support fix worked in training — `Episode_Reward/support` reached
5.71 of a 6.0 weight, i.e. ~95% of a full 2 s hold, against the alternating
version's 0.21 s. But the exported policy, evaluated at both iteration 500 and
750, stands squarely on TWO feet: 100% double support, per-foot loads 3.5 N and
3.5 N against a 7.85 N body weight, 0.04 rad/s, 1° tilt. It never lifts a foot.

Ruled out, in order:

1. **Reward bug** — instrumented the real training env: `found` is shape (N, 2),
   per-foot detection is correct, and a zero-action policy scores exactly 0.0.
   Unit tests already cover alternation and holding.
2. **Eval too strict** — switched foot contact from geometric touch to LOAD
   (>= 0.4 N, ~5% of body weight), matching the training sensor's netforce
   reduction. No change: still 100% double support.
3. **Harness cannot see single support at all** — it can. `spin_one_leg` run 1
   measured 89% single support through the same code path.

What is left is a sim-to-sim gap in the ACTUATOR. Training uses the BAM voltage
model; `infer_policy` — my eval, and the deployment rehearsal path — uses
position servos with the current limit applied as a force clip. Its own comment
says as much. A policy that unloads a foot under BAM does not necessarily do so
under a clipped position servo, and the eval is the deployment-realistic one, so
the honest reading is that this policy does not transfer rather than that it
works.

Not chasing it further tonight: it needs a BAM-faithful eval path, which is a
bigger change than the remaining budget and hours justify, and three other
policies do transfer. Recorded so the next session starts from the diagnosis.

### 02:52 — the headstand works, and it was the one I bet against

I wrote in the task docstring that this was "the least likely of the tricks to
succeed". Measured at iteration 750:

| metric | value |
|---|---|
| inversion (feet - head) | **+0.123 m** (standing is -0.213) |
| head height | **0.1 cm** — on the floor |
| time inverted | 97% of the rollout |
| **longest continuous hold** | **11.68 s of 12 s** |

A held headstand, not a moment of a tumble. Three choices did the work, and all
three came from mistakes made earlier tonight:

1. **The all-collisions robot.** `robot_walk.xml` has two colliding geoms, both
   feet — the head would pass through the floor. One-line entity swap.
2. **Measuring feet-above-head, not tilt.** A robot on its back is "inverted" by
   any tilt measure. The geometric shape is what distinguishes a headstand, and
   verifying the two endpoints against the model (-0.213 standing, ~+0.17
   inverted) before training meant the gradient was known to be dense rather
   than hoped to be.
3. **Removing `fell_over`.** It terminates on the trunk tilt a headstand
   requires — left in, it would have ended the episode at the moment of success.
   And `upright` went to weight 0 because it rewards the opposite of the goal.

Point 3 is the one worth keeping: the inherited stack was actively hostile to
this task, and none of that shows up as an error. It shows up as a policy that
never learns.

### 02:50 — jump v3 bought height with stability, so v4 buys it back

All v3 checkpoints reach ~2.0-2.9 cm and every one ends fallen (trunk tilt
122-177°), with flight counts climbing 23 -> 48 -> 100. That is flailing, and it
is what a SQUARED per-flight payout pays for: aggression is profitable, so the
counterweight has to rise with it.

v4 raises `upright` 2.0 -> 4.0 (above the walking recipe's own value, rather
than being turned down with the terms that fight a crouch) and the landing
impact cost 0.002 -> 0.01. A test now asserts jump's upright weight EXCEEDS
walking's, so the next person to "turn down what fights a crouch" cannot quietly
include the term holding it upright.

Also worth recording: the walk model has no trunk collision, so a fallen robot's
feet simply leave the ground and the eval reads 91% "airborne". The number is
real; it means falling, not flying.

### 03:10 — sprint fine-tune landed: steering bought for 21% of the speed

The fine-tune recovered as it trained (0.516 at +1000 iterations, 0.600 at
+2250) and the steering genuinely improved:

| | run 1, iter 1500 | run 2, iter 3750 |
|---|---|---|
| best speed | **0.758 m/s** | 0.600 m/s |
| tracking ratio at cmd 0.6 | 0.48 | **0.89** |
| drift at ZERO yaw command | +0.338 rad/s | **+0.141** |
| yaw at cmd -1.0 | -0.281 (ratio 0.19) | -0.317 (ratio 0.32) |
| yaw at cmd +1.0 | +0.688 | +0.490 |

Zero-command drift halved and both directions now respond in the right
direction, which run 1 could not do. It cost 21% of top speed. That is a real
trade, not a free fix, and both policies are kept because which one is "best"
depends on whether you want a straight-line number or a robot you can drive.

One more signal in favour of the slower one: the exported action ranges are
[-5.42, 1.56] for the fast policy and [-0.65, 0.56] for the steerable one. The
fast policy is commanding far more extreme joint targets, which on real XL330s
is a sim2real risk the speed number does not show.

Deployment path verified end to end for every shipped policy: Modal -> ONNX ->
`obs[1,61] -> actions[1,14]` -> driving in `sim.sh`, which is the same
`infer_policy` rehearsal the runtime uses.

### 03:35 — jump: four designs, one consistent trade, no clean win

All runs finished. Every jump design lands on the same trade-off and none escapes
it:

| version | objective | height | stability |
|---|---|---|---|
| v2 | episode-max record | 0.9 cm | **upright**, one jump per episode |
| v3 | + per-flight peak^2 | 2.9 cm | falls (tilt 122-177°) |
| v4 | + upright 4.0, impact 0.01 | 2.6 cm | falls (tilt 104-179°) |

v4's extra upright pressure did not fix it: 1.8 -> 2.3 -> 2.6 cm across its
checkpoints with tilt rising 104° -> 120° -> 179°. Raising the counterweight
bought height, not stability.

The reward breakdown says why the tax cannot win: `jump_payout` +8.08 against
`action_rate_l2` **-7.04**, with episodes ending at 439 of 1000 steps. The raw
action rate is enormous — the policy is jittering violently, paying almost all
of its winnings back, and still coming out ahead. Any penalty large enough to
stop that would also make not-jumping the best move.

**The most likely root cause, stated as a hypothesis because I could not test
it:** the jump inherits the velocity recipe's `robot_walk.xml`, which has two
colliding geoms and both are feet. There is no trunk or head contact, so a bad
landing has nothing to catch it — the robot tips and keeps going, which is also
why the eval reads 91% "airborne" for a fallen robot. Every task in this repo
that involves meeting the ground (standup, roulade, sitstand, ground-pick) uses
`robot_allcollisions.xml` instead. The headstand needed that model and got it;
the jump needed it too and I did not notice until the budget was gone.

Both ends are shipped and labelled: `jump-highest-unstable.onnx` (2.6 cm, falls)
and `jump-stable-shallow.onnx` (0.9 cm, upright). Neither is good. The next run
should start by swapping the robot model, which is one line.
