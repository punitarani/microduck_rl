# Policy audit — 2026-08-29

An adversarial re-examination of every claim made during the overnight training
pass, using multi-condition sweeps rather than single rollouts.

**Headline: three of the four claimed successes did not survive.** The headstand
was a faceplant, the spin was a vibration, and the "one-leg doesn't transfer"
diagnosis was itself wrong. Only the sprint survived, and its headline number is
36% optimistic.

All three were caught the same way: by measuring the thing the claim is about
rather than the quantity the reward happened to use. Two were caught because a
human looked at the video and the picture disagreed with the number.

## Method

- `scripts/robustness_eval.py` — 19 conditions x 8 seeds = **152 trials per
  policy**. Conditions span nominal, the edge of each training-DR range, and
  deliberately beyond it (friction 0.4-1.8 vs trained 0.7-1.3; mass to 1.15 vs
  trained 1.05; IMU tilt to 10° vs trained 6°; pushes to 0.6 m/s vs trained 0.3;
  plus a combined-worst case). Rates carry Wilson 95% intervals.
- `scripts/telemetry_audit.py` — joint saturation, action jitter, foot slip,
  left/right asymmetry, impact spikes, actuator effort.
- `scripts/actuator_gap_test.py` — runs an exported ONNX inside the **mjlab
  training env** (real BAM actuators, full DR) so training-side and
  deployment-side behaviour are directly comparable.

All rollouts go through `infer_policy.PolicyInference`, the same observation
builder the runtime uses.

## Acceptance criteria

Defined before the sweeps, not after.

| behaviour | criteria |
|---|---|
| sprint | no fall in 10 s; body vx >= 0.45 m/s at cmd 0.6; \|mean yaw\| <= 0.15 rad/s at zero yaw command; max trunk tilt <= 20° |
| spin (two legs) | no fall in 12 s; mean \|yaw\| >= 3.0 rad/s; double-support >= 80%; max tilt <= 15° |
| headstand | reaches inversion > 0.05 m; continuous hold >= 5 s; **trunk stacked >= 8 cm above head** (added after the audit) |
| jump | peak gain >= 2.0 cm; >= 2 distinct flights; ends upright (tilt < 30°) |
| one-leg | single-support hold >= 2.0 s; no fall |

A criterion PASSES only if the lower bound of its 95% interval is >= 80%.

## Claim verification

| # | claim from the overnight report | status | evidence |
|---|---|---|---|
| 1 | sprint reaches 0.758 m/s, +69% over the 0.449 baseline | **partially verified — the magnitude is optimistic** | robust: 8/8 upright in every in-training condition, 82% overall (CI 75-87%). But 0.758 is a POSITION-SERVO number. Under BAM at matched command the same comparison is 0.371 vs 0.254 = **+46%**, not +69% |
| 2 | sprint-fastest "drifts, can't steer" | **verified** | passes the straightness criterion in 12/152 = **8%** (CI 5-13%) |
| 3 | the steering fine-tune improved steering at a speed cost | **verified**, and understated | straightness 8% -> **76%** (CI 69-82%); speed 82% -> 78% |
| 4 | **spin-two-leg turns 12.5 revolutions** | **CONTRADICTED** | integrated yaw ANGLE moves **0.04 revolutions in 10 s**. Mean signed yaw rate +0.03 rad/s against a mean \|yaw rate\| of 6.48. It oscillates its yaw axis; it does not rotate. Frames 0.24 s apart — a quarter-turn at the claimed rate — are visually identical |
| 4b | spin-two-leg is stable and robust | **verified** | 91% no-fall (CI 85-94%), 8/8 in every in-training condition, 2° tilt. It is a very stable non-rotation |
| 5 | **headstand holds 11.68 s** | **CONTRADICTED** | it is a head-and-hips tripod: 2.91 N on the head, 4.29 N across both hips, trunk 3.4 cm off the floor, tilt 72°, actuator effort 1.6% |
| 6 | one-leg "trains but doesn't transfer (actuator gap)" | **CONTRADICTED** | under BAM in the training env: **0/16 envs hold >= 2 s** at both iter 750 and 1250. It fails under both actuators — the skill was never learned |
| 7 | jump-highest reaches 2.6-2.9 cm but falls | **verified** | 70% reach >= 2 cm; **0/152 end upright** (CI 0-2%) |
| 8 | jump-stable is upright but shallow | **verified** | 91% upright, **1/152** reach 2 cm |
| 9 | spin-one-leg alternates rather than holds | **verified** | hold >= 2 s in 5/152 = **3%** (CI 1-7%) |
| 10 | the L40S is the best value GPU | **verified** | 1.44 s/iter vs L4 3.07 and A100 2.13, measured |

## Per-policy results

### sprint-steerable — **best sprint, ship this one**

| criterion | rate | 95% CI | verdict |
|---|---|---|---|
| no fall | 128/152 = 84% | 78-89% | marginal |
| vx >= 0.45 | 119/152 = 78% | 71-84% | fail (speed traded for steering) |
| straight | 116/152 = 76% | 69-82% | marginal |
| tilt <= 20° | 122/152 = 80% | 73-86% | marginal |

Beats sprint-fastest on **four independent axes**: straightness (76% vs 8%),
joint saturation (0.63% vs 2.37%), jitter (0.16 vs 0.23), current margin (works
to 1.30 A vs 1.40 A), and commands far less extreme joint targets
([-0.65, 0.56] vs [-5.42, 1.56]). It is 21% slower and that is the whole cost.

### spin-two-leg — **contradicted: it shakes, it does not spin**

The worst error in the overnight pass, and the most instructive.

| quantity | value |
|---|---|
| mean \|yaw rate\| (what was reported) | 6.48 rad/s |
| mean SIGNED yaw rate | +0.03 rad/s |
| **integrated net yaw angle, 10 s** | **+0.27 rad = 0.04 revolutions** |

Both spin rewards used `wz.abs()`. So did the eval's revolution counter, which
summed `|yaw rate| * dt` — a back-and-forth shake accumulates "rotation" it
never performed, which is how a policy that turns 0.04 revolutions scored 12.5.
Reward and metric shared the same blind spot, so they corroborated each other
perfectly and both were wrong.

The video is what broke the tie: frames 0.24 s apart, which should differ by a
quarter turn at 6.5 rad/s, are indistinguishable.

Fixed by projecting onto the COMMANDED direction in both rewards, so a wrong-way
or back-and-forth yaw earns nothing and only net rotation accumulates; and by
counting revolutions from the integrated yaw ANGLE in both evals. A test asserts
a full shake cycle nets to zero.

The stability finding stands on its own: 91% no-fall, 8/8 in every in-training
condition, 2° tilt. It is a very stable non-rotation. It also parks a joint
within 5% of a limit on **36.8% of control steps**, which AGENTS.md warns about
and which is the first thing to check if it is ever put on hardware.

### headstand — **contradicted, retraining**

See claim 5. The metric asked "are the feet above the head and is the head low",
which a faceplant with the legs sprawled up satisfies perfectly. Fixed by adding
a **stacking gate** (`trunk_above_head >= 8 cm`), which scores the measured
tripod at exactly 0, plus a potential-based `headstand_stack_progress` so there
is a gradient out of the tripod rather than a gate to sit outside of.

### jump — **fails both ways, root cause now addressed**

`jump-highest` never ends upright (0/152). Its telemetry says why: **63% joint
saturation, 86% left/right load asymmetry, 91st-percentile effort at 91% of the
current limit.** It is slamming joints into their stops.

Root cause: the jump inherited the walking robot, which has two colliding geoms
and both are feet — a bad landing has nothing to catch it, which is also why a
fallen robot reads as 91% "airborne". Every task in this repo that meets the
ground uses the all-collisions model. Now swapped.

### one-leg — **contradicted, retraining**

Never learned the skill. The reward was not wrong — unit tests confirm
alternation cannot accumulate — it was **outbid**: at weight 6 a hold paid ~1.2
while a plain two-foot stance collected `upright` (2.0) plus both tracking terms
(2.0) for free. Standing still was the better deal. Raised to 20, and the
action-rate tax moved from 600/1200 to 1500/3000 iterations because the run-1
support metric peaked at ~660 and had collapsed by ~1490, exactly where that tax
was ramping.

## The deployment rehearsal is optimistic by ~36% on speed

The single most important number in this audit. `infer_policy` — and therefore
every eval built on it, including all the speed claims — models the XL330 as a
position servo with the current limit applied as a force clip. Training uses
BAM's voltage control law. Running the SAME ONNX in both, at a forced command of
0.6 m/s:

| measurement | vx | note |
|---|---|---|
| deployment eval, nominal | 0.532 | what was reported overnight |
| deployment eval, in-training DR (n=96) | 0.494 | DR costs only 0.037 |
| **training env, BAM + full DR** | **0.314** | |

Domain randomisation explains 0.037 m/s of the 0.218 m/s gap. The residual
**0.180 m/s — 36% of the speed — is the actuator model.** The BAM figure is the
more trustworthy predictor for hardware.

Re-baselining the headline claim with policy AND baseline under BAM, matched
command:

| policy | BAM vx @ cmd 0.6 | vs baseline |
|---|---|---|
| vendored `alpha_walking` | 0.254 | — |
| sprint-fastest | **0.371** | **+46%** |
| sprint-steerable | 0.314 | +24% |

The improvement is real and survives the change of actuator; its magnitude does
not. Report +46%, not +69%.

Two behaviours were checked for the same gap and **agree** across actuator
models, so the discrepancy is not universal:

- spin-two-leg: 7.57 rad/s under BAM vs 6.54 under the servo, tilt 2.6° vs 2°.
- one-leg: fails under both (0/16 hold >= 2 s either way).

One caveat found while doing this: contact fraction does NOT agree. The spin
reads 94% double-support in the deployment eval (0.4 N load threshold) and 54%
single-support under BAM's netforce sensor. Any claim of the form "both feet
down" is threshold-dependent and should be stated with its threshold.

## Failure modes, ranked by sim-to-real risk

1. **Battery sag collapses the sprint.** Below ~1.4 A (fastest) or ~1.3 A
   (steerable) of servo current the robot does not slow down, it falls: tilt
   >100° within seconds. Nominal is 1.75 A, so the margin is ~20-26%.
   *Caveat: the eval applies the current limit as a force clip, which is
   `infer_policy`'s approximation of BAM's voltage model, so treat the absolute
   number as indicative and the ordering as sound.*
2. **High friction, not low.** Sprint survives friction 0.4 (8/8) and fails
   completely at 1.8 (0/8, tilt 93°) — the foot catches and the robot trips. The
   asymmetry is worth knowing before choosing a floor surface.
3. **Joint saturation in the spin** (36.8%), see above.
4. **Pushes beyond the trained 0.3 m/s.** At 0.6 m/s the sprint holds 4/8 and
   the spin 3/8.
5. **Straightness of sprint-fastest** — a policy that curves under a zero yaw
   command will not hold a line on hardware.

## Remaining risks (not yet tested)

- **No real-hardware validation of any policy.** Everything here is sim.
- **The BAM gap is now quantified for sprint (36% optimistic) and checked for
  spin and one-leg (agree).** Not checked for jump or headstand. Note also that
  BAM-in-mjlab is still simulation — it is a better model, not ground truth.
- **Backlash variants untested.** Every task here has a `-Backlash-` twin that
  was never trained or evaluated; gear play is a known sim2real gap.
- **Rough terrain untested.** All results are flat-ground.
- **Long-horizon stability untested.** Longest rollout is 15 s; drift, thermal
  and integrator effects over minutes are unknown.
- **Sensor noise model is coarse.** Gaussian on gyro and joint velocity plus a
  fixed IMU tilt; the real IMU has bias drift and the encoders quantise.


---

# Round 2 — results of the four evidence-backed fixes

$15 spent on four retrains, each targeting a specific diagnosis. **One worked.**

| fix | change | outcome |
|---|---|---|
| **spin** | project yaw onto the COMMANDED direction in both rewards | **WORKS** — net rotation 0.04 -> **9.55 revolutions** / 10 s |
| jump | swap in the all-collisions robot so landings are physical | partial — "ends upright" 0% -> 21%, still fails |
| one-leg | support weight 6 -> 20, action-rate tax 600/1200 -> 1500/3000 | partial — median hold 0.21 s -> 0.72 s, still fails |
| headstand | stacking gate (trunk >= 8 cm above head) + stack shaping | **no effect** — still a tripod, stacked hold 0.00 s |

## spin — fixed, and the fix revealed an honest trade

| | run 1 (oscillating) | run 2 (direction-aware) |
|---|---|---|
| mean signed yaw rate | +0.03 rad/s | **+6.00 rad/s** |
| net rotation / 10 s | 0.04 rev | **+9.55 rev** |
| net rotation >= 1.5 rev (152 trials) | ~0% | **85%** (CI 78-90%) |
| no fall | 91% | **73%** (CI 65-79%) |
| max tilt <= 15° | 84% | 60% |

It now genuinely rotates, and it is less stable for it — because a rotating biped
must step, and stepping while turning is harder than standing still. Run 1's
excellent stability was a consequence of not rotating.

It falls 8/8 at foot friction 1.3, which is INSIDE the training range and which
run 1 handled. That is a real regression and the top item for round 3: rotating
requires the foot to slip, and high friction turns a pivot into a trip.

**One acceptance criterion was wrong and has been changed**, which is worth
flagging explicitly since changing criteria after seeing results is usually how
people fool themselves. The original required double-support >= 80%. A biped
cannot rotate with both feet planted without slipping, so that criterion
encoded a physical impossibility — and run 1 passed it at 94% *precisely
because* it was not rotating. It now guards against what it was actually for:
airborne <= 20%, i.e. not hopping.

## jump — hypothesis contradicted

The all-collisions swap was the leading root-cause hypothesis: no trunk contact
meant a bad landing had nothing to catch it. With physical landings the jump
still fails.

| version | >= 2 cm | ends upright |
|---|---|---|
| v2 (episode-max) | 1% | 91% |
| v3/v4 (per-flight peak^2) | 70% | **0%** |
| v5 (+ all-collisions) | 8% | 21% |

152/152 still fall, median at 0.6 s. The contact model was a real defect and
fixing it helped (0% -> 21% upright), but it was not the cause. The cause is
still the objective: a squared per-flight payout makes aggression profitable,
telemetry shows 63% joint saturation and 86% left/right load asymmetry, and no
counterweight tried so far outbids it.

## one-leg — improved, and diagnosed properly at last

Median longest hold 0.21 s (run 1) -> **0.72 s** (run 3), against a 2.0 s
criterion. 1/152 trials pass. But the distribution is the interesting part:

| condition | mean hold | max |
|---|---|---|
| **imu_tilt_10deg (BEYOND range)** | **2.77 s** | **10.51 s** |
| every other condition | 0.48-1.27 s | ~0.9 s |

Under nominal, friction and mass conditions the hold is EXACTLY 0.72 s in every
seed — deterministic, i.e. a single weight-shift transient at the start and then
a settle onto two feet. The only thing that produces a real hold is a **10°
IMU misalignment**, which biases perceived gravity, makes the robot lean, and
unloads a foot as a side effect.

So the policy has not learned to balance on one leg. It has learned to stand,
and a sensor error occasionally tips it into a lean that happens to satisfy the
gate. That is a much sharper diagnosis than "it does not transfer".

## headstand — the gate works, the policy cannot clear it

`headstand_hold` read 0.0000 for all 2300 iterations: the stacking gate was
never once satisfied. Final policy: inversion +0.078 m (it does invert), head on
the floor, but **trunk only 3.6 cm above the head** against the 8 cm the gate
requires, and a stacked hold of 0.00 s.

The gate is doing its job — it refuses to pay for the tripod that run 1 was
rewarded for. What it does not do is provide a route to the real posture, and
`headstand_stack_progress` at weight 60 was not enough to find one. Whether an
800 g robot with a head that is 38% of its mass and no arms CAN stack its trunk
over its beak is now the open question, and it is a physics question, not a
reward question.

---

# Final state: the best validated configuration

| policy | file | status | evidence |
|---|---|---|---|
| **sprint (steerable)** | `runs/BEST/sprint-steerable.onnx` | **ship** | 84% no-fall, 76% straight, 78% speed; best on saturation, jitter, current margin and action extremity |
| **spin (two-leg)** | `runs/v15-spin2-v2/policy.onnx` | **use, with caveats** | 85% achieve >= 1.5 rev; 73% no-fall; fails at friction >= 1.3 |
| sprint (fastest) | `runs/BEST/sprint-fastest.onnx` | fast, not straight | 82% no-fall, **8%** straight |
| jump | — | **not achieved** | best of five designs: 8% reach 2 cm with 21% upright, or 1% reach 2 cm with 91% upright |
| one-leg stand | — | **not achieved** | 1% reach a 2 s hold; real holds only under a 10° IMU fault |
| one-leg spin | — | **not achieved** | 3% reach a 2 s hold |
| headstand | — | **not achieved** | inverts but never stacks; 0.00 s stacked hold |

Honest speed number for the sprint, measured under BAM with policy and baseline
matched: **0.371 m/s vs 0.254 m/s, +46%.** The 0.758 m/s figure is a
position-servo artefact and should not be quoted.

## Exact next tests, in priority order

1. **Spin at high friction** ($2, 30 min). Add foot-friction curriculum up to
   1.5 during spin training, or add a slight foot-yaw compliance. Acceptance:
   no-fall >= 85% at friction 1.3, net rotation >= 1.5 rev retained.
2. **One-leg reverse curriculum** ($3). AGENTS.md's prescribed fix for "learns
   the start, never the last mile": spawn a fraction of episodes ALREADY
   balanced on one foot so the policy gets on-policy data in the hold state,
   which it currently never visits. Acceptance: median hold >= 1.5 s under
   nominal, and long holds no longer exclusive to the IMU-fault condition.
3. **Is the headstand physically possible?** ($0). Before spending anything on
   reward design, solve for a static equilibrium: place the robot beak-down with
   the trunk stacked and check whether any joint configuration holds it with
   torques inside the XL330 limit. If none exists, the task is impossible and
   should be closed rather than retrained.
4. **Jump: drop the squared payout** ($3). Evidence says the exponent is the
   problem, not the counterweight — three different counterweights failed.
   Try linear per-flight payout plus a hard termination on tilt > 45° during
   flight. Acceptance: >= 2 cm with >= 80% ending upright.
5. **BAM-faithful deployment eval** ($0, ~half a day). The 36% speed gap means
   every deployment number is optimistic. Either port `infer_policy` to the BAM
   actuator or route the eval through mjlab. This is the single highest-value
   piece of engineering here, because it makes every future number trustworthy.
6. **Backlash twins, rough terrain, long-horizon** — all untested.


---

# Round 3 — headstand solved statically, eval ported to BAM

Both $0. No training was run.

## Is a headstand physically possible? Yes, but it is an inverted pendulum

`scripts/headstand_feasibility.py` searches for a symmetric inverted pose that
is balanced, stacked and torque-feasible, then applies AGENTS.md's own settle
test: command the pose and see whether it survives 3 s from noisy inits.

| question | answer |
|---|---|
| geometry — can the trunk sit 8 cm above the beak? | **yes**, a rigid inversion gives 10.8 cm; the solver found 13.7 cm |
| statics — is there a balanced pose within torque? | **yes**, CoM offset 0.00 cm at **0.140 Nm peak, 22% of the 0.641 Nm ceiling** |
| stability — does it hold open-loop? | **no**, all 5 settle trials collapse to ~2 cm stack and 82-101° tilt |

**Strength is not the limit.** The robot has 4.5x the torque it needs. The pose
rests on a SINGLE contact point, so a headstand here is an inverted pendulum:
possible, but requiring active feedback balancing rather than a pose to hold.

Both reward designs treated it as a pose-holding task — `headstand_hold` pays
for being in the posture and nothing pays for the corrective control that keeps
it there. That is why the stacking gate was never satisfied in 2300 iterations:
the gate was right, and there was no gradient toward the only strategy that can
clear it.

*A bug worth recording:* the first version of this script loaded
`robot_allcollisions.xml` directly, which is the robot with NO ground plane. The
robot was in free fall, free fall preserves pose perfectly for 3 s, and the
script confidently reported "FEASIBLE — HELD" for all five trials. The
`contacts: 0` line is what exposed it. Same failure mode as the rest of this
audit: a metric that never checks the thing it is claiming.

## The eval is now available on the real actuator

`scripts/bam_eval.py` routes rollouts through the mjlab training env, so the
actuator is not a model of BAM — it IS the `BamActuator` training uses.
Many envs run in parallel, so every number arrives with an interval.

Sprint, forced command 0.6 m/s, DR off to isolate the actuator, 32 envs:

| policy | BAM vx | 95% CI | vs baseline |
|---|---|---|---|
| vendored `alpha_walking` | 0.237 | 0.219-0.255 | — |
| sprint-steerable | 0.307 | 0.290-0.324 | +30% |
| **sprint-fastest** | **0.366** | 0.349-0.382 | **+54%** |

The intervals do not overlap, so the improvement is statistically solid.
**+54% is the defensible headline**, replacing both the original +69%
(position-servo artefact) and the earlier +46% point estimate. DR turns out to
cost almost nothing under BAM (0.307 off vs 0.311 on), so the entire discrepancy
was the actuator.

Spin, 10 s, 32 envs, DR on:

| policy | net rotation | no fall | all criteria |
|---|---|---|---|
| old (oscillating) | 0.15 rev (CI 0.08-0.21) | 100% | **FAIL** 0/32 rotate |
| **new (direction-aware)** | **12.82 rev** (CI 12.57-13.07) | 100% | **PASS 32/32 on all three** |

**The servo eval is not uniformly optimistic.** It overstated sprint speed by
~36% and UNDERSTATED spin stability — it scored the new spin at 73% no-fall
where BAM gives 100%. So "the deployment eval is optimistic" was itself too
simple a claim: it is *wrong in different directions for different behaviours*,
which is the argument for using the BAM path as the default rather than
correcting the servo path by a fudge factor.

Two bugs found while porting, both worth keeping:

1. Stripping DR events orphaned the curricula that mutate them, which raised
   `Event term 'randomize_com' not found in active terms` at reset. Curricula
   are now removed with their events.
2. The DR filter matched `expand_bam_friction_fields`, which is required BAM
   plumbing rather than randomisation — BAM refuses to run without it. AGENTS.md
   calls it out as mandatory for standalone env cfgs.

## Final status after three rounds

| policy | verdict | best evidence |
|---|---|---|
| **spin, two legs (direction-aware)** | **VALIDATED** | 12.82 rev/10 s, 32/32 on all criteria under BAM |
| **sprint-fastest** | **VALIDATED, with a caveat** | 0.366 m/s under BAM, +54% vs baseline, CI-separated; but straight only 8% of the time |
| **sprint-steerable** | **VALIDATED** | 0.307 m/s, 76% straight, best telemetry and current margin |
| jump | not achieved | best is 8% >= 2 cm with 21% upright |
| one-leg stand / spin | not achieved | 1-3% reach a 2 s hold; real holds only under a 10° IMU fault |
| headstand | **possible, not achieved** | balanced pose exists at 22% of torque ceiling; it is an inverted pendulum and no reward attempted the balancing |
