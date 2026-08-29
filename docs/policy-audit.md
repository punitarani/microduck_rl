# Policy audit — 2026-08-29

An adversarial re-examination of every claim made during the overnight training
pass, using multi-condition sweeps rather than single rollouts.

**Headline: two of the four claimed successes did not survive.** The headstand
was a faceplant that satisfied a badly-chosen metric, and the "one-leg stand
doesn't transfer" diagnosis was itself wrong — the policy never learned the
skill under *either* actuator model. Both were caught by looking at behaviour
rather than at rewards, and the headstand specifically by watching the video.

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
| 4 | spin-two-leg is genuine: 12.5 rev, never falls | **verified** | 91% no-fall (CI 85-94%), 95% rate >= 3 rad/s; **8/8 in every in-training condition** |
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

### spin-two-leg — **passes, with one hardware caveat**

91% no-fall, 95% rate, 8/8 in every in-training condition, and robust to battery
sag where the sprint is not. **But 36.8% joint saturation**: over a third of
control steps have a joint within 5% of its mechanical limit. AGENTS.md warns
about exactly this, and a saturated joint has no authority left to react. Not
disqualifying in sim; it is the first thing to check on hardware.

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
