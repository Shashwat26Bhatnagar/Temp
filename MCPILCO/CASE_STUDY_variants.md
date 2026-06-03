# Case Study: When GFN-MC-PILCO Cost Functions Succeed or Fail

A systematic comparison of cost-function designs across two control tasks,
explaining mechanically — from the code — why each succeeds or fails.

---

## 0. The one-sentence finding

> **The cost function must match the *structure of the task*. A goal-reaching
> distributional cost (Variant C) solves cartpole but fails UR5; a fixed-metric
> per-step tracking cost (Variant H) solves UR5 but cannot even be applied to
> cartpole. Every cost that divides the error by a *learned/particle variance*
> introduces a cheat that decouples the cost from the task.**

---

## 1. Two task archetypes (this is the root of everything)

| | **Cartpole swing-up** | **UR5 circle tracking** |
|---|---|---|
| Task type | **Goal-reaching** | **Trajectory tracking** |
| Objective | reach ONE terminal state (θ=π) | follow a reference q_ref(t) at EVERY step |
| State dim | 4 | 12 |
| Reference at step k? | **none** — path is policy-dependent | **yes** — q_ref(t_k) is given by IK |
| Actuation | **underactuated** (1 force, 2 DOF) | **fully actuated** (6 torques, 6 joints) |
| Mid-trajectory | must go θ<0 to pump energy | next waypoint reachable in one step |
| Solutions | **bimodal** (θ→+π and θ→−π) | unimodal |

These differences decide which cost works. A cost that assumes a per-step
reference cannot be written for cartpole; a cost that only pins the endpoint
wastes the dense supervision available in UR5.

---

## 2. The variants under study (mechanics from the code)

### Variant C — distributional reverse-KL to a target
`policy_learning/variant_c_cost.py` (cartpole), `ur5_variant_c_cost.py` (UR5)

```
1. Collapse all P particles to ONE diagonal Gaussian (moment matching):
       μ_q, Σ_q = mean, var  over particles
2. KL( p_target || q_particles ),  reverse KL, closed form:
       L_k = ½ Σ_d [ σ²_p/Σ_q + (μ_q−μ_p)²/Σ_q − 1 + log(Σ_q/σ²_p) ]
                                        ▲
                              Σ_q (particle spread) in the DENOMINATOR
   cartpole: μ_p = FIXED terminal [0,0,π,0];   weighting = QUADRATIC
   UR5:      μ_p = q_ref(t_k) per step;          weighting = quadratic
```

### Variant H — fixed-metric per-particle tracking
`policy_learning/ur5_variant_h_cost.py` (lines 196–198)

```
L_k = (1/P) Σ_i  ½ Σ_d  (x^i_{k,d} − μ_ref_{k,d})² / σ²_d
                                                       ▲
                                       σ²_d is a FIXED CONSTANT
   μ_ref = get_mu_at_steps = interpolated IK reference q_ref(t_k)
   per-particle (no collapse);   weighting = UNIFORM
```
**Note:** Variant H never queries the GFN network. It uses `q_ref` as the
target and the GFN's σ only as a fixed metric. Mechanically it is **classic
MC-PILCO trajectory tracking** with a Mahalanobis weight.

### Variant K — live conditional GFN transition
`policy_learning/variant_k_cost.py` (current)

```
For each particle, query the GFN as a transition kernel from its own state:
       μ_GFN, σ²_GFN = predict_next_state(x_k, t=k/N_h)
       L_k = Σ_m w_m · KL( GFN(s_{k+1}) || GP(s_{k+1}) )
                                                ▲
                                  GP variance in the DENOMINATOR
   target = GFN denoising drift (NOT a physical trajectory)
   weighting = uniform;   per-particle;   truncated BPTT
```

---

## 3. CARTPOLE — controlled comparison: C (works) vs K (fails)

Both run on the identical cartpole, GP model, policy, and safety. Only the
cost differs.

| | Variant C ✅ | Variant K ✗ |
|---|---|---|
| Target | fixed terminal goal N([0,0,π,0]) | GFN conditional per step (denoising path) |
| Weighting | **quadratic** (end-loaded) | **uniform** (every step pinned) |
| What it demands | "**end up** at θ=π" | "**follow this path** at every step" |
| Denominator | Σ_q (particle spread) | GP variance per particle |

**Empirical:**
- **C:** swing-up solved, θ reaches and holds π (comparable to RBF baseline ~4 trials).
- **K:** trial 2 reaches π (overshoots), trial 3 rises but never reaches π,
  trials 4–5 oscillate around a mean position.

**Why C works on cartpole.** Quadratic weighting `w_k=(k/N_h)²` makes the
early/middle steps contribute almost nothing. So the policy is **free** to do
the energy-pumping swing (θ<0 first) — only the *terminal* cloud is required
to sit at π. This exactly matches the goal-reaching structure: specify the
goal, leave the path free.

**Why K fails on cartpole — three compounding reasons:**
1. **Path tracking forbids energy pumping.** Uniform weighting + a per-step
   target demands θ track the GFN's *monotone* denoising path (0→π). But
   cartpole MUST go θ<0 first. The cost punishes the only physical solution.
   (`diagnose_variant_k.py` plots this mismatch directly.)
2. **Wrong target.** The GFN conditional is a denoiser drift in diffusion
   space, not a physical cartpole trajectory. At t=0.3 it says "be at θ≈0.9";
   the policy needs to be going backward.
3. **Self-defeating confidence loop.** Reverse KL divides by GP variance.
   After trial 2 reaches π, the GP gets confident there (σ²_GP shrinks), the
   trace term σ²_GFN/σ²_GP explodes, and the gradient **ejects** the policy
   from the very region it just learned. → trial 3 can't return. This is the
   exact "trial 2 good, then degrades" signature.

---

## 4. UR5 — H (works) vs the distributional family (fail)

**Empirical (real-system mean RMS joint error, last rollout — the only
task-comparable metric):**

| Variant | Trials | Mean RMS joint err | Note |
|---|---|---|---|
| **H** (seed 2) | 8 | **1.031 rad** | best; most trials |
| C | 4 | 1.205 rad | competitive per-trial |
| H (seed 1) | 5 | 1.747 rad | |
| F | 5 | 1.710 rad | |
| I (JSD) | 5 | 2.356 rad | |
| G | 4 | 2.858 rad | worst |
| E | 3 | — | cost ~1e8, diverged |

**Honest caveat:** *every* UR5 variant is ~1 rad / ~1 m end-effector error in
absolute terms — none tracks the circle well. The dominant error is the
**zeros-start catch-up** (the arm flies in from θ=0, ~2.8 rad from q_ref[0]),
which inflates all of them. So "H works" means "H is marginally best and
stable," not "H tracks accurately."

**Why H is best.** Its mean term is `(x_i − μ_ref)² / σ²` with σ² a **fixed
constant**. The only way to lower it is to move each particle toward the
feasible reference. No cheat exists. Direct, robust tracking — the proven
classical recipe.

**Why the distributional family underperforms — two villains:**

### Villain 1 — variance-in-the-denominator ("wide-cloud cheat")
In C/F/G the mean term is `(μ_q − μ_ref)² / Σ_q` with Σ_q the **particle
spread**, which the policy can inflate. Two ways to lower the cost:
1. move μ_q toward μ_ref (what we want), or
2. **just make Σ_q bigger** — denominator grows, cost drops, *mean still wrong*.

The optimizer takes the cheap path: spray a wide diffuse cloud. **KL goes
down, tracking error does not.** (Documented as the "wide cloud" loophole in
Variant D's notes.) Variant H has no Σ_q in the denominator, so the cheat
cannot exist.

### Villain 2 — moment-matching collapse in 12D
`gaussian_moments_from_particles` crushes 50 particles in 12-D into one
diagonal Gaussian: it discards cross-joint correlations and per-particle
structure. μ_q can be "right on average" while no single particle tracks. H
scores every particle individually, so an off particle always pays.

---

## 5. The cross-task matrix (the key insight)

|                              | **Cartpole (goal-reaching)** | **UR5 (tracking)** |
|------------------------------|------------------------------|--------------------|
| **C** (distributional, end-target) | ✅ **works** — goal matches a terminal distribution; quadratic weighting leaves the path free | ✗ wide-cloud cheat exploitable at every step; moment-collapse in 12D |
| **H** (fixed-metric tracking) | ✗ **inapplicable** — there is no per-step reference to track; forcing one = an infeasible ramp = the Variant-K failure | ✅ **works** — feasible per-step reference + uncheatable fixed metric |
| **K** (live GFN conditional) | ✗ wrong target + path forbids pumping + confidence ejection | ✗ wrong target (denoising≠circle) + variance-in-denominator |

**Read the diagonal:** the cost that succeeds is the one whose *structure
matches the task's structure*.
- Cartpole is goal-reaching → a **terminal distributional** cost (C) fits.
- UR5 is tracking → a **per-step fixed-metric** cost (H) fits.
- Swapping them fails: C's per-step distributional matching cheats on UR5;
  H's tracking has nothing to track on cartpole.

**This answers "why doesn't H work on cartpole?"** — H *requires* a feasible
reference trajectory at every step. Cartpole swing-up has none (the path is
policy-dependent and involves energy pumping). The only reference you could
invent is a 0→π ramp (or the GFN denoising path), which is physically
infeasible — and tracking it is exactly what makes Variant K fail. So H is
not "bad at cartpole"; it is **structurally undefined** for a goal-reaching
task.

**And K fails everywhere** because it is a per-step distributional-matching
cost (tracking-style) pointed at a denoising-path target (neither physical
nor goal) with a variance-in-denominator metric — it inherits every villain
at once.

---

## 6. Root-cause taxonomy (which failure bites where)

| Failure mechanism | Cartpole | UR5 | Cause |
|---|---|---|---|
| Variance-in-denominator cheat | K | C, F, G, K | Σ_q or σ²_GP in the KL denominator |
| Moment-matching collapse | — | C, F, G | 50 particles → 1 diagonal Gaussian in 12-D |
| Multimodality / dead-zone | C-risk, K | — | two swing-up solutions (±π); mean lands at 0 |
| Non-physical target path | K | K | GFN denoising drift ≠ physical trajectory |
| Path forbids energy pumping | K | — | per-step tracking of monotone ramp on underactuated system |
| Confidence-ejection loop | K | K | reverse KL + GP confidence after success |
| Zeros-start catch-up | — | all | initial state 2.8 rad from q_ref[0] |

**On multimodality specifically (a common misconception):** it is a *cartpole*
issue (±π solutions → bimodal cloud → moment-match mean at the dead zone,
which is why classic MC-PILCO uses |θ|). It is **not** the reason the UR5
distributional variants fail — the UR5 cloud is essentially unimodal. UR5's
failure is the variance-in-denominator cheat + 12-D moment collapse.

---

## 7. Design principles (the takeaways)

1. **Match the cost to the task structure.** Goal-reaching → terminal
   distributional target with end-loaded weighting. Tracking → per-step
   feasible reference with uniform weighting.
2. **Never put a learned/particle variance in the denominator of the mean
   term.** It creates a cheat (inflate variance instead of tracking) that
   decouples the cost from the task. Use a fixed metric.
3. **Score particles individually; do not moment-collapse**, especially in
   high dimension where a diagonal Gaussian discards correlations.
4. **Targets must be physically reachable.** A GFN denoising path is not a
   physical trajectory; tracking it forces infeasible motion.
5. **Beware reverse KL when the model gets confident at the goal** — it ejects
   the policy from its own solution.

### The uncomfortable meta-result

The variant that works best on UR5 (**H**) barely uses the GFN — it is
classical tracking with the GFN's σ as a metric. The variants that genuinely
lean on the GFN's distributional output (**C, F, G, K**) are exactly the ones
that struggle. The honest research contribution is therefore a **negative
result with a mechanism**: *distributional / variance-based surrogates are
seductive but the variance-in-denominator is a trap; fixed-metric,
task-structured costs are robust.* Where the GFN does help unambiguously is as
a **goal specifier** for goal-reaching tasks (cartpole Variant C), not as a
per-step transition model for tracking.

---

## 8. Reproducing the case study

```bash
# Cartpole: C (works) vs K (fails)
python test_mcpilco_cartpole_variant_c.py -seed 1
python test_mcpilco_cartpole_variant_k.py -seed 1
python evaluate_variant_c.py -log results_variant_c_5/1/log.pkl
python evaluate_variant_j.py -log results_variant_k_reverse/1/log.pkl \
       -save results_variant_k_reverse/1/plots

# Cartpole mismatch visual (GFN denoising path vs physical swing-up)
python diagnose_variant_k.py -log results_variant_k_reverse/1/log.pkl

# UR5: H vs distributional family
python evaluate_ur5_variant_h.py -log results_ur5_variant_h/2/log.pkl
python evaluate_ur5_variant_c.py -log results_ur5_variant_c_5/1/log.pkl
```
