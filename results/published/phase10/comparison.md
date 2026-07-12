# Phase 10 — Preference Drift and Adaptation

Seven `hnsw_ranker` arms on the medium config (10k users x 100k reels x 200 interactions/user =
2.0M impressions per arm), seed 42, with **scheduled whole-population preference drift at
interaction 100** (round 10 of 20): four disjoint quarter-cohorts (deterministic userId-hash
ranges) each switch their hidden preference to a distinct concentrated 3-topic mix
(weights 0.6/0.25/0.15). Config: `configs/phase10-drift.json`; each arm's fully-resolved variant
is committed as `<arm>/arm_config.json`. All seven arms ran CONCURRENTLY (see caveats).

Arms: **(a) session-weight sweep** — session-aware 0.65/0.35 (shipped default), long-term-only
1.0/0.0, session-heavy 0.4/0.6; **(b) learning-rate sweep** — long-term eta in {0.05, 0.15, 0.30}
at the default 0.65/0.35 blend (default eta=0.02 is the session-aware arm); **(c) frozen** —
learning disabled (the "static model" comparator).

## Headline: session-aware adapts faster than static / long-term-only

- **Estimated<->hidden alignment (the unconfounded adaptation measure): the session-aware arm is
  >= the long-term-only arm at EVERY post-drift round**, widening from 0.370 vs 0.360 (drift
  round) to **0.662 vs 0.472** (final). The frozen arm never adapts at all: flat 0.335 from the
  drift round to the end.
- **Drifted-cohort reward: session-aware >= long-term-only at every round from one round after
  drift onward**, with a monotonically widening gap (+0.001 at round 11 to +0.134 at round 19).
  Final-2-round mean: **0.431 vs 0.295 (+46%)**; the frozen arm limps at 0.174.
- Session-aware pays a real drop at drift (round-9 0.215 -> round-10 0.187, drop 0.023 vs its
  3-round pre-drift baseline 0.2098) and re-crosses 95% of its own baseline within
  **20 interactions**; its estimate<->hidden alignment re-crosses 95% of its pre-drift value at
  the very first post-drift measurement (recovery driven by the session window, which refills
  with new-preference interactions within ~2 rounds).
- Overall reward/impression across the whole run: session-aware 0.2564 vs long-term-only 0.1945
  (+32%) vs frozen 0.1369 (+87%).

## Read the curves from slopes and alignment, not instantaneous levels

Every arm — even frozen — jumps UP at the drift round (frozen: 0.073 -> 0.211). The drift target
`normalize(sum w_i * centre_i)` is a noise-free, concentrated topic mix, so the achievable
affinity ceiling of a drifted user is higher than that of an original (noisy, flatter-mix) user:
part of the post-drift level is a **reward-scale shift, not adaptation**. Two visible
consequences, both documented rather than corrected: (1) the frozen arm's own-baseline "recovery"
is instant and its `reward_drop` is negative — its pre-drift baseline had already collapsed to
0.068 (the Phase 7 frozen-arm exhaustion effect) and the ceiling shift alone lifts it over the
bar; (2) the frozen arm's alignment jumps 0.216 -> 0.335 at drift because concentrated mixes sit
closer to the static global-average estimate than noisy per-user originals do. Adaptation
claims in this report therefore rest on post-drift slopes, round-by-round dominance, and final
levels — all of which favour session-aware over long-term-only over frozen, unambiguously.
The same ceiling shift is why `interactions_to_common_target` (a common absolute bar of 0.95 x
the session-aware baseline) reads 10 for the barely-personalized arms (long-term-only, frozen):
the ceiling jump carries them over a bar derived from a *pre-shift* baseline. The in-tree
statistical test uses an equal-weight drift mix at small scale, where the shift is mild and the
common-target comparison is meaningful; there session-aware reaches the common bar no later than
long-term-only.

## Learning-rate sensitivity (long-term eta, drift scenario)

Monotone within this 200-interaction horizon: higher eta digs a **deeper trough** (alignment min
0.370 at eta=0.02 -> 0.206 at eta=0.30; reward drop 0.023 -> 0.149, note the higher etas also
start from higher pre-drift baselines 0.2098 -> 0.2588) but recovers **steeper and higher**
(final alignment 0.662 -> 0.747; final-2-round drifted reward 0.431 -> 0.475; overall
reward/impression 0.2564 -> 0.2687, the best of all arms). The deeper trough is the flip side of
a more strongly-committed estimate at drift time. No instability penalty is visible at this
horizon; whether eta=0.30 stays ahead over longer horizons or noisier regimes is untested here
(honest limitation, not a claim).

## Adaptation metrics table

See `comparison.csv` (one row per arm): overall reward/impression and true affinity, pre-drift
baseline, trough, drop, own-baseline 95% recovery interactions, common-target crossing,
post-drift 5-round and final 2-round means, alignment (pre-drift / post-drift min / final), and
cumulative sampled regret over the adaptation window (affinity units — Phase 4 deviation — and
whole-population, not cohort-split). Full per-round curves are in each arm's
`learning_curve.csv` (extended with `drifted_mean_reward,control_mean_reward,drifted_alignment,
control_alignment`; control columns are `nan` here because the whole population drifts).

## Figures (`figures/`, generated by `scripts/plot_results.py`)

- `drift_recovery.png` — drifted-cohort reward per round, drift-start marker, per-arm 95%
  own-baseline guides (the phase's recovery-curve exit criterion).
- `reward_curve.png`, `alignment_curve.png`, `cumulative_regret.png` — all seven arms overlaid.

Regenerate with:
`UV_PYTHON=3.12 uv run --project scripts scripts/plot_results.py results/published/phase10/<arms...> --labels ... --out results/published/phase10/figures`

## Caveats

- The seven arms ran concurrently on one machine: absolute latencies carry cache/memory
  contention (retrieval p50 ~2.3-4.4 ms vs ~1.9 ms solo in Phase 7); TDD 27's p95 < 10 ms holds
  in every arm; cross-arm quality comparisons are like-for-like. `metadata.json` records
  `reel_rank_dirty: true` (runs predate the phase commit — same provenance situation as Phases
  1/4-9).
- The learning-rate sweep varies the LONG-TERM eta. TDD 11.2's "session learning rate" does not
  exist in this implementation: Phase 7 (per its phase plan) built the session vector as a
  lambda-window recompute (TDD 11.3), which has no incremental rate. Recorded as a Phase 10
  deviation in commit.md.
- Whole-population drift maximizes measurable signal; the four distinct cohort mixes prevent the
  population from collapsing onto a single preference (which would let popularity alone "adapt").
  Popularity/trending realignment still contributes to every arm's post-drift level — it is part
  of why even frozen holds ~0.17-0.21 — but it cannot produce the learning arms' slopes, and the
  alignment metric is untouched by it.
