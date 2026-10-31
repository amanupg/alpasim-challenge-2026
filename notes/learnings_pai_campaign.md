# PAI campaign: lessons (Sep 26 – Oct 30, 2026)

The detailed record, with every number and job ID, is in `LEDGER.md`. This file holds what we learned, technical and process.

## What worked (in order of impact)

| Change | Effect |
|---|---|
| **Route snap** | Replace the plan's geometry with a cubic Hermite curve onto the route when the plan is > 2 m off and the route is within 15° of heading. Official 0.611 → 0.652. |
| **Fine-tuning the action expert on simulator-rendered data with recovery (DAgger) labels** | The recovery labels mattered. Recording-following data alone cut open-loop ADE by 21–24% but was neutral or negative closed-loop. |
| **WiSE-FT** | 50/50 average of the fine-tuned and released expert restored speed. |
| **Route text** ("Turn left in X m") | Worthless with the released expert, +0.036 with the averaged fine-tuned one. **Components interact**, so re-test on top of the current best system. |
| **Forced `<traj_future_start>` token** | The model skipped it in 30–42% of plans. Read raw outputs. |
| **Wait-then-go start rule** | Small (+0.004) but deterministic, and every flip is explained. |

Final result: v7 + wait-go. Held-out 321: v5 → v7 +0.057 [+0.023, +0.093]; + wait-go +0.004 [+0.000, +0.009].

## What did not work, and why that is informative

**Inference-time changes the model wasn't trained for** (train–test mismatch costs more than the intended benefit):
- 2 Hz replanning;
- 20 flow steps instead of 10;
- six cameras;
- speed scaling;
- time warp;
- NVIDIA's lateral-consistency selection (−0.073; it drove non-default samples 89% of the time).

**Every fine-tune variant after run C** (properly dosed, Oct 29):
- T2c (teacher + RoaD) −0.020;
- T2d (outcome-weighted) −0.021;
- R2 (RoaD) −0.058;
- seeds Cs2 −0.034 and Cs3 −0.026;
- seed soup +0.003.

Pattern: **each improves pace (fewer rear-ends, more full passes) but breaks more lane and branch decisions than it fixes.** Zero-score corridor exits are fragile coin flips that any model change tips both ways. **Stability of discrete decisions, not average accuracy, is the bottleneck.**

**The Alpamayo 2-Super teacher (34B):** per sample it was no closer to the recording than the student on the student's own states (0.83 vs 0.82 m). Its value was diversity. A student cannot distil cues it does not see (6 vs 4 cameras, frozen VLM).

**Run C was a lucky draw:** three other seeds of the same recipe all scored lower.

## Evaluation lessons

- **Optimise and validate the metric that ranks you, not a convenient stand-in (the costliest lesson).**
  - v7 beat v5 officially on mean scene score (+0.025), at-fault distance and distance to GT, yet its PCS fell 2226 → 2151 (rank 16 → 28).
  - PCS is an item-response (zoib) fit over all teams' per-scene scores; it weighs *which* scenes change. A held-out W/L of 65/48 can raise the mean and still lower PCS.
  - We validated only the mean, and the "+28 PCS" projection was never tested. Before trusting a gain, fit a local PCS proxy (zoib/IRT on per-scene scores of several of our own and reference entries), and treat scene losses on easy scenes as costly.

- **Determinism:** with the Harmonizer off, closed loop on one GPU type repeats bit for bit. Use this for paired, exact comparisons. The ±0.05 noise came from the Harmonizer.
- **Screening vs held-out:** about 40 decisions on 120 scenes produce winner's curse. Confirm on an untouched lockbox before adopting.
- **Local vs official:** local gains converted at about 0.6×. Never quote a screening number as the expected official result.
- **Open-loop ADE does not predict closed-loop score.** Use it only as a "did training learn?" check, and only for recording targets.

## Process lessons (the hard ones)

1. **Validity before verdicts.**
   - Several runs were reported as "rejected" when they were weak or broken tests:
     - T2b was under-trained (flat loss from my lr choice);
     - T had only 30% teacher labels;
     - R and T were rejected inside Harmonizer noise.
   - Each was caught only when Aman probed. The fix is now standing practice: a validity line on every result (loss trend, weights moved, activation check), a stated power, honest verdict wording, and up-front disclosure of my own design choices that weaken a test.
   - See memory `validity-before-verdicts`.
2. **Questions are not orders:** judge ideas on merit, and say "not worth it" with reasons.
3. **Deterministic gains only** for the final image.
4. **No image push or warmup without Aman's live supervision and approval.** He submits; I never submit.
5. **Never put credentials in command lines.** The CLI token goes through a hidden prompt; challenge, HF and RunPod sign-ins are done by Aman.
6. **Cluster:**
   - `source /etc/profile` first; size `--mem` from measured MaxRSS (MaxMemPerCPU makes big asks unschedulable).
   - Watch the shared 5 TB fileset, which other users fill.
   - Per-item approval for every deletion.
   - Self-healing socket watchers.
7. **Rental:** worth it for labelling with a big model, but not for simulator work, whose stack lives on Insomnia.

## For the innovation report

The strongest, most transferable findings are:
1. bit-exact determinism with the Harmonizer off, enabling paired held-out confirmation;
2. the lane-decision fragility analysis;
3. the bf16 AdamW silent-update-loss finding, and stochastic rounding as the fix;
4. teacher-on-student-states analysis (diversity, not accuracy).
