"""Failure taxonomy over local AlpaSim runs (stdlib only; reads aggregate/results-summary.json).

Each rollout gets one PRIMARY category, checked in the evaluator's order of severity:
  error (rollout did not produce metrics) > at-fault collision > offroad > left corridor >
  low progress (progress_clipped_rel < 0.8, so progress_score < 1) > pass.
Several runs of the SAME config on the same scenes also show which scenes are STABLE
(same category in every run) and which FLIP (run-to-run noise), so later comparisons can
separate real effects from simulator nondeterminism.
Usage: python3 failure_taxonomy.py RUN_DIR [RUN_DIR ...]
"""
import collections
import json
import os
import sys

ORDER = ["error", "at-fault collision", "offroad", "left corridor", "low progress", "pass"]


def category(r):
    m = r.get("score_metrics") or {}
    if r.get("failure_reason") and not m.get("progress_rel"):
        return "error"
    if (m.get("collision_at_fault") or 0) > 0:
        return "at-fault collision"
    if (m.get("offroad") or 0) > 0:
        return "offroad"
    if (m.get("left_corridor_laterally") or 0) > 0:
        return "left corridor"
    if (m.get("progress_clipped_rel") or 0) < 0.8 and (m.get("gt_dist_traveled_m") or 99) >= 5.0:
        return "low progress"
    return "pass"


def load(run):
    d = json.load(open(os.path.join(run, "aggregate", "results-summary.json")))
    return {r["clipgt_id"]: (category(r), float(r.get("score") or 0.0), r) for r in d["rollouts"]}


def main():
    runs = sys.argv[1:]
    per_run = {os.path.basename(r.rstrip("/")): load(r) for r in runs}
    for name, rows in per_run.items():
        c = collections.Counter(v[0] for v in rows.values())
        n = len(rows)
        print(f"{name}: n={n}  " + "  ".join(f"{k} {c.get(k, 0)} ({100 * c.get(k, 0) / n:.0f}%)" for k in ORDER))
    if len(per_run) > 1:
        scenes = set.intersection(*(set(v) for v in per_run.values()))
        stable = [s for s in scenes if len({per_run[r][s][0] for r in per_run}) == 1]
        print(f"\nscenes in all runs: {len(scenes)}; same category in every run: {len(stable)}; flip: {len(scenes) - len(stable)}")
        c = collections.Counter(per_run[next(iter(per_run))][s][0] for s in stable)
        print("stable categories: " + ", ".join(f"{k} {c.get(k, 0)}" for k in ORDER))
        print("\nscenes that fail in EVERY run (the reliable failure set):")
        for s in sorted(stable):
            cat = per_run[next(iter(per_run))][s][0]
            if cat != "pass":
                m = per_run[next(iter(per_run))][s][2].get("score_metrics") or {}
                print(f"  {cat:20} {s}  progress {m.get('progress_clipped_rel')}  dist_to_gt {m.get('dist_to_gt_trajectory')}")


if __name__ == "__main__":
    main()
