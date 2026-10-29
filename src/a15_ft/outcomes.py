"""Scene outcomes of capture runs, keyed by session (= rollout_id = the capture's session directory), for
outcome-weighted training (train.py --outcomes). Usage: python -m a15_ft.outcomes OUT.json RUN_DIR [RUN_DIR ...]
pass: score >= 0.95; fail: at-fault collision, offroad or left the corridor (the zero-score events); partial: the rest
(rear-ended, low progress)."""
import json
import os
import sys


def classify(r):
    m = r["metrics"]
    if r["score"] >= 0.95:
        return "pass"
    if m.get("collision_at_fault") or m.get("offroad") or m.get("left_corridor_laterally"):
        return "fail"
    return "partial"


def main():
    out, runs = sys.argv[1], sys.argv[2:]
    res = {}
    for d in runs:
        f = os.path.join(d, "aggregate", "results-summary.json")
        if not os.path.exists(f):
            print(f"no results in {d}", flush=True)
            continue
        for r in json.load(open(f))["rollouts"]:
            res[r["rollout_id"]] = {"score": float(r["score"]), "cls": classify(r), "clip": r["clipgt_id"]}
    json.dump(res, open(out, "w"), indent=0)
    n = {c: sum(v["cls"] == c for v in res.values()) for c in ("pass", "partial", "fail")}
    print(f"{len(res)} sessions: {n}", flush=True)


if __name__ == "__main__":
    main()
