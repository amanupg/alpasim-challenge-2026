# Final PAI submission runbook (supervised; Aman approves every step)

Image: the local `teams/amanupg:v7`, image id `05e650bdbecb`, git `6ed1db2`.
- v7 settings plus the wait-then-go start fix (`A15_WAITGO_K=3`).
- Expert sha256 `f9da7cf3…` (ftC_w50).
- Verified as the `alpasim` user on 2026-10-28; the driver unit tests pass.

Held-out evidence (321 scenes, leaderboard rendering):

| Comparison | Diff [90% interval] | Wins / losses |
|---|---|---|
| v5 → v7 | +0.057 [+0.023, +0.093] | 65 / 48 |
| v7 → v7 + wait-go | +0.004 [+0.000, +0.009] | 4 / 0 |

```
URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/amanupg:v7
CLI="python3 external/alpasim/e2e_challenge/competitor_cli/alpasim_challenge.py"   # run from alpasim/
```

## Step 0. Sign in (Aman)

1. `$CLI configure-token`. Aman enters the token; it is never typed by Claude and never put on a command line.
2. `$CLI me` must show `authenticated: true`.
3. `$CLI limits` must show `pai` remaining 1 and the `pai-warmup` remaining count. If no PAI warmup is left, stop and decide.

## Step 1. Re-check the local image (Claude, about 2 min)

Run as `alpasim` inside the image:
- driver md5s: driver `cfa0141e`, policy `b880af2b`, route_choice `00527e05`, trajectory `ade286bc`;
- env `A15_WAITGO_K=3`;
- expert sha `f9da7cf3`;
- tests PASS.

## Step 2. Push (Aman approves; Claude runs it, about 20-40 min upload)

1. `$CLI ecr-login`
2. `docker push $URI`
3. Record the pushed digest in LEDGER.md.

## Step 3. Free warmup: NOT the official slot (Aman approves)

`$CLI submit --track pai-warmup $URI`

- **The track must read `pai-warmup`.** `pai` alone spends the last official slot.
- Watch it with `$CLI status <id>` until it finishes (about 2-3 h).
- Check that every rollout ran, the wave time is within the budget, and there was no out-of-memory or crash.
- Compare with v5's official run: 9,522 of 14,400 s.

## Step 4. Official submission (Aman runs it himself)

`$CLI submit --track pai $URI`

- Afterwards: `$CLI submissions`, and record the ID and digest in LEDGER.md.
- When the score appears, fill the report's TODO (official v7 result) and add the warmup's wave time and peak memory.
