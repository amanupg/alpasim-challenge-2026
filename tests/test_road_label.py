"""RoaD target selection (a15_ft.train.road_target): nearest policy sample within 3 m, else recovery label."""
import ast
import os
import sys

import numpy as np

# load only the numpy helpers from train.py (the module imports torch and the model stack)
src = open(os.path.join(os.path.dirname(__file__), "..", "src", "a15_ft", "train.py")).read()
ns = {"np": np}
for node in ast.parse(src).body:
    if isinstance(node, ast.FunctionDef) and node.name in ("recovery_offset", "smooth_recovery", "_corners", "road_target"):
        exec(compile(ast.Module([node], []), "train.py", "exec"), ns)

bad = 0


def check(name, ok):
    global bad
    bad += not ok
    print(("ok  " if ok else "BAD ") + name)


t = 0.1 * np.arange(1, 65)
eye = np.repeat(np.eye(3)[None], 64, 0)
line = lambda y: np.stack([10 * t, np.full_like(t, y), np.zeros_like(t)], 1)
s = {"fut_xyz": line(0.0).astype(np.float32), "fut_rot": eye.astype(np.float32),
     "samp_xyz": np.stack([line(2.0), line(0.5), line(-1.0)]), "samp_rot": np.stack([eye] * 3)}
out, i, d = ns["road_target"](s)
check("picks the sample nearest the recording (0.5 m off)", i == 1 and abs(d - 0.5) < 1e-6)
check("target is that sample", np.allclose(out["fut_xyz"], line(0.5)))
s2 = dict(s, samp_xyz=np.stack([line(5.0), line(-4.0)]), samp_rot=np.stack([eye] * 2))
out2, i2, d2 = ns["road_target"](s2)
check("no sample within 3 m -> recovery label (-1)", i2 == -1 and abs(d2 - 4.0) < 1e-6)
check("recovery label is the recording here (ego on it)", np.allclose(out2["fut_xyz"], line(0.0), atol=1e-5))
print("PASS" if bad == 0 else f"FAIL ({bad})")
sys.exit(bad > 0)
