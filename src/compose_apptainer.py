"""Run a wizard-generated docker-compose file with Apptainer on one Slurm node.

Insomnia has neither docker nor pyxis/enroot, so the wizard is run with
wizard.run_method=NONE (it only writes docker-compose.yaml) and this script plays the part
of `docker compose up --exit-code-from runtime-0`:
  * every service becomes `apptainer exec --nv <sif> bash -c <command>`, host network
    (the challenge dev presets set debug_flags.use_localhost, so services talk over localhost);
  * services start in file order, each in its own process group, logs in <logdir>/<name>.log;
  * before the runtime starts, every published port of the other services must accept a TCP
    connection (as the wizard's SLURM deployment does);
  * the runtime runs in the foreground; its exit code is ours; everything else is torn down.
GPU device_ids in the compose file index into this job's CUDA_VISIBLE_DEVICES.
Host sides of volumes under /repo (the checkout baked into the SIF, where the wizard ran)
are mapped to the identical host checkout given by --repo-host.

Usage: python3 compose_apptainer.py COMPOSE.json --sif-dir DIR --repo-host DIR --logdir DIR
(COMPOSE.json = the YAML converted to JSON by the SIF's python; the host python has no yaml)
"""
import argparse, json, os, signal, socket, subprocess, sys, time

# image -> file or sandbox directory name under --sif-dir (sandboxes: this apptainer lacks squashfuse)
IMAGES = [("nvcr.io/nvidia/nre/nre-ga", "nre-ga_26.04"), ("alpasim-base", "alpasim-base")]


def sif_for(image, sif_dir):
    for prefix, f in IMAGES:
        if image.split(":")[0].endswith(prefix):
            return os.path.join(sif_dir, f)
    raise SystemExit(f"no SIF mapped for image {image!r}")


def host_path(p, repo_host):
    return repo_host + p[len("/repo"):] if p == "/repo" or p.startswith("/repo/") else p


def gpu_env(svc, gpus):
    devs = (((svc.get("deploy") or {}).get("resources") or {}).get("reservations") or {}).get("devices") or []
    for d in devs:
        if d.get("count") == "all":
            return ",".join(gpus)
        if d.get("device_ids"):
            return ",".join(gpus[int(i)] for i in d["device_ids"])
    return None


def build(name, svc, a, gpus):
    cmd = ["apptainer", "exec", "--nv", "--writable-tmpfs"]
    for v in svc.get("volumes") or []:
        parts = v.split(":")
        hp = host_path(parts[0], a.repo_host)
        if not os.path.exists(hp) and not os.path.splitext(hp)[1]:
            os.makedirs(hp)          # docker creates missing bind sources as directories
        cmd += ["-B", ":".join([hp] + parts[1:])]
    if svc.get("working_dir"):
        cmd += ["--pwd", svc["working_dir"]]
    env = {}
    for e in svc.get("environment") or []:
        if "=" in e:
            k, val = e.split("=", 1); env[k] = val
    g = gpu_env(svc, gpus)
    if g is not None:
        env["CUDA_VISIBLE_DEVICES"] = g
    for k, val in env.items():
        cmd += ["--env", f"{k}={val}"]
    cmd.append(sif_for(svc["image"], a.sif_dir))
    c = svc.get("command")
    if c:
        assert svc.get("entrypoint") == "bash" and c[0] == "-c", (name, svc.get("entrypoint"), c[:1])
        cmd += ["bash", "-c", c[1].replace("$$", "$")]
    return cmd


def ports(svc):
    return [int(p.split(":")[0]) for p in svc.get("ports") or []]


def is_open(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("compose")
    ap.add_argument("--sif-dir", required=True)
    ap.add_argument("--repo-host", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    services = json.load(open(a.compose))["services"]
    gpus = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")
    runtime = [n for n in services if n.startswith("runtime")]
    others = [n for n in services if n not in runtime]
    os.makedirs(a.logdir, exist_ok=True)
    procs = []

    def teardown(*_):
        for n, p in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        time.sleep(5)
        for n, p in procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
    signal.signal(signal.SIGTERM, lambda *x: (teardown(), sys.exit(143)))

    try:
        for n in others:
            cmd = build(n, services[n], a, gpus)
            print(f"[compose] start {n}: {' '.join(cmd[:-1])} <command>", flush=True)
            if a.dry_run:
                continue
            procs.append((n, subprocess.Popen(cmd, stdout=open(os.path.join(a.logdir, f"{n}.log"), "w"),
                                              stderr=subprocess.STDOUT, start_new_session=True)))
        if a.dry_run:
            for n in runtime:
                print(f"[compose] runtime {n}: {' '.join(build(n, services[n], a, gpus)[:-1])} <command>")
            return 0
        want = {p: n for n in others for p in ports(services[n])}
        t0 = time.time()
        while want:
            for n, p in procs:
                if p.poll() is not None:
                    raise SystemExit(f"[compose] {n} exited early with {p.returncode}; see {a.logdir}/{n}.log")
            want = {p: n for p, n in want.items() if not is_open(p)}
            if time.time() - t0 > a.timeout:
                raise SystemExit(f"[compose] ports never opened: {want}")
            time.sleep(2)
        print(f"[compose] all service ports open after {time.time() - t0:.0f}s", flush=True)
        rc = 0
        for n in runtime:
            cmd = build(n, services[n], a, gpus)
            print(f"[compose] runtime {n} (foreground)", flush=True)
            with open(os.path.join(a.logdir, f"{n}.log"), "w") as f:
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT) or rc
        print(f"[compose] runtime exit {rc}", flush=True)
        return rc
    finally:
        teardown()


if __name__ == "__main__":
    sys.exit(main())
