import subprocess, sys

SCRIPT = "sac.py"

EXPERIMENTS = {
    "sac_noln_utd1":            "--n 2  --m 2 --utd 1",
    "sac_ln_utd1":              "--n 2  --m 2 --utd 1  --layernorm",
    "sac_ln_utd20":             "--n 2  --m 2 --utd 20 --layernorm",
    "sac_synther_noln_utd20":   "--n 2  --m 2 --utd 20 --synthetic",
    "sac_synther_ln_utd20":     "--n 2  --m 2 --utd 20 --synthetic --layernorm",
    "redq_noln_utd20":          "--n 10 --m 2 --utd 20",
}


for name, flags in EXPERIMENTS.items():
    cmd = [sys.executable, SCRIPT, "--run-name", name, *flags.split()]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!! {name} failed (code {result.returncode}), continuing")