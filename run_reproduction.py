from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(*args: str) -> None:
    print("+", sys.executable, *args)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["verify", "summaries", "full"], default="verify")
    args = parser.parse_args()
    if args.stage == "verify":
        run("verify_release.py")
        return
    if args.stage == "summaries":
        run("-m", "blackchin_tilapia_analysis.scripts.21_pooled_metrics_stats_v2")
        run("verify_release.py", "--skip-checksums")
        return
    run("verify_inputs.py")
    run("-m", "blackchin_tilapia_analysis.scripts.20_full_experiment_v2")
    run("-m", "blackchin_tilapia_analysis.scripts.21_pooled_metrics_stats_v2")
    run("verify_release.py", "--skip-checksums")


if __name__ == "__main__":
    main()
