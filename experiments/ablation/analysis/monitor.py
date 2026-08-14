#!/usr/bin/env python
"""Monitor ablation experiment progress, validate completeness, and summarize errors.

Usage:
    python monitor.py                    # Quick status summary
    python monitor.py --validate         # Full validation (checks every JSON)
    python monitor.py --watch            # Refresh every 30s until complete
    python monitor.py --errors           # Show all error details
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "ablation" / "outputs"

CONDITIONS = {"A": "full_pipeline", "B": "no_rag", "C": "visual_only", "D": "baseline"}
EXPECTED_SPECIMENS = 317


def count_outputs(condition: str) -> int:
    """Count completed JSON outputs for a condition."""
    cond_dir = OUTPUT_DIR / f"condition_{condition}"
    if not cond_dir.exists():
        return 0
    return len(list(cond_dir.glob("*.json")))


def check_slurm_jobs() -> list[dict]:
    """Query squeue for ablation jobs."""
    try:
        result = subprocess.run(
            ["squeue", "-u", subprocess.getoutput("whoami"), "--name=ablation,ablation-baseline",
             "--format=%i|%j|%T|%M|%N|%R", "--noheader"],
            capture_output=True, text=True, timeout=10,
        )
        jobs = []
        for line in result.stdout.strip().splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 6:
                jobs.append({
                    "job_id": parts[0].strip(),
                    "name": parts[1].strip(),
                    "state": parts[2].strip(),
                    "time": parts[3].strip(),
                    "node": parts[4].strip(),
                    "reason": parts[5].strip(),
                })
        return jobs
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def validate_outputs(condition: str) -> dict:
    """Validate all JSON outputs for a condition.

    Returns:
        {"valid": int, "errors": int, "empty": int, "details": [...]}
    """
    cond_dir = OUTPUT_DIR / f"condition_{condition}"
    if not cond_dir.exists():
        return {"valid": 0, "errors": 0, "empty": 0, "missing": EXPECTED_SPECIMENS, "details": []}

    valid = 0
    errors = 0
    empty = 0
    details = []

    for path in sorted(cond_dir.glob("*.json")):
        if path.stat().st_size == 0:
            empty += 1
            details.append({"file": path.name, "issue": "empty file"})
            continue

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors += 1
            details.append({"file": path.name, "issue": f"invalid JSON: {e}"})
            continue

        # Check for stage errors
        stages = data.get("stages", {})
        stage_errors = []
        for stage_name, stage_data in stages.items():
            if stage_data.get("error") and not stage_data.get("skipped"):
                stage_errors.append(f"{stage_name}: {stage_data['error']}")

        if stage_errors:
            errors += 1
            details.append({
                "file": path.name,
                "specimen": data.get("specimen_id", "?"),
                "issue": "; ".join(stage_errors),
            })
        else:
            valid += 1

    total = valid + errors + empty
    return {
        "valid": valid,
        "errors": errors,
        "empty": empty,
        "missing": max(0, EXPECTED_SPECIMENS - total),
        "total": total,
        "details": details,
    }


def print_status(validate: bool = False, show_errors: bool = False) -> bool:
    """Print current experiment status. Returns True if all complete."""
    print("=" * 70)
    print("ABLATION EXPERIMENT STATUS")
    print("=" * 70)

    # SLURM jobs
    jobs = check_slurm_jobs()
    if jobs:
        print(f"\n--- Active SLURM Jobs ({len(jobs)}) ---\n")
        print(f"{'Job ID':<15} {'Name':<20} {'State':<12} {'Time':<10} {'Node'}")
        print("-" * 65)
        for j in jobs:
            print(f"{j['job_id']:<15} {j['name']:<20} {j['state']:<12} {j['time']:<10} {j['node'] or j['reason']}")
    else:
        print("\n  No active SLURM jobs.")

    # Output counts
    print(f"\n--- Output Progress (target: {EXPECTED_SPECIMENS} per condition) ---\n")
    all_complete = True
    total_valid = 0

    for cond, name in CONDITIONS.items():
        if validate:
            result = validate_outputs(cond)
            status_parts = [f"{result['valid']} valid"]
            if result["errors"]:
                status_parts.append(f"{result['errors']} errors")
            if result["empty"]:
                status_parts.append(f"{result['empty']} empty")
            if result["missing"]:
                status_parts.append(f"{result['missing']} missing")
            status = ", ".join(status_parts)
            pct = result["valid"] / EXPECTED_SPECIMENS * 100
            total_valid += result["valid"]

            if result["valid"] < EXPECTED_SPECIMENS:
                all_complete = False

            print(f"  {cond} ({name:<14}): {status} ({pct:.0f}%)")

            if show_errors and result["details"]:
                for d in result["details"][:10]:
                    print(f"    ERROR: {d.get('specimen', d['file'])}: {d['issue']}")
                if len(result["details"]) > 10:
                    print(f"    ... and {len(result['details']) - 10} more")
        else:
            count = count_outputs(cond)
            pct = count / EXPECTED_SPECIMENS * 100
            total_valid += count
            if count < EXPECTED_SPECIMENS:
                all_complete = False
            print(f"  {cond} ({name:<14}): {count:>3}/{EXPECTED_SPECIMENS} ({pct:.0f}%)")

    total_expected = EXPECTED_SPECIMENS * len(CONDITIONS)
    print(f"\n  Total: {total_valid}/{total_expected} ({total_valid/total_expected*100:.0f}%)")

    # Check log files for recent activity
    log_dir = OUTPUT_DIR / "logs"
    if log_dir.exists():
        logs = sorted(log_dir.glob("*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            latest = logs[0]
            mtime = time.strftime("%H:%M:%S", time.localtime(latest.stat().st_mtime))
            print(f"\n  Latest log activity: {latest.name} at {mtime}")

    print("\n" + "=" * 70)

    if all_complete:
        print("\nAll conditions complete. Run analysis:")
        print("  python experiments/ablation/analysis/compute_metrics.py")

    return all_complete


def save_provenance_log() -> None:
    """Save experiment provenance metadata for reproducibility."""
    import platform

    provenance = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    # Git commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(PROJECT_ROOT), timeout=5,
        )
        provenance["git_commit"] = result.stdout.strip()
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
            text=True, cwd=str(PROJECT_ROOT), timeout=5,
        )
        provenance["git_branch"] = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Package versions
    try:
        import faiss
        provenance["faiss_version"] = faiss.__version__  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pass
    try:
        import torch
        provenance["torch_version"] = torch.__version__
        provenance["cuda_version"] = torch.version.cuda or "N/A"
    except ImportError:
        pass
    try:
        import sentence_transformers
        provenance["sentence_transformers_version"] = sentence_transformers.__version__
    except ImportError:
        pass

    # SLURM job IDs
    jobs = check_slurm_jobs()
    if jobs:
        provenance["slurm_jobs"] = jobs

    # Experiment config
    config_path = PROJECT_ROOT / "experiments" / "ablation" / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            provenance["experiment_config"] = yaml.safe_load(f)

    # Condition completion counts
    provenance["completion"] = {
        cond: count_outputs(cond) for cond in CONDITIONS
    }

    out_path = OUTPUT_DIR / "provenance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    print(f"Provenance log saved to {out_path}")


def main() -> None:
    """Run monitoring commands."""
    parser = argparse.ArgumentParser(description="Monitor ablation experiments.")
    parser.add_argument("--validate", action="store_true", help="Validate all output JSONs")
    parser.add_argument("--errors", action="store_true", help="Show error details")
    parser.add_argument("--watch", action="store_true", help="Refresh every 30s until complete")
    parser.add_argument("--provenance", action="store_true", help="Save provenance log")
    args = parser.parse_args()

    if args.provenance:
        save_provenance_log()
        return

    if args.watch:
        try:
            while True:
                subprocess.run(["clear"])
                done = print_status(validate=args.validate, show_errors=args.errors)
                if done:
                    save_provenance_log()
                    break
                print("\nRefreshing in 30s... (Ctrl+C to stop)")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_status(validate=args.validate or args.errors, show_errors=args.errors)


if __name__ == "__main__":
    main()
