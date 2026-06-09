"""Pipeline runner — one entry point over the named stages.

    python -m src.pipeline list
    python -m src.pipeline run --exp <exp> [--from N] [--to N] [--only N[,N]] [--dry]
    python -m src.pipeline run --exp <exp> --with-rollout      # render rollout first

Stages are selected by NAME (see ``list``). v1 shells out to the existing
``scripts/NN_*.sh``; behaviour is identical to ``scripts/run_all.sh`` but the
order + selection live in ``src/pipeline/stages.py`` (single source of truth).

Run from the repo root. Env (proxy / GEMINI_API_KEY) is inherited by each stage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.pipeline.stages import STAGES, NAMES, get, index

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
DEFAULT_CONFIG = "configs/rollout/rollout_landmark_rxr.yaml"


def _resolve_plan(args) -> list:
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        bad = [n for n in names if n not in NAMES]
        if bad:
            sys.exit(f"unknown stage(s): {', '.join(bad)}\nknown: {', '.join(NAMES)}")
        return [get(n) for n in names]
    for n in (args.from_, args.to):
        if n and n not in NAMES:
            sys.exit(f"unknown stage: {n}\nknown: {', '.join(NAMES)}")
    lo = index(args.from_) if args.from_ else 0
    hi = index(args.to) if args.to else len(STAGES) - 1
    if lo > hi:
        sys.exit(f"--from {args.from_} comes after --to {args.to}")
    return list(STAGES[lo:hi + 1])


def _run_dir(args):
    """Best-effort resolve of results/{run}/ for the input pre-check. Returns
    None if the config can't be resolved (the check is then skipped)."""
    try:
        import yaml
        from src.pipeline.config import get_run_dir, resolve_exp
        with open(REPO / args.config) as f:
            cfg = yaml.safe_load(f)
        resolve_exp(cfg, args.exp, apply_current=False)
        return get_run_dir(cfg)
    except Exception:
        return None


def _warn_missing_inputs(stage, args) -> None:
    run_dir = _run_dir(args)
    if run_dir is None:
        return
    missing = [n for n in stage.needs if not (run_dir / n).exists()]
    if missing:
        print(f"  ⚠ {stage.name}: expected input(s) missing under {run_dir}: "
              f"{', '.join(missing)} — run the upstream stage first.")


def cmd_list(args) -> None:
    print(f"{'#':>2}  {'stage':16} {'script':30} description")
    print("-" * 88)
    for i, s in enumerate(STAGES):
        print(f"{i:>2}  {s.name:16} {s.script:30} {s.desc}")


def cmd_run(args) -> None:
    plan = _resolve_plan(args)
    print("Plan:", " → ".join(s.name for s in plan))
    print("exp :", args.exp)
    print()

    if args.with_rollout:
        exp_path = Path(args.exp)
        if not exp_path.is_file():
            sys.exit("--with-rollout needs --exp to be a selection YAML path")
        cmd = ["bash", str(SCRIPTS / "rollout.sh"),
               "--config", args.config, "--selection", args.exp]
        print("=" * 64, "\n  rollout\n   ", " ".join(cmd), "\n" + "=" * 64)
        if not args.dry and subprocess.run(cmd, cwd=str(REPO)).returncode != 0:
            sys.exit("✗ rollout failed")

    # Advisory: if we start mid-pipeline, the first stage's inputs must exist.
    if plan and index(plan[0].name) > 0:
        _warn_missing_inputs(plan[0], args)

    for s in plan:
        cmd = ["bash", str(SCRIPTS / s.script), "--exp", args.exp, *s.extra_flags]
        print("=" * 64, f"\n  {s.name}\n   ", " ".join(cmd), "\n" + "=" * 64)
        if args.dry:
            continue
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        if rc != 0:
            sys.exit(f"✗ stage {s.name} failed (exit {rc})")

    print("\n✓ pipeline complete" if not args.dry else "\n(dry run — nothing executed)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m src.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run a stage / range / the whole pipeline")
    pr.add_argument("--exp", required=True,
                    help="selection YAML path or bare expname")
    pr.add_argument("--from", dest="from_", metavar="STAGE",
                    help="start at this stage (inclusive)")
    pr.add_argument("--to", metavar="STAGE", help="stop after this stage (inclusive)")
    pr.add_argument("--only", metavar="STAGE[,STAGE...]",
                    help="run only these stage(s)")
    pr.add_argument("--with-rollout", action="store_true",
                    help="render the rollout first (needs --exp to be a YAML path)")
    pr.add_argument("--config", default=DEFAULT_CONFIG)
    pr.add_argument("--dry", action="store_true", help="print the plan; run nothing")
    pr.set_defaults(func=cmd_run)

    pl = sub.add_parser("list", help="list stages in pipeline order")
    pl.set_defaults(func=cmd_list)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
