"""The ``rdw`` command line: run, list, and inspect workflow runs.

Commands:

* ``rdw run script.py [--resume RUN_ID] [--budget N] [...]`` — load a user
  script that exposes ``async def workflow(wf)`` and execute it inside a
  :class:`~rdw.engine.Workflow`. ``--resume`` replays journal-cached agent
  results and goes live at the first divergence.
* ``rdw runs`` — list runs under ``.rdw/runs`` (newest first).
* ``rdw show <run>`` — dump a run's journal in readable form.

A workflow script is ordinary async Python::

    # review.py
    from pydantic import BaseModel

    class Verdict(BaseModel):
        approve: bool
        summary: str

    async def workflow(wf):
        async with wf.phase("review"):
            v = await wf.agent("Review HEAD~1..HEAD strictly.",
                               schema=Verdict, label="reviewer")
        wf.log(f"approve={v.approve}")
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

from .budget import Budget
from .engine import Workflow, new_run_id
from .errors import RdwError
from .journal import read_journal_lines


def _load_workflow_fn(script_path: Path) -> Any:
    """Import ``script_path`` and return its ``async def workflow(wf)``."""
    if not script_path.exists():
        raise RdwError(f"script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("rdw_user_script", script_path)
    if spec is None or spec.loader is None:
        raise RdwError(f"cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rdw_user_script"] = module
    spec.loader.exec_module(module)
    fn = getattr(module, "workflow", None)
    if fn is None or not inspect.iscoroutinefunction(fn):
        raise RdwError(
            f"{script_path} must define `async def workflow(wf)` "
            "(the entry point rdw run invokes)"
        )
    return fn


def _write_meta(wf: Workflow, script: Path, budget: float | None) -> None:
    meta = {
        "run_id": wf.run_id,
        "script": str(script.resolve()),
        "budget": budget,
        "model": wf.default_model,
        "effort": wf.default_effort,
        "created": time.time(),
    }
    (wf.journal.run_dir / "meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


async def _drive(wf: Workflow, fn: Any) -> None:
    async with wf:
        await fn(wf)


def cmd_run(args: argparse.Namespace) -> int:
    script = Path(args.script)
    fn = _load_workflow_fn(script)
    run_id = args.resume or new_run_id()
    wf = Workflow.open(
        run_id=run_id,
        root=args.root,
        resume=bool(args.resume),
        budget=Budget(total=args.budget),
        model=args.model,
        effort=args.effort,
        cwd=args.cwd,
        concurrency=args.concurrency,
    )
    _write_meta(wf, script, args.budget)
    try:
        asyncio.run(_drive(wf, fn))
    except KeyboardInterrupt:
        print(f"\ninterrupted — resume with: rdw run {script} --resume {wf.run_id}")
        return 130
    finally:
        print(wf.report())
        print(f"run dir: {wf.journal.run_dir}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    runs_dir = Path(args.root) / "runs"
    if not runs_dir.is_dir():
        print("no runs yet")
        return 0
    entries = sorted(
        (p for p in runs_dir.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not entries:
        print("no runs yet")
        return 0
    for run_dir in entries:
        lines = read_journal_lines(run_dir)
        agents = [ln for ln in lines if ln.get("type") == "agent"]
        ok = sum(1 for a in agents if a.get("status") == "ok")
        credits = sum(float(a.get("credits") or 0.0) for a in agents)
        meta_path = run_dir / "meta.json"
        script = ""
        if meta_path.exists():
            try:
                script = Path(json.loads(meta_path.read_text()).get("script", "")).name
            except (json.JSONDecodeError, OSError):
                pass
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(run_dir.stat().st_mtime))
        print(
            f"{run_dir.name}  {stamp}  {ok}/{len(agents)} agents ok  "
            f"{credits:.2f} AIU  {script}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    run_dir = Path(args.root) / "runs" / args.run
    if not run_dir.is_dir():
        print(f"no such run: {args.run}", file=sys.stderr)
        return 1
    lines = read_journal_lines(run_dir)
    if not lines:
        print("(empty journal)")
        return 0
    for obj in lines:
        kind = obj.get("type")
        if kind == "agent":
            mark = "✓" if obj.get("status") == "ok" else "✗"
            loc = f"[{obj['phase']}] " if obj.get("phase") else ""
            wall = max(0.0, float(obj.get("ended") or 0) - float(obj.get("started") or 0))
            print(
                f"{mark} #{obj.get('index'):<3} {loc}{obj.get('label')}: "
                f"{obj.get('status')}  {float(obj.get('credits') or 0):.2f} AIU  {wall:.1f}s"
            )
            if args.verbose:
                if obj.get("status") == "ok":
                    print("    " + json.dumps(obj.get("result"), ensure_ascii=False)[:2000])
                elif obj.get("error"):
                    print(f"    error: {obj['error']}")
        elif kind == "divergence":
            print(f"! divergence at position {obj.get('index')} — live from here")
        elif kind == "log":
            loc = f"[{obj['phase']}] " if obj.get("phase") else ""
            print(f"· {loc}{obj.get('message')}")
        else:
            print(f"? {json.dumps(obj, ensure_ascii=False)[:200]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rdw",
        description="Dynamic multi-agent workflows for the GitHub Copilot SDK.",
    )
    parser.add_argument(
        "--root",
        default=".rdw",
        help="run-store root directory (default: .rdw)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_root(p: argparse.ArgumentParser) -> None:
        # argparse subparsers do not recognize parent optionals placed after
        # the subcommand, so each subcommand re-declares --root. SUPPRESS keeps
        # a pre-subcommand `rdw --root DIR run ...` working: the subparser only
        # writes the attribute when the flag actually appears after it.
        p.add_argument(
            "--root",
            default=argparse.SUPPRESS,
            help="run-store root directory (default: .rdw)",
        )

    p_run = sub.add_parser("run", help="execute a workflow script")
    add_root(p_run)
    p_run.add_argument("script", help="path to a script exposing `async def workflow(wf)`")
    p_run.add_argument("--resume", metavar="RUN_ID", help="resume a prior run's journal")
    p_run.add_argument("--budget", type=float, help="hard AI-credit ceiling for the run")
    p_run.add_argument("--model", help="default model for agents")
    p_run.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh"],
        help="default reasoning effort for agents",
    )
    p_run.add_argument("--cwd", help="default working directory for agents")
    p_run.add_argument(
        "--concurrency", type=int, help="max simultaneous live agent sessions"
    )
    p_run.set_defaults(func=cmd_run)

    p_runs = sub.add_parser("runs", help="list recorded runs")
    add_root(p_runs)
    p_runs.set_defaults(func=cmd_runs)

    p_show = sub.add_parser("show", help="dump a run's journal")
    add_root(p_show)
    p_show.add_argument("run", help="run id (see `rdw runs`)")
    p_show.add_argument("-v", "--verbose", action="store_true", help="include results/errors")
    p_show.set_defaults(func=cmd_show)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Console entry point (``rdw = rdw.cli:main``)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RdwError as exc:
        print(f"rdw: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
