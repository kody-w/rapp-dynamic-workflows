"""The ``rdw`` command line: run, list, and inspect workflow runs.

Commands:

* ``rdw run script.py [--resume RUN_ID] [--budget N] [--arg K=V] [...]`` —
  load a user script that exposes ``async def workflow(wf)`` and execute it
  inside a :class:`~rdw.engine.Workflow`. ``--resume`` replays journal-cached
  agent results and goes live at the first divergence.
* ``rdw runs`` — list runs under ``.rdw/runs`` (newest first).
* ``rdw show <run>`` — dump a run's journal in readable form.

Run parameters (``--arg key=value`` / ``--args-json file.json``) surface as
``wf.args`` and are part of run identity — the sanctioned channel for
run-scoped values like timestamps, replacing ``time.time()`` calls that would
silently bust the replay cache. On ``--resume`` the recorded args are reloaded
automatically so an identical resume needs no re-typing.

A workflow script is ordinary async Python (``PHASES`` is optional
pre-declaration for progress display)::

    # review.py
    from pydantic import BaseModel

    PHASES = ["review"]

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
import ast
import asyncio
import importlib.util
import inspect
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from .budget import Budget
from .engine import MAX_AGENTS_PER_RUN, MAX_WAVE_ITEMS, Workflow, new_run_id, phase_rollup_lines
from .errors import RdwError, RdwWarning
from .journal import AgentRecord, read_journal_lines


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


def _parse_run_args(pairs: list[str], json_file: str | None) -> dict[str, Any]:
    """Build the run-args dict from ``--args-json`` then ``--arg`` overrides.

    Each ``--arg`` value is tried as JSON first (``n=3`` → int 3,
    ``flags=[1,2]`` → list) and falls back to the raw string (``name=x`` →
    ``"x"``), so scalars round-trip without quoting gymnastics.
    """
    out: dict[str, Any] = {}
    if json_file:
        try:
            loaded = json.loads(Path(json_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RdwError(f"cannot load --args-json {json_file}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise RdwError(f"--args-json {json_file} must contain a JSON object")
        out.update(loaded)
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise RdwError(f"--arg expects KEY=VALUE, got {pair!r}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _stored_run_args(run_dir: Path) -> dict[str, Any] | None:
    """The args recorded by the run's most recent attempt, or ``None``."""
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    attempts = meta.get("attempts")
    if isinstance(attempts, list) and attempts:
        stored = attempts[-1].get("args")
        if isinstance(stored, dict):
            return stored
    return None


_NONDETERMINISM_SUSPECTS = {
    ("time", "time"),
    ("time", "time_ns"),
    ("time", "monotonic"),
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("datetime", "today"),
    ("uuid", "uuid1"),
    ("uuid", "uuid4"),
}


def _lint_nondeterminism(script: Path) -> list[str]:
    """Best-effort AST lint: flag wall-clock/randomness calls in a script.

    Walks the source for ``time.time()``, ``datetime.now()``, ``random.*()``
    and ``uuid.uuid4()``-style attribute calls — the sources that shift
    fingerprints and bust the replay cache. Honestly best-effort: it cannot
    see through aliases, wrappers, or ``from x import y`` call sites; it
    exists to point at ``wf.now()`` / ``wf.random()`` / ``wf.uuid()`` /
    ``--arg``, not to prove determinism.
    """
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    flagged: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        base = node.func.value
        base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
        if base_name == "random" or (base_name, node.func.attr) in _NONDETERMINISM_SUSPECTS:
            flagged.append(f"{script.name}:{node.lineno}: {base_name}.{node.func.attr}(...)")
    return flagged


def _write_meta(
    wf: Workflow,
    script: Path,
    budget: float | None,
    *,
    run_args: dict[str, Any] | None = None,
    resume: bool = False,
    phases: list[str] | None = None,
) -> None:
    """Merge run metadata into ``meta.json`` — never overwrite history.

    The original ``created`` stamp survives every re-invocation, and each
    attempt (fresh or ``--resume``) appends to an ``attempts`` list, so a
    thrice-retried run reads as three attempts instead of silently reflecting
    only the last one.
    """
    meta_path = wf.journal.run_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                meta = loaded
        except (OSError, json.JSONDecodeError):
            meta = {}  # unreadable prior meta: start fresh rather than crash
    now = time.time()
    meta.setdefault("created", now)
    meta["run_id"] = wf.run_id
    meta["script"] = str(script.resolve())
    meta["budget"] = budget
    meta["model"] = wf.default_model
    meta["effort"] = wf.default_effort
    if phases is not None:
        meta["phases"] = phases
    attempts = meta.setdefault("attempts", [])
    attempts.append(
        {
            "ts": now,
            "budget": budget,
            "model": wf.default_model,
            "effort": wf.default_effort,
            "resume": resume,
            "args": dict(run_args or {}),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


async def _drive(wf: Workflow, fn: Any) -> None:
    async with wf:
        await fn(wf)


def cmd_run(args: argparse.Namespace) -> int:
    script = Path(args.script)
    fn = _load_workflow_fn(script)
    if args.strict:
        flagged = _lint_nondeterminism(script)
        if flagged:
            warnings.warn(
                "script calls nondeterministic sources that will shift "
                "fingerprints across runs — prefer wf.now()/wf.random()/"
                "wf.uuid() (journaled) or --arg for run-scoped values:\n  "
                + "\n  ".join(flagged),
                RdwWarning,
                stacklevel=2,
            )
    run_args = _parse_run_args(args.arg, args.args_json)
    run_id = args.resume or new_run_id()
    if args.resume:
        stored = _stored_run_args(Path(args.root) / "runs" / run_id)
        if not args.arg and not args.args_json:
            # No args supplied: reuse the recorded ones so an identical
            # resume replays without re-typing the invocation.
            run_args = stored or {}
        elif stored is not None and stored != run_args:
            warnings.warn(
                f"--resume {run_id} with different args than recorded "
                f"({stored!r} -> {run_args!r}); args are part of run identity, "
                "so fingerprints will diverge and cached results won't replay",
                RdwWarning,
                stacklevel=2,
            )
    phases_decl = fn.__globals__.get("PHASES")
    phases = (
        [str(p) for p in phases_decl]
        if isinstance(phases_decl, (list, tuple)) and phases_decl
        else None
    )
    wf = Workflow.open(
        run_id=run_id,
        root=args.root,
        resume=bool(args.resume),
        budget=Budget(total=args.budget),
        model=args.model,
        effort=args.effort,
        cwd=args.cwd,
        concurrency=args.concurrency,
        transcripts=args.transcripts,
        args=run_args,
        max_agents=args.max_agents,
        max_wave=args.max_wave,
    )
    if phases:
        wf.declare_phases(phases)
    _write_meta(
        wf, script, args.budget, run_args=run_args, resume=bool(args.resume), phases=phases
    )
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


def _effective_agent_records(lines: list[dict[str, Any]]) -> list[AgentRecord]:
    """Reduce raw journal lines to effective agent records for stats.

    Mirrors ``Journal._load``: the last record per ``(fp, seq)`` wins (retries
    supersede failures), ordered by display index. Unparseable lines are
    skipped — ``show`` must render damaged journals, not crash on them.
    """
    effective: dict[tuple[str, int], AgentRecord] = {}
    for obj in lines:
        if obj.get("type") != "agent":
            continue
        try:
            rec = AgentRecord.from_obj(obj)
        except (KeyError, TypeError, ValueError):
            continue
        effective[rec.key] = rec
    return sorted(effective.values(), key=lambda r: (r.index, r.started))


def _show_stats_lines(lines: list[dict[str, Any]]) -> list[str]:
    """``rdw show --stats`` body: run totals plus per-phase rollups.

    The rollup arithmetic lives in ``engine.phase_rollup_lines`` (shared with
    ``wf.report()``); this helper only adds the run-total header with the
    model/tool-call counters from journaled usage telemetry.
    """
    records = _effective_agent_records(lines)
    if not records:
        return ["(no agent records)"]
    ok = sum(1 for r in records if r.status == "ok")
    credits = sum(r.credits for r in records)
    wall = sum(max(0.0, r.ended - r.started) for r in records)
    counters = {"input_tokens": 0, "output_tokens": 0, "model_calls": 0, "tool_calls": 0}
    for rec in records:
        usage = rec.usage or {}
        for key in counters:
            counters[key] += int(usage.get(key) or 0)
    total = f"total: {len(records)} agent(s) ({ok} ok), {credits:.2f} AIU"
    tokens = counters["input_tokens"] + counters["output_tokens"]
    if tokens:
        total += (
            f", {tokens} tok ({counters['input_tokens']} in / {counters['output_tokens']} out)"
        )
    if counters["model_calls"]:
        total += f", {counters['model_calls']} model call(s)"
    if counters["tool_calls"]:
        total += f", {counters['tool_calls']} tool call(s)"
    total += f", {wall:.1f}s wall"
    return [total, *phase_rollup_lines(records)]


def cmd_show(args: argparse.Namespace) -> int:
    run_dir = Path(args.root) / "runs" / args.run
    if not run_dir.is_dir():
        print(f"no such run: {args.run}", file=sys.stderr)
        return 1
    lines = read_journal_lines(run_dir)
    if not lines:
        print("(empty journal)")
        return 0
    if args.stats:
        for line in _show_stats_lines(lines):
            print(line)
        return 0
    attempt = 0
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
                if obj.get("request") is not None:
                    print(
                        "    request: "
                        + json.dumps(obj["request"], ensure_ascii=False)[:2000]
                    )
        elif kind == "boundary":
            attempt += 1
            info = obj.get("info") or {}
            budget_total = info.get("budget_total")
            budget_txt = f"{budget_total}" if budget_total is not None else "unlimited"
            print(f"=== attempt {attempt} ({obj.get('event')}, budget {budget_txt}) ===")
        elif kind == "refusal":
            b = obj.get("budget") or {}
            spent = float(b.get("spent") or 0.0)
            total = b.get("total")
            cap = f"{float(total):.2f}" if total is not None else "∞"
            print(f"! refused at ceiling: {obj.get('label')} (spent {spent:.2f}/{cap})")
        elif kind == "value":
            print(f"~ {obj.get('kind')}[{obj.get('seq')}] = {obj.get('value')}")
        elif kind == "divergence":
            where = (
                f"value {obj['kind']}[{obj.get('seq')}]"
                if obj.get("kind")
                else f"position {obj.get('index')}"
            )
            print(f"! divergence at {where} — live from here")
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
    p_run.add_argument(
        "--transcripts",
        action="store_true",
        help="write per-agent session transcripts to <run-dir>/agents/*.jsonl",
    )
    p_run.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="run parameter (repeatable); surfaces as wf.args and is part of "
        "run identity — the sanctioned channel for timestamps etc.",
    )
    p_run.add_argument(
        "--args-json",
        metavar="FILE",
        help="JSON object file of run parameters (individual --arg overrides win)",
    )
    p_run.add_argument(
        "--max-agents",
        type=int,
        default=MAX_AGENTS_PER_RUN,
        help=f"lifetime cap on agent calls this run (default {MAX_AGENTS_PER_RUN})",
    )
    p_run.add_argument(
        "--max-wave",
        type=int,
        default=MAX_WAVE_ITEMS,
        help=f"cap on items per parallel/pipeline wave (default {MAX_WAVE_ITEMS})",
    )
    p_run.add_argument(
        "--strict",
        action="store_true",
        help="best-effort lint: warn on time/random/uuid calls in the script "
        "(prefer wf.now()/wf.random()/wf.uuid() or --arg)",
    )
    p_run.set_defaults(func=cmd_run)

    p_runs = sub.add_parser("runs", help="list recorded runs")
    add_root(p_runs)
    p_runs.set_defaults(func=cmd_runs)

    p_show = sub.add_parser("show", help="dump a run's journal")
    add_root(p_show)
    p_show.add_argument("run", help="run id (see `rdw runs`)")
    p_show.add_argument("-v", "--verbose", action="store_true", help="include results/errors")
    p_show.add_argument(
        "--stats",
        action="store_true",
        help="per-phase token/credit rollups instead of the journal dump",
    )
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
