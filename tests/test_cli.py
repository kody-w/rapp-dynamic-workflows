"""CLI happy path: `rdw run` then `rdw run --resume` replays from the journal.

The tmp workflow script swaps in its own fake runtime (the documented test
seam: any BaseRuntime subclass), so no Copilot client is ever constructed —
the autouse conftest guard would fail the test if one were.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from rdw import cli
from rdw.errors import DivergenceWarning, RdwError, RdwWarning

SCRIPT_TEMPLATE = '''
from types import SimpleNamespace

from rdw.runtime import BaseRuntime

COUNTER = {counter!r}
RESULTS = {results!r}


class _Session:
    def __init__(self, n):
        self.session_id = f"cli-fake-{{n}}"

    def on(self, handler):
        return lambda: None

    async def send_and_wait(self, prompt, *, timeout=60.0):
        return SimpleNamespace(data=SimpleNamespace(content=f"echo:{{prompt}}"))

    async def abort(self):
        pass

    async def disconnect(self):
        pass


class _FakeRuntime(BaseRuntime):
    def __init__(self):
        super().__init__(2)
        self.n = 0

    async def create_session(self, **kwargs):
        self.n += 1
        with open(COUNTER, "a") as fh:
            fh.write("session\\n")
        return _Session(self.n)


async def workflow(wf):
    wf.runtime = _FakeRuntime()  # test seam: never touch the real client
    async with wf.phase("gather"):
        a = await wf.agent("alpha", label="a")
        b = await wf.agent("beta", label="b")
    wf.log("both done")
    with open(RESULTS, "a") as fh:
        fh.write(f"{{a}}|{{b}}\\n")
'''


@pytest.fixture
def cli_setup(tmp_path):
    counter = tmp_path / "sessions.log"
    results = tmp_path / "results.log"
    script = tmp_path / "wf_script.py"
    script.write_text(SCRIPT_TEMPLATE.format(counter=str(counter), results=str(results)))
    root = tmp_path / "rdw-root"
    return script, root, counter, results


def _session_count(counter: Path) -> int:
    return len(counter.read_text().splitlines()) if counter.exists() else 0


def test_cli_run_and_resume_happy_path(cli_setup, capsys):
    script, root, counter, results = cli_setup

    rc = cli.main(["--root", str(root), "run", str(script)])
    assert rc == 0
    assert _session_count(counter) == 2  # two live agents
    assert results.read_text() == "echo:alpha|echo:beta\n"

    run_dirs = list((root / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    run_id = run_dir.name

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["run_id"] == run_id
    assert Path(meta["script"]).name == "wf_script.py"

    out = capsys.readouterr().out
    assert f"run {run_id}" in out
    assert "0 cache hit(s)" in out

    # ---- resume: identical prompts, so both agents replay from the journal
    rc = cli.main(["--root", str(root), "run", str(script), "--resume", run_id])
    assert rc == 0
    assert _session_count(counter) == 2  # unchanged: zero new live sessions
    assert results.read_text().splitlines() == ["echo:alpha|echo:beta"] * 2

    out = capsys.readouterr().out
    assert "2 cache hit(s)" in out
    assert "DIVERGED" not in out


def test_cli_runs_and_show(cli_setup, capsys):
    script, root, counter, results = cli_setup
    assert cli.main(["--root", str(root), "run", str(script)]) == 0
    run_id = next((root / "runs").iterdir()).name
    capsys.readouterr()

    assert cli.main(["--root", str(root), "runs"]) == 0
    out = capsys.readouterr().out
    assert run_id in out and "2/2 agents ok" in out

    assert cli.main(["--root", str(root), "show", run_id]) == 0
    out = capsys.readouterr().out
    assert "[gather] a: ok" in out
    assert "both done" in out

    assert cli.main(["--root", str(root), "show", "no-such-run"]) == 1


def test_cli_root_flag_accepted_after_subcommand(cli_setup, capsys):
    """The README-documented form `rdw run script.py --root DIR` must work
    (argparse subparsers do not inherit parent optionals placed after the
    subcommand), for `runs` and `show` too."""
    script, root, counter, results = cli_setup

    assert cli.main(["run", str(script), "--root", str(root)]) == 0
    run_id = next((root / "runs").iterdir()).name
    capsys.readouterr()

    assert cli.main(["runs", "--root", str(root)]) == 0
    assert run_id in capsys.readouterr().out

    assert cli.main(["show", run_id, "--root", str(root)]) == 0
    assert "[gather] a: ok" in capsys.readouterr().out

    # both positions at once: the post-subcommand value wins
    assert cli.main(["--root", str(root / "ignored"), "runs", "--root", str(root)]) == 0
    assert run_id in capsys.readouterr().out


def test_cli_rejects_script_without_workflow_fn(tmp_path, capsys):
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n")
    rc = cli.main(["--root", str(tmp_path / "root"), "run", str(bad)])
    assert rc == 2
    assert "async def workflow" in capsys.readouterr().err


def test_cli_missing_script(tmp_path, capsys):
    rc = cli.main(["--root", str(tmp_path / "root"), "run", str(tmp_path / "nope.py")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


# --------------------------------------------------------------- run args


def test_parse_run_args_coercion_and_precedence(tmp_path):
    args_file = tmp_path / "args.json"
    args_file.write_text('{"n": 1, "name": "from-file", "keep": [1, 2]}')
    parsed = cli._parse_run_args(["n=3", "name=x", "flag=true"], str(args_file))
    assert parsed == {"n": 3, "name": "x", "flag": True, "keep": [1, 2]}
    # values are tried as JSON, falling back to plain strings
    assert cli._parse_run_args(["v=1.5"], None) == {"v": 1.5}
    assert cli._parse_run_args(["v=not json"], None) == {"v": "not json"}
    with pytest.raises(RdwError, match="KEY=VALUE"):
        cli._parse_run_args(["malformed"], None)
    with pytest.raises(RdwError, match="JSON object"):
        bad = tmp_path / "bad.json"
        bad.write_text("[1, 2]")
        cli._parse_run_args([], str(bad))


ARGS_SCRIPT = '''
from types import SimpleNamespace

from rdw.runtime import BaseRuntime

COUNTER = {counter!r}


class _Session:
    def __init__(self, n):
        self.session_id = f"cli-args-{{n}}"

    def on(self, handler):
        return lambda: None

    async def send_and_wait(self, prompt, *, timeout=60.0):
        return SimpleNamespace(data=SimpleNamespace(content=f"echo:{{prompt}}"))

    async def abort(self):
        pass

    async def disconnect(self):
        pass


class _FakeRuntime(BaseRuntime):
    def __init__(self):
        super().__init__(2)
        self.n = 0

    async def create_session(self, **kwargs):
        self.n += 1
        with open(COUNTER, "a") as fh:
            fh.write("session\\n")
        return _Session(self.n)


async def workflow(wf):
    wf.runtime = _FakeRuntime()  # test seam: never touch the real client
    await wf.agent(f"n is {{wf.args['n']}}", label="a")
'''


def test_cli_resume_reloads_args_from_meta(tmp_path):
    counter = tmp_path / "sessions.log"
    script = tmp_path / "args_wf.py"
    script.write_text(ARGS_SCRIPT.format(counter=str(counter)))
    root = tmp_path / "root"

    assert cli.main(["--root", str(root), "run", str(script), "--arg", "n=3"]) == 0
    assert _session_count(counter) == 1
    run_id = next((root / "runs").iterdir()).name
    meta = json.loads((root / "runs" / run_id / "meta.json").read_text())
    assert meta["attempts"][-1]["args"] == {"n": 3}

    # resume WITHOUT --arg: stored args reload, prompt matches, full replay
    assert cli.main(["--root", str(root), "run", str(script), "--resume", run_id]) == 0
    assert _session_count(counter) == 1  # zero new sessions

    # resume with DIFFERENT args: loud warning, fingerprints diverge, live run
    with pytest.warns(RdwWarning, match="different args"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DivergenceWarning)
            rc = cli.main(
                ["--root", str(root), "run", str(script), "--resume", run_id, "--arg", "n=4"]
            )
    assert rc == 0
    assert _session_count(counter) == 2  # went live under the new identity


# ------------------------------------------------------------ meta merging


def test_cli_meta_merges_attempts_and_preserves_created(cli_setup):
    script, root, counter, results = cli_setup

    assert cli.main(["--root", str(root), "run", str(script)]) == 0
    run_id = next((root / "runs").iterdir()).name
    meta_path = root / "runs" / run_id / "meta.json"
    first = json.loads(meta_path.read_text())
    assert len(first["attempts"]) == 1
    assert first["attempts"][0]["resume"] is False

    assert cli.main(["--root", str(root), "run", str(script), "--resume", run_id]) == 0
    second = json.loads(meta_path.read_text())
    assert second["created"] == first["created"]  # original stamp survives
    assert len(second["attempts"]) == 2
    assert second["attempts"][1]["resume"] is True
    assert second["run_id"] == run_id


# ------------------------------------------------------------------ PHASES


def test_cli_phases_declaration_lands_in_meta(cli_setup):
    script, root, counter, results = cli_setup
    script.write_text('PHASES = ["gather", "build"]\n' + script.read_text())

    assert cli.main(["--root", str(root), "run", str(script)]) == 0
    run_id = next((root / "runs").iterdir()).name
    meta = json.loads((root / "runs" / run_id / "meta.json").read_text())
    assert meta["phases"] == ["gather", "build"]


# ------------------------------------------------------------------ strict


def test_lint_nondeterminism_flags_wall_clock_and_rng(tmp_path):
    script = tmp_path / "dirty.py"
    script.write_text(
        "import time, random, uuid, datetime\n"
        "async def workflow(wf):\n"
        "    t = time.time()\n"
        "    d = datetime.datetime.now()\n"
        "    r = random.randint(1, 6)\n"
        "    u = uuid.uuid4()\n"
        "    ok = wf.now()  # sanctioned — not flagged\n"
    )
    flagged = cli._lint_nondeterminism(script)
    assert len(flagged) == 4
    assert any("time.time" in f for f in flagged)
    assert any("random.randint" in f for f in flagged)


def test_cli_strict_warns_on_nondeterminism(cli_setup):
    script, root, counter, results = cli_setup
    script.write_text(script.read_text() + "\nimport time\nSTAMP = time.time()\n")
    with pytest.warns(RdwWarning, match="wf.now"):
        assert cli.main(["--root", str(root), "run", str(script), "--strict"]) == 0


# ----------------------------------------------------------- show rendering


def test_cli_show_renders_boundary_refusal_and_value_lines(cli_setup, capsys):
    script, root, counter, results = cli_setup
    assert cli.main(["--root", str(root), "run", str(script)]) == 0
    run_id = next((root / "runs").iterdir()).name
    journal_path = root / "runs" / run_id / "journal.jsonl"
    with journal_path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "type": "refusal",
                    "index": 9,
                    "fp": "c" * 64,
                    "seq": 0,
                    "label": "late-agent",
                    "phase": None,
                    "budget": {"total": 40.0, "spent": 41.2, "outstanding": 0.0},
                    "ts": 0,
                }
            )
            + "\n"
        )
        fh.write(json.dumps({"type": "value", "kind": "now", "seq": 0, "value": 123.5}) + "\n")
    capsys.readouterr()

    assert cli.main(["--root", str(root), "show", run_id]) == 0
    out = capsys.readouterr().out
    assert "=== attempt 1 (start, budget unlimited) ===" in out
    assert "! refused at ceiling: late-agent (spent 41.20/40.00)" in out
    assert "~ now[0] = 123.5" in out

    # -v adds the per-agent request context
    assert cli.main(["--root", str(root), "show", run_id, "-v"]) == 0
    assert "request:" in capsys.readouterr().out
