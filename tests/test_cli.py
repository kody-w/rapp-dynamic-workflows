"""CLI happy path: `rdw run` then `rdw run --resume` replays from the journal.

The tmp workflow script swaps in its own fake runtime (the documented test
seam: any BaseRuntime subclass), so no Copilot client is ever constructed —
the autouse conftest guard would fail the test if one were.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdw import cli

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
