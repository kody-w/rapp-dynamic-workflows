# Contributing

Thanks for helping build rapp-dynamic-workflows. The bar for merging is
simple: every SDK claim verified against real source, and **zero live
inference in the test suite**.

## The prime directive: tests never burn credits

Copilot sessions cost real AI credits against the logged-in account, and CI has
no account at all. Therefore:

- **No test may create a real `CopilotClient` or spawn the `copilot` binary.**
  The entire suite runs against fake runtimes.
- **No test may import-time trigger a child process.** `rdw` imports `copilot`
  lazily inside `CopilotRuntime` for exactly this reason; keep it that way. A
  regression here is a bug even if the tests pass.
- **Live examples are opt-in only.** Anything that performs real inference
  lives under `examples/` and must guard itself:

  ```python
  import os, sys
  if os.environ.get("RDW_LIVE") != "1":
      sys.exit("live example: set RDW_LIVE=1 to run (spends real AI credits)")
  ```

  CI never sets `RDW_LIVE`. Reviewers will reject any test — unit, integration,
  or doctest — that requires it.

### How to test without a login

The seam is `rdw.BaseRuntime`: subclass it, implement `create_session`, and
return an object satisfying the `SessionHandle` protocol (`session_id`,
`send_and_wait`, `on`, `abort`, `disconnect`). A minimal fake:

```python
from rdw import BaseRuntime

class FakeSession:
    def __init__(self, sid, script):
        self.session_id = sid
        self._script = script          # callable(prompt) -> event or action
        self._handlers = []

    async def send_and_wait(self, prompt, *, timeout=60.0):
        return self._script(prompt, self._handlers)

    def on(self, handler):
        self._handlers.append(handler)
        return lambda: self._handlers.remove(handler)

    async def abort(self): ...
    async def disconnect(self): ...

class FakeRuntime(BaseRuntime):
    def __init__(self, script, concurrency=None):
        super().__init__(concurrency)
        self._script = script
        self.sessions = 0

    async def create_session(self, **kwargs):
        self.sessions += 1
        return FakeSession(f"s-{self.sessions}", self._script)
```

Wire it into a workflow explicitly — no run directory magic needed:

```python
from pathlib import Path
from rdw import Workflow, Budget, Journal, Progress

wf = Workflow(
    run_id="test",
    runtime=FakeRuntime(script),
    budget=Budget(total=10.0),
    journal=Journal(Path(tmp_path)),
    progress=Progress(force_plain=True),
)
async with wf:
    result = await wf.agent("hello", label="a")
```

What fakes should simulate (all of these have observable engine behavior worth
asserting):

| Behavior under test | How the fake simulates it |
|---|---|
| Structured output | Find the `submit_result` tool in `kwargs["tools"]`, call its `handler` with a fake `ToolInvocation`-shaped object carrying `arguments` |
| Validation retry | Call the handler with invalid args first (assert the failure `ToolResult`), then valid args |
| Nudge ladder / stonewall | Never call the handler; count `send_and_wait` invocations (1 initial + 2 nudges), expect `AgentSchemaError` |
| Timeout + abort | `await asyncio.sleep(...)` past the timeout; record that `abort()` was called |
| Budget accounting | Fire fake events at subscribed handlers: objects with `.type` (string or `.value`-bearing) of `assistant.usage` / `session.usage_checkpoint` and matching `.data` shapes (`data.copilot_usage.total_nano_aiu` / `data.total_nano_aiu`) |
| Concurrency cap | Track simultaneous in-flight `send_and_wait` calls, assert ≤ the semaphore size |

Journal, schema, budget, patterns, and CLI (`runs`/`show`) tests need no
runtime at all — they are pure functions over files and objects.

## Verifying SDK claims

The #1 project risk is hallucinated SDK APIs. Any change that touches the
`copilot` package surface (currently confined to `rdw/runtime.py` and
`rdw/schema.py`) must cite the real signature:

1. Read the installed source —
   `python -c "import copilot, pathlib; print(pathlib.Path(copilot.__file__).parent)"` —
   specifically `client.py` (`CopilotClient.__init__`, `create_session`,
   `resume_session`), `session.py` (`send_and_wait`, `on`, `abort`,
   `disconnect`), and `tools.py` (`Tool`, `ToolResult`, `ToolInvocation`).
2. Reference the verification in your PR description (file and symbol, e.g.
   "`create_session` kwarg `session_limits` — client.py").
3. Never widen the `SessionHandle`/`Runtime` protocols beyond what the engine
   actually calls; the protocols are the documented contract fakes implement.

Docs count too: README/architecture statements about SDK behavior follow the
same rule.

## Development setup

```console
$ git clone https://github.com/kody-w/rapp-dynamic-workflows
$ cd rapp-dynamic-workflows
$ python -m venv .venv && . .venv/bin/activate
$ pip install -e ".[test]"
$ pytest          # fast, offline, no copilot login required
$ ruff check .    # line length 100
```

`pytest` is configured with `asyncio_mode = "auto"` — write `async def` tests
directly, no decorator needed.

## Ground rules

- Python 3.11+. Dependencies stay at `github-copilot-sdk`, `pydantic`, `rich` —
  propose anything heavier in an issue first.
- `rich` imports must stay guarded (`Progress` runs plain-mode without it);
  `copilot` imports must stay lazy.
- Failure semantics are API: `parallel` branches resolve to `None` and never
  raise; `pipeline` drops items to `None`; `journal.jsonl` stays append-only.
  Changes to these are breaking changes.
- New failure modes get a typed error in `rdw/errors.py`, not string matching.
- Public functions carry docstrings in the existing style (Args/Returns/Raises);
  honest-limitation notes ("this can overshoot by one step") are part of the
  house voice — keep them.
- Don't commit `.rdw/` run directories; the repo `.gitignore` already excludes
  them.

## Pull requests

Keep PRs focused. Include: what changed, the SDK-source citations for any
surface changes, and test coverage via fakes. If your change affects resume or
budget semantics, add a journal-level test showing the old and new behavior —
those two subsystems are where silent regressions hurt users most.
