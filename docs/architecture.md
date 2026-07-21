# Architecture

How rapp-dynamic-workflows is put together, and why. Every SDK claim in this
document is verified against the installed `github-copilot-sdk` 1.0.7 source
(`copilot/client.py`, `copilot/session.py`, `copilot/tools.py`).

## Contents

- [Design goals](#design-goals)
- [Process model](#process-model)
- [Module map](#module-map)
- [The runtime layer](#the-runtime-layer)
- [Anatomy of one `agent()` call](#anatomy-of-one-agent-call)
- [The session-event model](#the-session-event-model)
- [Schema forcing internals](#schema-forcing-internals)
- [Budget accounting](#budget-accounting)
- [Journal internals](#journal-internals)
- [Progress rendering](#progress-rendering)
- [Failure taxonomy](#failure-taxonomy)
- [Known limits](#known-limits)

## Design goals

1. **Zero-token orchestration.** Control flow is plain Python. The model never
   decides which agent runs next, never blocks a synchronous tool call waiting
   for a subagent, and never spends a token on coordination. This is the core
   property borrowed from Claude Code's Workflow tool.
2. **Hermetic agents.** Each `agent()` call is a fresh session that is a
   deterministic function of (prompt, options): isolated system prompt, no
   custom instructions, no shared conversation state, its own model/effort/cwd
   and credit cap. Hermeticity is what makes fingerprinted replay *meaningful* —
   a cached result is valid because nothing ambient could have changed it.
3. **Degrade, don't crash.** A failed branch is `None`, a capped wave shrinks,
   a diverged journal goes live from the divergence point. The only errors that
   propagate out of the combinators are cancellation and bugs.
4. **Honest guarantees.** Where the substrate can't support a Claude-Workflow
   guarantee (constrained decoding, enforced determinism, real-time budget
   cutoffs), the docs say so and the design converts silent failure into loud,
   typed failure.

## Process model

```
your script (python)                      copilot --headless --stdio (node)
┌─────────────────────────┐   JSON-RPC    ┌──────────────────────────────┐
│ Workflow                │◄────────────► │ session strategy-1           │
│  ├ Budget (AIU taps)    │   over stdio  │ session strategy-2           │
│  ├ Journal (jsonl)      │               │ session reviewer             │
│  ├ Progress (rich/plain)│               │ ... (one per live agent)     │
│  └ CopilotRuntime ──────┼── spawns ────►│                              │
└─────────────────────────┘  (lazily, ×1) └──────────────────────────────┘
```

- **One client, N sessions.** `CopilotClient` spawns a single
  `copilot --headless --stdio` child and multiplexes every concurrent session
  over it (the SDK keeps `self._sessions: dict[str, CopilotSession]`). Per-agent
  client processes would pay seconds of node startup per agent; sessions on a
  shared runtime are cheap, which is what makes 8-way fan-outs a one-liner.
- **Lazy spawn.** The client is created and `start()`ed on the first
  `create_session` only. A resumed run that serves everything from the journal,
  or a `rdw runs`/`rdw show` invocation, never spawns a child process at all.
  (`import copilot` itself is deferred into the runtime for the same reason.)
- **Auth is inherited.** The SDK drives the already-logged-in `copilot` binary.
  There are no API keys anywhere in rdw.
- **Single point of failure, acknowledged.** If the shared runtime dies, every
  in-flight session dies with it. The journal bounds the blast radius: finished
  calls are already persisted, so the recovery story is "resume the run".

## Module map

| Module | Responsibility | Key exports |
|---|---|---|
| `rdw/engine.py` | The Workflow: `agent`/`parallel`/`pipeline`/`phase`/`log`/`report`, journaled `now`/`random`/`uuid`, args identity, safety caps, ContextVar binding for the module-level API | `Workflow`, `current_workflow`, `new_run_id`, `MAX_AGENTS_PER_RUN`, `MAX_WAVE_ITEMS` |
| `rdw/runtime.py` | Session factory protocol, shared concurrency semaphore, lazy singleton `CopilotClient`, permission defaulting | `Runtime`, `SessionHandle`, `BaseRuntime`, `CopilotRuntime` |
| `rdw/schema.py` | Pydantic/dict → `submit_result` Tool compilation, submit instruction and nudge text, schema fingerprints, journal (de)serialization | `build_submit_tool`, `SubmitCapture`, `SchemaSpec`, `schema_fingerprint`, `dump_value`/`load_value` |
| `rdw/journal.py` | Append-only `journal.jsonl`, fingerprints, replay cache, divergence handling | `Journal`, `AgentRecord`, `fingerprint` |
| `rdw/budget.py` | AIU accounting from session events, admission gate, per-session `session_limits` | `Budget` |
| `rdw/patterns.py` | Quality patterns built purely on the public API | `adversarial_verify`, `judge_panel`, `loop_until_dry`, `loop_until_budget` |
| `rdw/progress.py` | Live rich tree (TTY) / plain lines (CI) with liveness heartbeat, thread-safe | `Progress`, `fmt_tokens` |
| `rdw/transcripts.py` | Opt-in observability taps: filtered per-agent transcript writer, token/tool usage accumulator | `TranscriptWriter`, `UsageTap` |
| `rdw/cli.py` | `rdw run` / `runs` / `show`, script loading, `meta.json` | `main` |
| `rdw/errors.py` | Typed failure taxonomy | see [Failure taxonomy](#failure-taxonomy) |

Dependency direction is strictly downward: `patterns` → `engine` →
(`runtime`, `schema`, `journal`, `budget`, `progress`) → `errors`. Only
`runtime.py` and `schema.py` import from the `copilot` package, and `runtime.py`
does so lazily.

## The runtime layer

`Runtime` is a three-method protocol — `create_session(**kwargs)`, `slot()`,
`close()` — so everything above it is testable against a fake.

**`BaseRuntime`** owns the concurrency semaphore. The default cap is
`min(16, cpu_count - 2)` (floor 1): enough to make fan-outs real, conservative
enough not to invite the quota-burning failure mode where an unbounded wave
saturates the account. The semaphore is created lazily so a runtime can be
constructed outside an event loop (CLI argument-parsing time). `agent()`
acquires a slot only for *live* sessions — journal replays bypass it.

**`CopilotRuntime`** adds the lazy singleton client (double-checked under an
`asyncio.Lock`), a `client_factory` test seam, and one important default:

> **Permissions are resolved before the first turn.** Headless sessions
> otherwise fail their first tool call with
> `denied-no-approval-rule-and-could-not-request-from-user` — an error observed
> in production session logs. Unless the caller passes their own
> `on_permission_request`, rdw installs `PermissionHandler.approve_all`.
> Schema-only agents are still effectively read-only because their
> `available_tools` is narrowed to `submit_result`; agents you hand real tools
> to should get a narrower policy via a runtime subclass.

**Fakes** subclass `BaseRuntime` and implement `create_session` to return an
object satisfying `SessionHandle` (`session_id`, `send_and_wait`, `on`,
`abort`, `disconnect`). This is the seam the entire test suite runs through —
no login, no network, no child process.

## Anatomy of one `agent()` call

```
agent(prompt, schema=Model, ...)
│
├─ 1. index = journal.next_index()            # display position only
├─ 2. fp = sha256(prompt, opts)               # opts: schema hash, model,
│     seq = journal.next_occurrence(fp)       #   effort, tool names, cwd;
│                                             # seq = Nth call with this fp
├─ 3. journal.lookup(fp, seq)
│      ├─ ok record       → load_value() → return   (no session, no credits)
│      ├─ error record    → fall through (retry, not divergence)
│      └─ miss (cached records still unreplayed)
│                         → DivergenceWarning, marker line, live from here
├─ 4. budget.ensure_available()               # fail-fast; BudgetExceeded
├─ 5. async with runtime.slot():              # concurrency cap
│      ├─ grant = budget.reserve()            # half the uncommitted remainder;
│      │                                      #   may raise BudgetExceeded
│      ├─ create_session(
│      │     model, reasoning_effort, working_directory,
│      │     tools=[..., submit_result],      # when schema
│      │     system_message={"mode":"append", ...},
│      │     available_tools=["submit_result"],   # schema + no user tools
│      │     session_limits={"max_ai_credits": grant},
│      │     skip_custom_instructions=True,
│      │     include_sub_agent_streaming_events=False)
│      ├─ session.on(budget.tap(session_id))  # AIU accounting
│      ├─ session.on(progress tap)            # output-token ticks
│      ├─ send_and_wait(prompt, timeout=...)
│      │     └─ on TimeoutError: session.abort() first, then AgentTimeout
│      ├─ schema path: nudge up to 2× if idle without submit,
│      │               else AgentSchemaError
│      └─ finally: unsubscribe taps, session.disconnect(), grant.release()
├─ 6. journal.record(AgentRecord(status, result|error, session_id,
│                                 credits, started, ended))
└─ 7. return validated instance | text
```

Two details worth calling out:

- **Abort-on-timeout.** The raw SDK `send_and_wait` raises `TimeoutError`
  *without* aborting the in-flight turn — the model keeps generating (and
  billing) server-side. rdw always calls `session.abort()` before raising
  `AgentTimeout`, so a timed-out agent stops spending immediately.
- **Errors are journaled too.** An `AgentError` writes an `error` record
  (with session_id and spend attribution) before propagating. On resume, a
  matching error record re-executes live rather than replaying the failure —
  and the recorded `session_id` leaves the door open for a future
  `resume_session(session_id, ...)` reattach of half-finished work.

## The session-event model

Everything rdw observes about a running agent arrives through
`session.on(handler)` push subscriptions — the design deliberately **never
parses the on-disk `events.jsonl`** session logs, which grow to 50–150 MB on
long sessions and interleave base64 `session.binary_asset` blobs.

Facts about the event stream, established from both the SDK's generated event
types and multi-day production session logs:

- **Envelope.** Every event is `{type, data, id, timestamp, parentId}`;
  `parentId` chains each event to its predecessor, forming a causal linked
  list. Subagent-scoped events add a top-level `agentId`.
- **Turn anatomy.** `user.message` → `assistant.turn_start` → N model steps,
  each an `assistant.message` carrying `content` and a `toolRequests[]` batch
  (multiple tools are commonly requested in one step) → paired
  `tool.execution_start`/`tool.execution_complete` per request → a final
  `assistant.message` with empty `toolRequests` → `assistant.turn_end`; at full
  idle, `session.usage_checkpoint` fires.
- **The two usage channels.** `assistant.usage` events carry per-model-call
  cost (`data.copilot_usage.total_nano_aiu`); `session.usage_checkpoint`
  carries *cumulative* session cost (`data.total_nano_aiu`, at idle). rdw's
  `Budget.tap()` consumes both and takes
  `max(checkpoint, Σ per-call deltas)` per session, which is robust to either
  channel lagging or missing and can never double-count.
- **Event volume control.** Sessions are created with
  `include_sub_agent_streaming_events=False`; rdw agents are leaf sessions and
  should never fan out model-driven subagents of their own.
- **Handler discipline.** SDK event handlers may fire on the client's receive
  thread, not the asyncio loop. Every tap (budget, progress, usage, and the
  opt-in transcript writer) is synchronous, cheap, lock-guarded, and wrapped so
  that a malformed event — or a full disk, in the transcript's case — can never
  raise into the SDK's dispatch loop. `Budget`, `Progress`, `UsageTap`, and
  `TranscriptWriter` are thread-safe for the same reason.

Consequence: `send_and_wait` resolving (the `session.idle` condition) can race
the final usage events by a moment. Spend attribution recorded in the journal
is best-effort-at-completion; run-level totals catch up as checkpoints land.

## Schema forcing internals

See the README for the user-facing story. Implementation notes:

- `build_submit_tool` constructs a `copilot.tools.Tool` directly (name
  `submit_result`, `parameters=Model.model_json_schema()` or the raw dict,
  `skip_permission=True`) rather than going through the `@define_tool`
  decorator — same wire shape, but the handler closes over a per-call
  `SubmitCapture` cell, which a module-level decorated function couldn't do.
- The Pydantic handler replicates the SDK's own invalid-arguments contract: on
  `ValidationError` it returns a `ToolResult` with `result_type="failure"` and
  `text_result_for_llm="Invalid tool arguments:\n<field: msg lines>"`. To the
  model this is indistinguishable from the SDK's native validation bounce, so
  the free in-band retry loop behaves identically.
- Raw-dict schemas get the only validation possible without adding a JSON-schema
  validator dependency: argument-is-object and top-level `required` keys.
  Anything richer should use a Pydantic model.
- `SubmitCapture.called` (not `value is not None`) is the completion signal, so
  a model legitimately submitting empty/falsy content is not re-nudged.
- The success `ToolResult` tells the model "Result recorded. Your task is
  complete; end your turn." — closing the loop matters; without it some models
  keep narrating and re-submitting.
- On journal replay, `load_value` re-validates the stored payload against the
  current model class. Fingerprints already include the schema hash, so a
  schema edit normally misses the cache first — re-validation is the belt to
  that suspender.

## Budget accounting

`Budget` is a small state machine over per-session accumulators:

```
_SessionSpend(checkpoint, usage_sum)   # both in nano-AIU
    .nano = max(checkpoint, usage_sum)
Budget.spent() = Σ sessions .nano / 1e9
```

- **Admission gate with reservations.** Before every *live* agent start the
  engine calls `budget.reserve()`, which admits against the **uncommitted**
  remainder: `total − spent − Σ outstanding grants`. In-flight sessions count
  before their usage events land, so a concurrent wave near the ceiling is
  refused rather than blindly admitted (`BudgetExceeded(spent, total, label)`).
  `ensure_available()` remains as a cheap fail-fast using the same math.
  Replays skip both — they cost nothing. Grants are released (and replaced by
  the session's actual accounted spend) when the session finishes.
- **Per-session cap.** Each admitted session's
  `session_limits={"max_ai_credits": grant}` is **half the uncommitted
  remainder** at admission, so concurrent grants shrink geometrically
  (½, ¼, ⅛ … of remaining) and always sum to less than the remaining budget —
  worst-case spend is bounded by `total`, never `total + N × remaining`. A
  small floor keeps the limit positive at the boundary, and admission refuses
  outright below 0.01 uncommitted credits so the ceiling can't be approached
  by an endless tail of dust-capped sessions. The SDK marks `session_limits`
  Experimental; rdw treats it as defense in depth and never depends on it for
  correctness.
- **Wave semantics.** `parallel()` absorbs `BudgetExceeded` like any other
  branch failure: branches admitted before the ceiling finish and are
  journaled; branches refused after it resolve to `None` (and leave a
  `refusal` journal line). Budget exhaustion degrades a wave rather than
  aborting the run. The deliberate asymmetry: `AgentLimitExceeded` (the
  `max_agents` runaway cap) *propagates* out of `parallel()`/`pipeline()` —
  a run-level misconfiguration must crash loudly, not degrade to `None`s.
- **Safety caps.** Orthogonal to credits: `max_agents` (default 1000) bounds
  total `agent()` calls per run — cached replays count, since the cap bounds
  calls, not spend — and `max_wave` (default 4096) rejects oversized
  `parallel`/`pipeline` waves with `ValueError` before any branch runs. They
  exist because the default budget is unlimited-with-accounting: a bugged
  `while` loop without `--budget` now fails fast instead of billing until
  killed.
- **Attribution.** `session_spent(session_id)` prices a single agent; the
  journal records it per call, which is what `wf.report()`, `rdw runs`, and
  `rdw show` aggregate.

## Journal internals

`journal.jsonl` is genuinely append-only — no line is ever rewritten. Six
line types:

```jsonc
{"type": "agent", "index": 4, "fp": "sha256…", "seq": 0, "label": "strategy-2",
 "phase": "design", "status": "ok", "result": {"kind": "model", "value": {…}},
 "session_id": "…", "credits": 1.73, "started": …, "ended": …,
 "request": {"model": …, "effort": …, "tools": [], "timeout": 600.0,
             "prompt_chars": 1834, "session_limits": {"max_ai_credits": 30.0},
             "budget": {"total": 40.0, "spent": 1.5, "outstanding": 19.25}}}
{"type": "value", "kind": "now", "seq": 0, "value": 1784736000.5}
{"type": "refusal", "index": 7, "fp": "sha256…", "seq": 0, "label": "late-agent",
 "phase": "build", "budget": {"total": 40.0, "spent": 41.2, "outstanding": 0.0}, "ts": …}
{"type": "boundary", "event": "resume", "info": {"ts": …, "pid": …,
 "budget_total": 40.0, "model": …, "effort": …, "rdw_version": "0.1.0",
 "cache_records_loaded": 12}}
{"type": "divergence", "index": 4, "fp": "sha256…", "ts": …}
{"type": "log", "message": "…", "phase": "design", "ts": …}
```

Only `agent` and `value` lines participate in replay. `request` is forensic
context attached to both ok and error records (the error path is the one that
pays for it) and is deliberately excluded from the fingerprint. `refusal`
lines record budget-ceiling refusals with the exact snapshot that refused
them — never replayable, so a retry under a raised budget runs live.
`boundary` lines bracket each process attempt (`rdw show` renders them as
`=== attempt N … ===`). `value` lines are `wf.now()`/`wf.random()`/`wf.uuid()`
results, keyed `(kind, occurrence)` exactly like agent records and sharing the
same pending/divergence machinery.

Loading is event-sourced replay: lines are applied in order and the last
`agent` record per `(fp, seq)` key wins. This gives three properties at once:

1. **Crash safety** — records land as each call completes (flushed and
   fsynced); a killed run keeps every finished branch. A torn *final* line
   (crash mid-append) is skipped with a `JournalWarning` on the next resume —
   the crash the journal exists to recover from can never permanently disable
   `--resume` — and the next append repairs a missing trailing newline so two
   records can never merge. Interior corruption still raises `JournalError`.
2. **History** — every generation of the run remains on disk and `rdw show`
   renders all of it, divergences included.
3. **Correct multi-resume** — after a divergence, the live re-runs append
   records under the same or new keys; the *next* resume's last-wins load
   replays the newest results, not the superseded ones.

Fingerprints hash `(prompt, normalized opts)` — **content, not position**. The
Nth call with a given fingerprint claims occurrence number N, so repeated
identical prompts each keep their own slot. Positional keying was deliberately
rejected: under `parallel()`/`pipeline()` the order agents *start* depends on
live session latency, so a global call-order index diverges an *identical*
resumed run (and cached replays, which complete synchronously, reshuffle
ordering further). Content-addressed keys are sound because sessions are
hermetic — a record is a function of (prompt, opts), not of when it ran.
Normalizing opts to result-affecting fields only (schema hash, model, effort,
sorted tool names, cwd) means cosmetic edits — labels, timeouts, log lines —
never bust the cache. `resume=False` runs still append, so any run can be
resumed later by id.

Divergence means: a lookup missed while unreplayed cached records remain — the
script changed relative to the journal. It is detected once, marked once,
warned once (`DivergenceWarning`, a `UserWarning` subclass — visible by
default, promotable to an error with
`warnings.simplefilter("error", rdw.DivergenceWarning)` for CI-grade
strictness), and then the run simply executes live from that call onward. A
miss *after* the cache is fully consumed is just new work appended to the
script and does not diverge.

## Progress rendering

Two renderers behind one thread-safe `Progress` API:

- **rich tree** (TTY and `rich` importable): a `Live` display re-rendered from
  state at 4 fps — run header with live budget summary, phases as branches,
  agents as leaves with status glyph, output-token counter (fed by
  `assistant.usage` events), elapsed seconds and a `last: <tool> <n>s ago`
  activity marker for running agents (fed by `tool.execution_start`), and
  result detail.
- **plain lines** (non-TTY, `rich` missing, or `force_plain=True`): one line
  per state transition, grep-able, with token ticks deliberately suppressed so
  CI logs don't flood.

Plain mode additionally runs a **heartbeat**: an asyncio task (started in
`Progress.start`, cancelled in `Progress.stop`) prints one bounded summary
line every 30 s while agents are running — `· 3 running: strategy-1 84s/8.2k
last: bash 3s ago, …`, capped at four agents plus `+N more`. Nothing prints
when nothing is running. Rationale: with `DEFAULT_TIMEOUT=600` a wedged agent
is otherwise indistinguishable from a working one for ten minutes of silent
piped log.

`rich` is a declared dependency but imports are guarded — the package imports
and runs plain-mode even where rich is absent.

## Transcripts & usage telemetry

Both live in `rdw/transcripts.py` and subscribe next to the budget/progress
taps in `_run_session` — same delivery channel, zero extra SDK calls:

- **`UsageTap`** (always on): accumulates `input_tokens` / `output_tokens` /
  `cache_read_tokens`, `model_calls` (one per `assistant.usage`), and
  `tool_calls` (one per `tool.execution_start`). The engine snapshots it onto
  `AgentRecord.usage` for **both** ok and error records — the error path is
  the one that pays for telemetry. `usage` is serialized in the journal line
  but excluded from the fingerprint, and an empty snapshot serializes as
  `null`, so records stay compatible in both directions. `wf.report()` and
  `rdw show --stats` share one rollup implementation
  (`engine.phase_rollup_lines`).
- **`TranscriptWriter`** (opt-in via `Workflow.open(transcripts=True)` /
  `rdw run --transcripts`): writes `agents/<index>-<label>.jsonl` in the run
  dir, keeping only `assistant.message`, `tool.execution_start` /
  `tool.execution_complete` (arguments/results truncated at 2000 chars),
  `assistant.usage`, and `session.error`. Streaming deltas and binary assets
  are filtered *at the tap* — the design never parses the CLI's own 50–150 MB
  `events.jsonl`. File and directory creation are lazy (no kept events → no
  file), the label is sanitized before becoming a path segment, and the
  relative transcript path is recorded in the agent's journaled `request`
  context so `rdw show -v` points at it. Write failures are swallowed:
  transcripts are best-effort forensics, never a source of run failure.

## Failure taxonomy

| Error | Raised when | `parallel()` behavior |
|---|---|---|
| `AgentError` | Session failed (transport, session.error, unexpected) | → `None` |
| `AgentTimeout` (⊂ AgentError) | Turn exceeded `timeout`; session was aborted first | → `None` |
| `AgentSchemaError` (⊂ AgentError) | Idle without `submit_result` after the nudge ladder | → `None` |
| `BudgetExceeded` | Admission gate refused a new agent (a `refusal` line is journaled) | → `None` (wave degrades) |
| `AgentLimitExceeded` | The run hit its `max_agents` lifetime cap | **propagates** (run-level misconfiguration) |
| `JournalError` | journal.jsonl unreadable / structurally invalid | propagates (setup-time) |
| `WorkflowContextError` | Module-level helper with no active Workflow | propagates (programming error) |
| `DivergenceWarning` (⊂ RdwWarning) | Resume stream stopped matching the journal | warning, not an error |
| `JournalWarning` (⊂ RdwWarning) | Torn final journal line skipped (crash mid-append) | warning, not an error |

The rule: *per-branch* failures are absorbed by the combinators; *run-level*
misconfiguration propagates immediately. That is why `BudgetExceeded` degrades
(hitting the ceiling is an expected runtime condition) while
`AgentLimitExceeded` crashes the wave (a runaway loop is a bug the cap exists
to expose). Oversized waves (`> max_wave` items) raise a plain `ValueError`
before any branch runs — explicit error, never silent truncation.

## Known limits

Stated plainly, because they shape what you should build on top:

1. **Schema forcing can be stonewalled.** Bounded retries + typed failure, not
   constrained decoding. Failure rate is model/effort-dependent.
2. **Budget overshoot by one step.** Usage events land after model steps; a
   single runaway step can exceed the ceiling before the gate or `abort()`
   reacts. `session_limits` tightens this where the Experimental API holds.
3. **Determinism by convention.** Scripts that branch on raw wall-clock, RNG,
   or mutated external state produce loud divergences, not silent corruption —
   but they do lose cache reuse from that point. The sanctioned channels
   (`wf.now()`/`wf.random()`/`wf.uuid()` for in-run nondeterminism, `--arg`
   for run-scoped inputs, `rdw run --strict` to lint for the raw calls) make
   the convention easy to follow, not enforced.
4. **Shared-runtime SPOF.** One child process hosts all sessions; a runtime
   crash fails every in-flight agent (journal-resume is the recovery path).
5. **Experimental SDK surface.** `session_limits` is marked Experimental in the
   SDK; a CLI/SDK update could change it. rdw's enforcement never *depends* on
   it, only strengthens with it.
6. **Hermetic ≠ sandboxed.** Agents you give real tools to can mutate shared
   external state (files, shells, services); the tool allowlist narrows but
   cannot eliminate this. Pure-schema agents are effectively read-only.
