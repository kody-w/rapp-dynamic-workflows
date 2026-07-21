# Examples

Each example is a self-contained workflow script: it defines
`async def workflow(wf)` and runs under the `rdw` CLI or standalone with
plain `python`. **None of them spend anything by default** — every script
checks the `RDW_LIVE=1` environment variable, and without it prints what it
*would* do and exits 0. Set the variable only on a machine where `copilot`
is logged in and you are willing to spend the listed credits.

```bash
# dry (free, prints an explanation):
rdw run examples/01_hello_agent.py

# live:
RDW_LIVE=1 rdw run examples/01_hello_agent.py --budget 2
```

Always pass `--budget` on live runs — it is a hard admission gate, so a
misbehaving workflow stops instead of spending indefinitely.

| # | Script | Pattern | Typical cost (AIU) |
|---|--------|---------|--------------------|
| 01 | [`01_hello_agent.py`](01_hello_agent.py) | One schema-forced agent — the submit-tool pattern that replaces the missing native `response_format` | 0.5–1 |
| 02 | [`02_parallel_review.py`](02_parallel_review.py) | **The canonical pattern**: parallel per-dimension reviewers, then `adversarial_verify` skeptic panels to kill false positives | 8–20 |
| 03 | [`03_pipeline_migration.py`](03_pipeline_migration.py) | `pipeline(items, *stages)` over files — per-item flow, no barrier between stages, poisoned items drop to `None` | 5–10 |
| 04 | [`04_judge_panel.py`](04_judge_panel.py) | Generate stance-diverse candidates in parallel, rank them with `judge_panel` (one independent judge per lens) | 4–8 |
| 05 | [`05_budget_loop.py`](05_budget_loop.py) | Loop-until-budget: critic→rewrite rounds until convergence or the `BudgetExceeded` gate ends the loop gracefully | == your `--budget` |
| 06 | [`06_resume_demo.py`](06_resume_demo.py) | Kill/resume walkthrough — Ctrl-C mid-run, `--resume` replays finished agents from the journal for free, live from the first divergence | 3–6 cold; tail only on resume |

Suggested order: 01 → 02 → 06. 01 teaches the one core primitive
(schema-forced `agent`), 02 is the pattern you will actually reuse, and 06
shows why long runs are safe to interrupt.

Cost figures are rough — they depend on the model, reasoning effort, and how
chatty the task turns out to be. The `--budget` ceiling is the real
guarantee, not the estimate.

## Reading the results

```bash
rdw runs               # list runs, newest first, with spend
rdw show <RUN_ID>      # per-agent journal: status, credits, wall time
rdw show <RUN_ID> -v   # include each agent's submitted result
```
