# browser_use.learning

Offline learning layer for the take-home: agent history → filtered action cache → Playwright locators. No LLM in this package. Locators come only from observed DOM evidence (never invent selectors, never use highlight indexes).

## End-to-end flow (step by step)

1. **Input** — An `AgentHistoryList` from a finished browser-use run.
2. **Extract** — Pull meaningful click/type (and similar) steps from history.
3. **Classify / filter** — Mark required vs redundant / exploratory / retry; keep the required path.
4. **Derive locators** — Build Playwright locator commands from recorded DOM attributes.
5. **Persist** — Write `action_cache.json` (stable cache of learned steps).
6. **Hand off** — Optexity’s `automation_builder` turns that cache into an Automation for fast replay.
7. **Optional repair / health** — On failure, minimal single-step repair and simple health states.

This package stays **site-agnostic**. URLs, credentials, and field values belong in Optexity fixtures, not here.

## What this PR changes (and why it looks large)

Most of the diff is a **new package** plus **CI unit tests**. Two small shims keep older import paths working. There is no large unrelated refactor of the agent core.

**Main files to review first:**

| File | Role |
|---|---|
| `extractor.py` | History → candidate actions |
| `redundancy.py` | Keep required steps; drop noise |
| `locator.py` | DOM evidence → Playwright locator command |
| `cache.py` | Save / load `ActionCache` |
| `models.py` | Cache / health / classification schemas |
| `repair.py` | Minimal single-step repair |
| `health.py` / `warming.py` / `analytics.py` / `events.py` | Health, optional warm, reports, structured logs |

**Also in this PR:**

| Path | Role |
|---|---|
| `tests/ci/test_*.py` (learning-related) | Unit tests for redundancy, locators, cache, repair, health, … |
| `browser_use/agent/action_cache.py` | Thin shim → learning cache |
| `browser_use/agent/locator_derivation.py` | Thin shim → learning locator |

Optexity wiring (agentic hooks, Automation builder, fixtures, demos) is in the **optexity** PR under `assignment/` and `optexity/inference/`.

## Tests

```bash
pytest tests/ci/test_redundancy.py tests/ci/test_action_cache.py \
  tests/ci/test_locator_derivation.py tests/ci/test_cache_repair.py \
  tests/ci/test_cache_health.py tests/ci/test_cache_warming.py \
  tests/ci/test_cache_analytics.py -o addopts=
```
