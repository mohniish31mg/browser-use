# browser_use.learning

Offline learning layer: agent history → filtered action cache → Playwright
locators. Locators come from observed DOM evidence only (no highlight indexes).

Flow: history → extract/filter → derive locators → `action_cache.json` →
Optexity `automation_builder` → Automation.

## Modules

| Module | Role |
|---|---|
| `models.py` | Cache / health / classification schemas |
| `extractor.py` | History → actions |
| `redundancy.py` | required / redundant / exploratory / retry |
| `locator.py` | DOM evidence → Playwright command |
| `cache.py` | Persist / load ActionCache |
| `repair.py` | Minimal single-step repair |
| `health.py` | HEALTHY → REPAIRING → DEGRADED |
| `warming.py` | Optional TTL / warm |
| `analytics.py` | Measured performance reports |
| `events.py` | Structured cache_* logs |

## Tests

```bash
pytest tests/ci/test_redundancy.py tests/ci/test_action_cache.py \
  tests/ci/test_locator_derivation.py tests/ci/test_cache_repair.py \
  tests/ci/test_cache_health.py tests/ci/test_cache_warming.py \
  tests/ci/test_cache_analytics.py -o addopts=
```

Site URLs and field values belong in **Optexity fixtures**, not in this package.
