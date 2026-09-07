"""Structured cache / repair / health logging events (no secrets)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger('browser_use.learning.cache_events')

# Phase 1 repair events
CACHE_HIT = 'cache_hit'
CACHE_MISS = 'cache_miss'
CACHE_REPAIR_STARTED = 'cache_repair_started'
CACHE_REPAIR_DETERMINISTIC_SUCCESS = 'cache_repair_deterministic_success'
CACHE_REPAIR_LLM_SUCCESS = 'cache_repair_llm_success'
CACHE_REPAIR_FAILED = 'cache_repair_failed'

# Phase 2 health / degraded-mode events
CACHE_EXECUTION_FAILED = 'cache_execution_failed'
CACHE_MARKED_DEGRADED = 'cache_marked_degraded'
CACHE_REPAIR_ATTEMPTED = 'cache_repair_attempted'
CACHE_REPAIR_SUCCEEDED = 'cache_repair_succeeded'
CACHE_REPAIR_EXHAUSTED = 'cache_repair_exhausted'

# Phase 3 optional TTL / warming events
CACHE_WARMING_STARTED = 'cache_warming_started'
CACHE_WARMING_SKIPPED = 'cache_warming_skipped'
CACHE_WARMING_SUCCEEDED = 'cache_warming_succeeded'
CACHE_WARMING_FAILED = 'cache_warming_failed'


def log_cache_event(event: str, **fields: Any) -> None:
	"""Emit a single structured log line. Never log secrets / API keys / passwords."""
	safe = {k: v for k, v in fields.items() if v is not None and k not in _REDACT_KEYS}
	# Avoid dumping large page HTML / screenshots / cookies.
	for key in list(safe):
		val = safe[key]
		if isinstance(val, str) and len(val) > 500:
			safe[key] = val[:500] + '…'
	parts = ' '.join(f'{k}={v!r}' for k, v in safe.items())
	logger.info('event=%s %s', event, parts)


_REDACT_KEYS = frozenset(
	{
		'password',
		'api_key',
		'token',
		'authorization',
		'secret',
		'cookie',
		'cookies',
		'llm_api_key',
		'ANTHROPIC_API_KEY',
		'LLM_MODEL_API_KEY',
		'screenshot',
		'html',
		'page_content',
	}
)
