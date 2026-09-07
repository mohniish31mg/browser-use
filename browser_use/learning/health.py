"""Cache health / degraded-mode state machine (Phase 2).

State model:

    HEALTHY ──cache step fails──► REPAIRING
                                    │
                     repair success │
                                    ▼
                                 HEALTHY

                     repair exhausted / repeated failures
                                    ▼
                                 DEGRADED

    DISABLED — operator kill-switch (no automatic execution/repair)

DEGRADED / DISABLED never mean "LLM the whole workflow".
Recovery is explicit: request_relearn() or mark_healthy() after a successful
manual/offline relearn.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from browser_use.learning.events import (
	CACHE_EXECUTION_FAILED,
	CACHE_MARKED_DEGRADED,
	CACHE_REPAIR_ATTEMPTED,
	CACHE_REPAIR_EXHAUSTED,
	CACHE_REPAIR_SUCCEEDED,
	log_cache_event,
)
from browser_use.learning.models import (
	ActionCache,
	CacheExecutionFailure,
	CacheHealth,
	CacheHealthPolicy,
	CacheHealthState,
	RepairOutcome,
	RepairResult,
	utc_now_iso,
)


def policy_from_env() -> CacheHealthPolicy:
	"""Load thresholds from env; unset vars keep safe defaults."""

	def _int(name: str, default: int) -> int:
		raw = os.environ.get(name)
		if raw is None or not str(raw).strip():
			return default
		return int(raw)

	def _float(name: str, default: float) -> float:
		raw = os.environ.get(name)
		if raw is None or not str(raw).strip():
			return default
		return float(raw)

	def _bool(name: str, default: bool) -> bool:
		raw = (os.environ.get(name) or '').strip().lower()
		if not raw:
			return default
		return raw in {'1', 'true', 'yes', 'on'}

	return CacheHealthPolicy(
		failures_to_degrade=_int('OPTEXITY_CACHE_FAILURES_TO_DEGRADE', 2),
		max_repair_attempts_per_window=_int('OPTEXITY_CACHE_MAX_REPAIR_ATTEMPTS', 2),
		max_llm_repairs_per_window=_int('OPTEXITY_CACHE_MAX_LLM_REPAIRS', 1),
		failure_window_seconds=_float('OPTEXITY_CACHE_FAILURE_WINDOW_SECONDS', 3600.0),
		allow_automatic_repair_when_degraded=_bool(
			'OPTEXITY_CACHE_ALLOW_REPAIR_WHEN_DEGRADED', False
		),
	)


def _parse_ts(value: str) -> datetime:
	# Support both Z and +00:00
	v = value.replace('Z', '+00:00')
	dt = datetime.fromisoformat(v)
	if dt.tzinfo is None:
		dt = dt.replace(tzinfo=timezone.utc)
	return dt


def _prune(timestamps: list[str], window_seconds: float, *, now: datetime | None = None) -> list[str]:
	now = now or datetime.now(timezone.utc)
	kept: list[str] = []
	for ts in timestamps:
		try:
			if (now - _parse_ts(ts)).total_seconds() <= window_seconds:
				kept.append(ts)
		except ValueError:
			continue
	return kept


def _ensure_health(cache: ActionCache) -> CacheHealth:
	return cache.health if cache.health is not None else CacheHealth()


def prune_health_windows(cache: ActionCache, policy: CacheHealthPolicy) -> ActionCache:
	health = _ensure_health(cache)
	health = health.model_copy(
		update={
			'failure_timestamps': _prune(health.failure_timestamps, policy.failure_window_seconds),
			'repair_attempt_timestamps': _prune(
				health.repair_attempt_timestamps, policy.failure_window_seconds
			),
			'llm_repair_timestamps': _prune(
				health.llm_repair_timestamps, policy.failure_window_seconds
			),
		}
	)
	return cache.model_copy(update={'health': health})


def can_attempt_automatic_repair(
	cache: ActionCache, policy: CacheHealthPolicy
) -> tuple[bool, str]:
	"""Whether automatic (non-explicit) repair may run."""
	cache = prune_health_windows(cache, policy)
	health = _ensure_health(cache)
	if health.state == CacheHealthState.DISABLED:
		return False, 'cache_disabled'
	if health.state == CacheHealthState.DEGRADED and not policy.allow_automatic_repair_when_degraded:
		return False, 'cache_degraded'
	if len(health.repair_attempt_timestamps) >= policy.max_repair_attempts_per_window:
		return False, 'repair_rate_limited'
	return True, 'ok'


def can_attempt_llm_repair(cache: ActionCache, policy: CacheHealthPolicy) -> tuple[bool, str]:
	cache = prune_health_windows(cache, policy)
	health = _ensure_health(cache)
	ok, reason = can_attempt_automatic_repair(cache, policy)
	if not ok:
		return False, reason
	if len(health.llm_repair_timestamps) >= policy.max_llm_repairs_per_window:
		return False, 'llm_repair_rate_limited'
	return True, 'ok'


def begin_repairing(cache: ActionCache, *, step_id: str | None, reason: str | None) -> ActionCache:
	health = _ensure_health(cache).model_copy(
		update={
			'state': CacheHealthState.REPAIRING,
			'updated_at': utc_now_iso(),
			'last_failure_step_id': step_id,
			'last_failure_reason': reason,
			'recommended_action': 'attempt_minimal_single_step_repair',
		}
	)
	log_cache_event(
		CACHE_REPAIR_ATTEMPTED,
		workflow_id=cache.workflow_id,
		step_id=step_id,
		state=health.state.value,
	)
	return cache.model_copy(update={'health': health})


def note_repair_attempt(cache: ActionCache, *, llm: bool) -> ActionCache:
	health = _ensure_health(cache)
	now = utc_now_iso()
	repair_ts = list(health.repair_attempt_timestamps) + [now]
	llm_ts = list(health.llm_repair_timestamps)
	if llm:
		llm_ts = llm_ts + [now]
	health = health.model_copy(
		update={
			'repair_attempt_timestamps': repair_ts,
			'llm_repair_timestamps': llm_ts,
			'updated_at': now,
		}
	)
	return cache.model_copy(update={'health': health})


def mark_healthy(cache: ActionCache, *, reason: str = 'repair_succeeded') -> ActionCache:
	health = _ensure_health(cache).model_copy(
		update={
			'state': CacheHealthState.HEALTHY,
			'updated_at': utc_now_iso(),
			'recommended_action': None,
			'relearn_requested': False,
			'last_failure_reason': None,
		}
	)
	log_cache_event(
		CACHE_REPAIR_SUCCEEDED,
		workflow_id=cache.workflow_id,
		state=health.state.value,
		reason=reason,
	)
	return cache.model_copy(update={'health': health})


def mark_degraded(
	cache: ActionCache,
	*,
	step_id: str | None,
	reason: str,
	recommended_action: str = 'request_explicit_relearn_or_manual_repair',
) -> ActionCache:
	now = utc_now_iso()
	health = _ensure_health(cache)
	failures = list(health.failure_timestamps) + [now]
	health = health.model_copy(
		update={
			'state': CacheHealthState.DEGRADED,
			'updated_at': now,
			'last_failure_at': now,
			'last_failure_step_id': step_id,
			'last_failure_reason': reason,
			'failure_timestamps': failures,
			'recommended_action': recommended_action,
		}
	)
	log_cache_event(
		CACHE_MARKED_DEGRADED,
		workflow_id=cache.workflow_id,
		step_id=step_id,
		reason=reason,
		state=health.state.value,
	)
	log_cache_event(
		CACHE_REPAIR_EXHAUSTED,
		workflow_id=cache.workflow_id,
		step_id=step_id,
		reason=reason,
	)
	return cache.model_copy(update={'health': health})


def mark_disabled(cache: ActionCache, *, reason: str = 'operator_disabled') -> ActionCache:
	health = _ensure_health(cache).model_copy(
		update={
			'state': CacheHealthState.DISABLED,
			'updated_at': utc_now_iso(),
			'recommended_action': 'enable_cache_after_relearn',
			'last_failure_reason': reason,
		}
	)
	return cache.model_copy(update={'health': health})


def record_unrecovered_failure(
	cache: ActionCache,
	policy: CacheHealthPolicy,
	*,
	step_id: str | None,
	reason: str,
) -> ActionCache:
	"""Record a failure that was not repaired; may transition to DEGRADED."""
	cache = prune_health_windows(cache, policy)
	health = _ensure_health(cache)
	now = utc_now_iso()
	failures = list(health.failure_timestamps) + [now]
	health = health.model_copy(
		update={
			'failure_timestamps': failures,
			'last_failure_at': now,
			'last_failure_step_id': step_id,
			'last_failure_reason': reason,
			'updated_at': now,
			'recommended_action': 'retry_with_minimal_repair_on_next_failure',
			# Leave REPAIRING until we decide degrade vs settle.
			'state': CacheHealthState.REPAIRING,
		}
	)
	cache = cache.model_copy(update={'health': health})
	cache = prune_health_windows(cache, policy)
	n_fail = len(_ensure_health(cache).failure_timestamps)
	if n_fail >= policy.failures_to_degrade:
		return mark_degraded(cache, step_id=step_id, reason=reason)

	# Under threshold: settle back to HEALTHY but keep failure counters in the window.
	health = _ensure_health(cache).model_copy(
		update={
			'state': CacheHealthState.HEALTHY,
			'recommended_action': 'retry_with_minimal_repair_on_next_failure',
		}
	)
	return cache.model_copy(update={'health': health})


def request_relearn(cache: ActionCache) -> ActionCache:
	"""Explicit recovery path: flag cache for offline/agentic relearn (manual)."""
	health = _ensure_health(cache).model_copy(
		update={
			'relearn_requested': True,
			'updated_at': utc_now_iso(),
			'recommended_action': 'run_agentic_relearn_then_mark_healthy',
			# Stay DEGRADED/DISABLED until operator marks healthy after relearn.
			'state': CacheHealthState.DEGRADED
			if _ensure_health(cache).state != CacheHealthState.DISABLED
			else CacheHealthState.DISABLED,
		}
	)
	log_cache_event(
		'cache_relearn_requested',
		workflow_id=cache.workflow_id,
		state=health.state.value,
	)
	return cache.model_copy(update={'health': health})


def explicit_recover_to_healthy(cache: ActionCache) -> ActionCache:
	"""Operator confirms cache is good again after manual repair/relearn."""
	health = _ensure_health(cache).model_copy(
		update={
			'state': CacheHealthState.HEALTHY,
			'updated_at': utc_now_iso(),
			'relearn_requested': False,
			'recommended_action': None,
			'failure_timestamps': [],
			'repair_attempt_timestamps': [],
			'llm_repair_timestamps': [],
			'last_failure_at': None,
			'last_failure_step_id': None,
			'last_failure_reason': None,
		}
	)
	log_cache_event(
		CACHE_REPAIR_SUCCEEDED,
		workflow_id=cache.workflow_id,
		state=health.state.value,
		reason='explicit_recovery',
	)
	return cache.model_copy(update={'health': health})


def build_execution_failure(
	cache: ActionCache,
	*,
	step_id: str | None,
	failure_reason: str,
	deterministic_repair_attempted: bool,
	llm_repair_attempted: bool,
	repair_outcome: RepairOutcome | None = None,
	message: str | None = None,
) -> CacheExecutionFailure:
	health = _ensure_health(cache)
	recommended = health.recommended_action or 'request_explicit_relearn_or_manual_repair'
	if health.state == CacheHealthState.DEGRADED:
		recommended = 'cache_degraded_call_request_relearn_then_explicit_recover_to_healthy'
	elif health.state == CacheHealthState.DISABLED:
		recommended = 'cache_disabled_enable_after_relearn'

	failure = CacheExecutionFailure(
		workflow_id=cache.workflow_id,
		step_id=step_id,
		failure_reason=failure_reason,
		deterministic_repair_attempted=deterministic_repair_attempted,
		llm_repair_attempted=llm_repair_attempted,
		cache_state=health.state,
		recommended_next_action=recommended,
		repair_outcome=repair_outcome,
		full_agentic_fallback=False,
		cache_version=cache.version,
		message=message
		or 'Cached step failed; full-workflow LLM rediscovery was NOT triggered.',
	)
	log_cache_event(
		CACHE_EXECUTION_FAILED,
		workflow_id=failure.workflow_id,
		step_id=failure.step_id,
		cache_state=failure.cache_state.value,
		deterministic_repair_attempted=failure.deterministic_repair_attempted,
		llm_repair_attempted=failure.llm_repair_attempted,
		full_agentic_fallback=False,
		reason=failure_reason,
	)
	return failure


def apply_repair_result_to_health(
	cache: ActionCache,
	policy: CacheHealthPolicy,
	result: RepairResult,
	*,
	llm_attempted: bool,
	deterministic_attempted: bool,
) -> tuple[ActionCache, CacheExecutionFailure | None]:
	"""Update health after a Phase-1 repair attempt. Returns failure if unrecovered."""
	if result.outcome in {
		RepairOutcome.CACHE_HIT,
		RepairOutcome.DETERMINISTIC_REPAIR,
		RepairOutcome.LLM_REPAIR,
	}:
		return mark_healthy(cache, reason=result.outcome.value), None

	cache = record_unrecovered_failure(
		cache,
		policy,
		step_id=result.step_id,
		reason=result.error or result.message or 'unrecoverable_repair',
	)
	failure = build_execution_failure(
		cache,
		step_id=result.step_id,
		failure_reason=result.error or result.message or 'unrecoverable_repair',
		deterministic_repair_attempted=deterministic_attempted,
		llm_repair_attempted=llm_attempted,
		repair_outcome=result.outcome,
		message=result.message,
	)
	return cache, failure


def health_summary(cache: ActionCache) -> dict[str, Any]:
	health = _ensure_health(cache)
	return {
		'workflow_id': cache.workflow_id,
		'state': health.state.value,
		'version': cache.version,
		'relearn_requested': health.relearn_requested,
		'recommended_action': health.recommended_action,
		'last_failure_step_id': health.last_failure_step_id,
		'failures_in_window': len(health.failure_timestamps),
		'repairs_in_window': len(health.repair_attempt_timestamps),
		'llm_repairs_in_window': len(health.llm_repair_timestamps),
	}
