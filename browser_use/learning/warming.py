"""Optional cache TTL + warming (Phase 3 enhancement).

This is NOT a platform Optexity TTL — the base assignment does not require one.
TTL is an optional local enhancement on ``ActionCache.lifetime``.

Design:
  cache created → optional TTL countdown → near expiry → lightweight validation
    → healthy: refresh expires_at / last_warmed_at
    → broken step: Phase 1 minimal repair (prefer deterministic; LLM gated by Phase 2)
    → repair fail: Phase 2 DEGRADED (no repeated LLM / no full-workflow rediscovery)

Warming is injectable (call ``maybe_warm_cache``); no heavyweight scheduler.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from enum import Enum

from pydantic import BaseModel, Field

from browser_use.learning.events import (
	CACHE_WARMING_FAILED,
	CACHE_WARMING_SKIPPED,
	CACHE_WARMING_STARTED,
	CACHE_WARMING_SUCCEEDED,
	log_cache_event,
)
from browser_use.learning.health import (
	apply_repair_result_to_health,
	begin_repairing,
	can_attempt_automatic_repair,
	can_attempt_llm_repair,
	mark_degraded,
	note_repair_attempt,
	policy_from_env as health_policy_from_env,
)
from browser_use.learning.models import (
	ActionCache,
	CacheHealthPolicy,
	CacheHealthState,
	CacheLifetime,
	CacheValidationStatus,
	RepairOutcome,
	utc_now_iso,
)
from browser_use.learning.repair import LocatorProbe, TargetedLlmRepair, repair_failed_step

# In-process claim set so concurrent warmers for the same cache do not duplicate work.
_WARM_IN_PROGRESS: set[str] = set()
_WARM_LOCKS_GUARD = asyncio.Lock()


class WarmingSkipReason(str, Enum):
	TTL_DISABLED = 'ttl_disabled'
	WARMING_DISABLED = 'warming_disabled'
	NOT_NEAR_EXPIRY = 'not_near_expiry'
	INSUFFICIENT_USE = 'insufficient_use'
	UNHEALTHY = 'unhealthy_state'
	ALREADY_IN_PROGRESS = 'already_in_progress'
	FORCED = 'forced'  # not a skip — used as marker


class WarmingPolicy(BaseModel):
	"""Configurable warming thresholds (env-overridable)."""

	# Default TTL applied when enabling lifetime on a new cache (None = leave unset).
	default_ttl_seconds: float | None = None
	# Warm when remaining TTL <= this many seconds.
	warm_when_remaining_seconds: float = 300.0
	# Only warm caches that have been used at least this many times.
	min_use_count: int = 1
	# Prefer deterministic repair during warm; LLM only if explicitly allowed + Phase 2 OK.
	allow_llm_during_warm: bool = False


class WarmingResult(BaseModel):
	warmed: bool = False
	skipped: bool = False
	skip_reason: str | None = None
	validation_status: CacheValidationStatus = CacheValidationStatus.UNKNOWN
	repaired_step_ids: list[str] = Field(default_factory=list)
	broken_step_ids: list[str] = Field(default_factory=list)
	cache_version: int | None = None
	expires_at: str | None = None
	last_warmed_at: str | None = None
	health_state: CacheHealthState | None = None
	message: str | None = None


def warming_policy_from_env() -> WarmingPolicy:
	def _float(name: str, default: float | None) -> float | None:
		raw = os.environ.get(name)
		if raw is None or not str(raw).strip():
			return default
		return float(raw)

	def _int(name: str, default: int) -> int:
		raw = os.environ.get(name)
		if raw is None or not str(raw).strip():
			return default
		return int(raw)

	def _bool(name: str, default: bool) -> bool:
		raw = (os.environ.get(name) or '').strip().lower()
		if not raw:
			return default
		return raw in {'1', 'true', 'yes', 'on'}

	ttl = _float('OPTEXITY_CACHE_TTL_SECONDS', None)
	return WarmingPolicy(
		default_ttl_seconds=ttl,
		warm_when_remaining_seconds=_float(
			'OPTEXITY_CACHE_WARM_REMAINING_SECONDS', 300.0
		)
		or 300.0,
		min_use_count=_int('OPTEXITY_CACHE_WARM_MIN_USE_COUNT', 1),
		allow_llm_during_warm=_bool('OPTEXITY_CACHE_WARM_ALLOW_LLM', False),
	)


def _parse_ts(value: str) -> datetime:
	v = value.replace('Z', '+00:00')
	dt = datetime.fromisoformat(v)
	if dt.tzinfo is None:
		dt = dt.replace(tzinfo=timezone.utc)
	return dt


def _now() -> datetime:
	return datetime.now(timezone.utc)


def _lifetime(cache: ActionCache) -> CacheLifetime:
	return cache.lifetime if cache.lifetime is not None else CacheLifetime()


def _cache_lock_key(cache: ActionCache) -> str:
	return cache.workflow_id or cache.source.get('url') or id(cache).__repr__()


async def _try_claim_warming(key: str) -> bool:
	"""Atomically claim warming for ``key``. Returns False if already claimed."""
	async with _WARM_LOCKS_GUARD:
		if key in _WARM_IN_PROGRESS:
			return False
		_WARM_IN_PROGRESS.add(key)
		return True


async def _release_warming(key: str) -> None:
	async with _WARM_LOCKS_GUARD:
		_WARM_IN_PROGRESS.discard(key)


def reset_warming_locks_for_tests() -> None:
	"""Test helper: clear in-process warming claims."""
	_WARM_IN_PROGRESS.clear()


def enable_ttl(
	cache: ActionCache,
	*,
	ttl_seconds: float,
	warming_enabled: bool = True,
	now: datetime | None = None,
) -> ActionCache:
	"""Attach/refresh TTL metadata. Enhancement — call explicitly or via env defaults."""
	now = now or _now()
	created = _lifetime(cache).created_at or utc_now_iso()
	expires = (now + timedelta(seconds=ttl_seconds)).isoformat()
	life = _lifetime(cache).model_copy(
		update={
			'ttl_seconds': float(ttl_seconds),
			'created_at': created,
			'expires_at': expires,
			'warming_enabled': warming_enabled,
		}
	)
	return cache.model_copy(update={'lifetime': life})


def record_cache_use(cache: ActionCache, *, n: int = 1) -> ActionCache:
	"""Bump relevance counter used by the warming policy."""
	life = _lifetime(cache).model_copy(update={'use_count': _lifetime(cache).use_count + n})
	return cache.model_copy(update={'lifetime': life})


def remaining_ttl_seconds(cache: ActionCache, *, now: datetime | None = None) -> float | None:
	life = _lifetime(cache)
	if life.ttl_seconds is None or not life.expires_at:
		return None
	now = now or _now()
	return (_parse_ts(life.expires_at) - now).total_seconds()


def should_warm(
	cache: ActionCache,
	policy: WarmingPolicy,
	*,
	now: datetime | None = None,
	force: bool = False,
) -> tuple[bool, str]:
	"""Decide whether warming should run. Never warms every cache blindly."""
	now = now or _now()
	life = _lifetime(cache)
	health_state = cache.health.state if cache.health else CacheHealthState.HEALTHY

	if force:
		return True, WarmingSkipReason.FORCED.value

	if life.warming_in_progress:
		return False, WarmingSkipReason.ALREADY_IN_PROGRESS.value

	if life.ttl_seconds is None or not life.expires_at:
		return False, WarmingSkipReason.TTL_DISABLED.value

	if not life.warming_enabled:
		return False, WarmingSkipReason.WARMING_DISABLED.value

	if health_state in {CacheHealthState.DEGRADED, CacheHealthState.DISABLED}:
		return False, WarmingSkipReason.UNHEALTHY.value

	if life.use_count < policy.min_use_count:
		return False, WarmingSkipReason.INSUFFICIENT_USE.value

	remaining = remaining_ttl_seconds(cache, now=now)
	assert remaining is not None
	if remaining > policy.warm_when_remaining_seconds:
		return False, WarmingSkipReason.NOT_NEAR_EXPIRY.value

	return True, 'near_expiry'


async def lightweight_validate_locators(
	cache: ActionCache,
	probe: LocatorProbe,
) -> list[str]:
	"""Cheap check: do kept step locators still resolve? No full workflow execution."""

	async def _ok(cmd: str) -> bool:
		value = probe(cmd)
		if hasattr(value, '__await__'):
			return bool(await value)  # type: ignore[misc]
		return bool(value)

	broken: list[str] = []
	for action in cache.actions:
		if action.status != 'kept':
			continue
		if not action.locator or not action.locator.command:
			broken.append(action.step_id)
			continue
		if not await _ok(action.locator.command):
			broken.append(action.step_id)
	return broken


def _refresh_ttl(cache: ActionCache, *, now: datetime | None = None) -> ActionCache:
	now = now or _now()
	life = _lifetime(cache)
	ttl = life.ttl_seconds
	if ttl is None:
		return cache
	life = life.model_copy(
		update={
			'last_warmed_at': now.isoformat(),
			'expires_at': (now + timedelta(seconds=ttl)).isoformat(),
			'warming_in_progress': False,
		}
	)
	return cache.model_copy(update={'lifetime': life, 'version': cache.version + 1})


async def maybe_warm_cache(
	cache: ActionCache,
	*,
	probe: LocatorProbe,
	llm_repair: TargetedLlmRepair | None = None,
	warming_policy: WarmingPolicy | None = None,
	health_policy: CacheHealthPolicy | None = None,
	now: datetime | None = None,
	force: bool = False,
) -> tuple[ActionCache, WarmingResult]:
	"""Injectable warming entrypoint (call from a cron/job/harness — no built-in scheduler)."""
	now = now or _now()
	warming_policy = warming_policy or warming_policy_from_env()
	health_policy = health_policy or health_policy_from_env()

	ok, reason = should_warm(cache, warming_policy, now=now, force=force)
	if not ok:
		life = _lifetime(cache).model_copy(update={'last_warm_skip_reason': reason})
		cache = cache.model_copy(update={'lifetime': life})
		log_cache_event(
			CACHE_WARMING_SKIPPED,
			workflow_id=cache.workflow_id,
			reason=reason,
			remaining_ttl=remaining_ttl_seconds(cache, now=now),
		)
		return cache, WarmingResult(
			warmed=False,
			skipped=True,
			skip_reason=reason,
			validation_status=_lifetime(cache).validation_status,
			cache_version=cache.version,
			expires_at=_lifetime(cache).expires_at,
			health_state=cache.health.state,
			message=f'Warming skipped: {reason}',
		)

	claim_key = _cache_lock_key(cache)
	if not await _try_claim_warming(claim_key):
		log_cache_event(
			CACHE_WARMING_SKIPPED,
			workflow_id=cache.workflow_id,
			reason=WarmingSkipReason.ALREADY_IN_PROGRESS.value,
		)
		return cache, WarmingResult(
			warmed=False,
			skipped=True,
			skip_reason=WarmingSkipReason.ALREADY_IN_PROGRESS.value,
			cache_version=cache.version,
			health_state=cache.health.state,
			message='Warming already in progress for this cache',
		)

	try:
		# Re-check after claim (idempotent against a just-finished warmer).
		ok, reason = should_warm(cache, warming_policy, now=now, force=force)
		if not ok:
			log_cache_event(
				CACHE_WARMING_SKIPPED,
				workflow_id=cache.workflow_id,
				reason=reason,
			)
			return cache, WarmingResult(
				warmed=False,
				skipped=True,
				skip_reason=reason,
				cache_version=cache.version,
				health_state=cache.health.state,
			)

		life = _lifetime(cache).model_copy(update={'warming_in_progress': True})
		cache = cache.model_copy(update={'lifetime': life})
		log_cache_event(
			CACHE_WARMING_STARTED,
			workflow_id=cache.workflow_id,
			version=cache.version,
			expires_at=life.expires_at,
			remaining_ttl=remaining_ttl_seconds(cache, now=now),
		)

		broken = await lightweight_validate_locators(cache, probe)
		repaired: list[str] = []

		if not broken:
			life = _lifetime(cache).model_copy(
				update={
					'validation_status': CacheValidationStatus.VALID,
					'warming_in_progress': False,
				}
			)
			cache = cache.model_copy(update={'lifetime': life})
			cache = _refresh_ttl(cache, now=now)
			log_cache_event(
				CACHE_WARMING_SUCCEEDED,
				workflow_id=cache.workflow_id,
				validation_status='valid',
				version=cache.version,
				expires_at=_lifetime(cache).expires_at,
			)
			return cache, WarmingResult(
				warmed=True,
				validation_status=CacheValidationStatus.VALID,
				cache_version=cache.version,
				expires_at=_lifetime(cache).expires_at,
				last_warmed_at=_lifetime(cache).last_warmed_at,
				health_state=cache.health.state,
				message='Lightweight validation passed; TTL refreshed',
			)

		# Broken steps → Phase 1 minimal repair, gated by Phase 2 circuit breaker.
		for step_id in broken:
			allowed, deny = can_attempt_automatic_repair(cache, health_policy)
			if not allowed:
				cache = mark_degraded(
					cache, step_id=step_id, reason=f'warm_repair_blocked:{deny}'
				)
				life = _lifetime(cache).model_copy(
					update={
						'validation_status': CacheValidationStatus.FAILED,
						'warming_in_progress': False,
					}
				)
				cache = cache.model_copy(update={'lifetime': life})
				log_cache_event(
					CACHE_WARMING_FAILED,
					workflow_id=cache.workflow_id,
					step_id=step_id,
					reason=deny,
					health_state=cache.health.state.value,
				)
				return cache, WarmingResult(
					warmed=False,
					validation_status=CacheValidationStatus.FAILED,
					broken_step_ids=broken,
					repaired_step_ids=repaired,
					cache_version=cache.version,
					health_state=cache.health.state,
					message=f'Warming aborted; repair blocked ({deny}); marked degraded',
				)

			cache = begin_repairing(cache, step_id=step_id, reason='warm_validation_miss')
			cache = note_repair_attempt(cache, llm=False)

			llm_ok, _ = can_attempt_llm_repair(cache, health_policy)
			allow_llm = bool(
				warming_policy.allow_llm_during_warm and llm_ok and llm_repair is not None
			)

			cache, result = await repair_failed_step(
				cache,
				step_id,
				failure_error='warm_validation_locator_missing',
				probe=probe,
				llm_repair=llm_repair if allow_llm else None,
				allow_llm=allow_llm,
			)
			cache, failure = apply_repair_result_to_health(
				cache,
				health_policy,
				result,
				llm_attempted=allow_llm and result.outcome == RepairOutcome.LLM_REPAIR,
				deterministic_attempted=True,
			)
			if failure is not None or result.outcome == RepairOutcome.UNRECOVERABLE:
				if cache.health.state != CacheHealthState.DEGRADED:
					cache = mark_degraded(
						cache,
						step_id=step_id,
						reason=result.error or 'warm_repair_failed',
					)
				life = _lifetime(cache).model_copy(
					update={
						'validation_status': CacheValidationStatus.FAILED,
						'warming_in_progress': False,
					}
				)
				cache = cache.model_copy(update={'lifetime': life})
				log_cache_event(
					CACHE_WARMING_FAILED,
					workflow_id=cache.workflow_id,
					step_id=step_id,
					reason=result.error or 'warm_repair_failed',
					health_state=cache.health.state.value,
				)
				return cache, WarmingResult(
					warmed=False,
					validation_status=CacheValidationStatus.FAILED,
					broken_step_ids=broken,
					repaired_step_ids=repaired,
					cache_version=cache.version,
					health_state=cache.health.state,
					message='Warming failed during minimal repair; cache DEGRADED; no repeated LLM',
				)
			repaired.append(step_id)

		life = _lifetime(cache).model_copy(
			update={
				'validation_status': CacheValidationStatus.REPAIRED,
				'warming_in_progress': False,
			}
		)
		cache = cache.model_copy(update={'lifetime': life})
		cache = _refresh_ttl(cache, now=now)
		log_cache_event(
			CACHE_WARMING_SUCCEEDED,
			workflow_id=cache.workflow_id,
			validation_status='repaired',
			version=cache.version,
			repaired_steps=len(repaired),
			expires_at=_lifetime(cache).expires_at,
		)
		return cache, WarmingResult(
			warmed=True,
			validation_status=CacheValidationStatus.REPAIRED,
			repaired_step_ids=repaired,
			broken_step_ids=broken,
			cache_version=cache.version,
			expires_at=_lifetime(cache).expires_at,
			last_warmed_at=_lifetime(cache).last_warmed_at,
			health_state=cache.health.state,
			message='Warming repaired broken step(s) and refreshed TTL',
		)
	except Exception as e:
		life = _lifetime(cache).model_copy(
			update={
				'warming_in_progress': False,
				'validation_status': CacheValidationStatus.FAILED,
			}
		)
		cache = cache.model_copy(update={'lifetime': life})
		log_cache_event(
			CACHE_WARMING_FAILED,
			workflow_id=cache.workflow_id,
			reason=type(e).__name__,
		)
		raise
	finally:
		await _release_warming(claim_key)
