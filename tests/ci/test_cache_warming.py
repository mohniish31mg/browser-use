"""Unit tests for optional cache TTL + warming (Phase 3)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from browser_use.learning.models import (
	ActionCache,
	CacheAction,
	CacheActionResult,
	CacheActionSource,
	CacheHealthPolicy,
	CacheHealthState,
	CacheLocator,
	CacheTarget,
	CacheValidationStatus,
)
from browser_use.learning.warming import (
	WarmingPolicy,
	WarmingSkipReason,
	enable_ttl,
	maybe_warm_cache,
	record_cache_use,
	remaining_ttl_seconds,
	reset_warming_locks_for_tests,
	should_warm,
)


@pytest.fixture(autouse=True)
def _clear_warm_claims():
	reset_warming_locks_for_tests()
	yield
	reset_warming_locks_for_tests()


def _action(
	step_id: str,
	command: str,
	*,
	attrs: dict | None = None,
	element_hash: int = 1,
) -> CacheAction:
	return CacheAction(
		sequence=element_hash,
		step_id=step_id,
		action_type='click',
		target=CacheTarget(
			node_name='BUTTON',
			attributes=attrs or {'id': 'ok', 'data-testid': 'alt'},
			xpath='html/body/button',
			element_hash=element_hash,
			stable_hash=element_hash,
			ax_name='Go',
		),
		result=CacheActionResult(success=True),
		status='kept',
		source=CacheActionSource(history_step=0, action_index=0),
		locator=CacheLocator(strategy='id', command=command, evidence={}),
	)


def _cache(actions: list[CacheAction] | None = None) -> ActionCache:
	return ActionCache(
		version=1,
		source={'url': 'https://example.com/warm'},
		workflow_id='wf_warm_test',
		actions=actions or [_action('step_1', 'locator("#ok")')],
	)


def test_not_near_expiry_skips_warming():
	now = datetime(2026, 1, 1, tzinfo=timezone.utc)
	cache = enable_ttl(_cache(), ttl_seconds=3600, warming_enabled=True, now=now)
	cache = record_cache_use(cache)
	policy = WarmingPolicy(warm_when_remaining_seconds=300, min_use_count=1)
	# Still far from expiry (3600 > 300)
	ok, reason = should_warm(cache, policy, now=now)
	assert ok is False
	assert reason == WarmingSkipReason.NOT_NEAR_EXPIRY.value


def test_near_expiry_allows_warming():
	now = datetime(2026, 1, 1, tzinfo=timezone.utc)
	cache = enable_ttl(_cache(), ttl_seconds=200, warming_enabled=True, now=now)
	cache = record_cache_use(cache)
	policy = WarmingPolicy(warm_when_remaining_seconds=300, min_use_count=1)
	ok, reason = should_warm(cache, policy, now=now)
	assert ok is True
	assert reason == 'near_expiry'


@pytest.mark.asyncio
async def test_successful_warming_refreshes_ttl_metadata():
	now = datetime(2026, 1, 1, tzinfo=timezone.utc)
	cache = enable_ttl(_cache(), ttl_seconds=100, warming_enabled=True, now=now)
	cache = record_cache_use(cache)
	version_before = cache.version
	# Advance clock slightly so refreshed expires_at moves forward.
	warm_at = now + timedelta(seconds=10)

	async def probe(cmd: str) -> bool:
		return cmd == 'locator("#ok")'

	policy = WarmingPolicy(warm_when_remaining_seconds=300, min_use_count=1)
	cache, result = await maybe_warm_cache(
		cache, probe=probe, warming_policy=policy, now=warm_at
	)
	assert result.warmed is True
	assert result.skipped is False
	assert result.validation_status == CacheValidationStatus.VALID
	assert cache.lifetime.last_warmed_at == warm_at.isoformat()
	assert cache.lifetime.expires_at == (warm_at + timedelta(seconds=100)).isoformat()
	assert cache.version == version_before + 1
	assert cache.lifetime.validation_status == CacheValidationStatus.VALID
	assert remaining_ttl_seconds(cache, now=warm_at) == pytest.approx(100, abs=1)


@pytest.mark.asyncio
async def test_warming_detects_changed_locator_and_repairs():
	# Primary id broken; data-testid alternative in evidence.
	action = _action(
		'step_broken',
		'locator("#gone")',
		attrs={'id': 'gone', 'data-testid': 'still-good'},
		element_hash=7,
	)
	action.locator = CacheLocator(
		strategy='id', command='locator("#gone")', evidence={'id': 'gone'}
	)
	now = datetime(2026, 1, 1, tzinfo=timezone.utc)
	cache = enable_ttl(_cache([action]), ttl_seconds=60, warming_enabled=True, now=now)
	cache = record_cache_use(cache)

	async def probe(cmd: str) -> bool:
		return cmd == 'get_by_test_id("still-good")'

	policy = WarmingPolicy(
		warm_when_remaining_seconds=300, min_use_count=1, allow_llm_during_warm=False
	)
	health = CacheHealthPolicy(failures_to_degrade=5, max_repair_attempts_per_window=5)
	cache, result = await maybe_warm_cache(
		cache,
		probe=probe,
		warming_policy=policy,
		health_policy=health,
		now=now,
	)
	assert result.warmed is True
	assert result.validation_status == CacheValidationStatus.REPAIRED
	assert 'step_broken' in result.repaired_step_ids
	assert cache.actions[0].locator is not None
	assert cache.actions[0].locator.command == 'get_by_test_id("still-good")'
	assert cache.health.state == CacheHealthState.HEALTHY
	assert cache.lifetime.last_warmed_at is not None


@pytest.mark.asyncio
async def test_repair_failure_during_warm_marks_degraded():
	action = _action('step_dead', 'locator("#gone")', attrs={}, element_hash=1)
	action.target.attributes = {}
	action.target.xpath = ''
	action.target.ax_name = None
	action.locator = CacheLocator(strategy='id', command='locator("#gone")', evidence={})
	now = datetime(2026, 1, 1, tzinfo=timezone.utc)
	cache = enable_ttl(_cache([action]), ttl_seconds=60, warming_enabled=True, now=now)
	cache = record_cache_use(cache)

	async def probe(_cmd: str) -> bool:
		return False

	llm_calls = {'n': 0}

	async def llm_repair(_action, _prompt: str):
		llm_calls['n'] += 1
		return None

	# failures_to_degrade=1 so first unrecovered warm failure → DEGRADED
	policy = WarmingPolicy(
		warm_when_remaining_seconds=300, min_use_count=1, allow_llm_during_warm=False
	)
	health = CacheHealthPolicy(failures_to_degrade=1, max_repair_attempts_per_window=5)
	cache, result = await maybe_warm_cache(
		cache,
		probe=probe,
		llm_repair=llm_repair,
		warming_policy=policy,
		health_policy=health,
		now=now,
	)
	assert result.warmed is False
	assert result.validation_status == CacheValidationStatus.FAILED
	assert cache.health.state == CacheHealthState.DEGRADED
	assert llm_calls['n'] == 0  # allow_llm_during_warm=False → no LLM spam


@pytest.mark.asyncio
async def test_duplicate_warming_request_prevented():
	now = datetime(2026, 1, 1, tzinfo=timezone.utc)
	cache = enable_ttl(_cache(), ttl_seconds=60, warming_enabled=True, now=now)
	cache = record_cache_use(cache)
	policy = WarmingPolicy(warm_when_remaining_seconds=300, min_use_count=1)

	started = asyncio.Event()
	release = asyncio.Event()

	async def slow_probe(cmd: str) -> bool:
		started.set()
		await release.wait()
		return cmd == 'locator("#ok")'

	async def warmer():
		return await maybe_warm_cache(
			cache, probe=slow_probe, warming_policy=policy, now=now
		)

	t1 = asyncio.create_task(warmer())
	await started.wait()
	# Second attempt while first holds the lock → skipped as in-progress
	cache2, result2 = await maybe_warm_cache(
		cache, probe=lambda _c: True, warming_policy=policy, now=now
	)
	assert result2.skipped is True
	assert result2.skip_reason == WarmingSkipReason.ALREADY_IN_PROGRESS.value
	release.set()
	cache1, result1 = await t1
	assert result1.warmed is True
	assert cache1.lifetime.warming_in_progress is False


@pytest.mark.asyncio
async def test_ttl_disabled_skips_warming():
	cache = _cache()  # no TTL
	cache = record_cache_use(cache)
	policy = WarmingPolicy(warm_when_remaining_seconds=300, min_use_count=1)
	ok, reason = should_warm(cache, policy)
	assert ok is False
	assert reason == WarmingSkipReason.TTL_DISABLED.value

	async def probe(_c: str) -> bool:
		return True

	cache, result = await maybe_warm_cache(cache, probe=probe, warming_policy=policy)
	assert result.skipped is True
	assert result.skip_reason == WarmingSkipReason.TTL_DISABLED.value


def test_insufficient_use_skips():
	now = datetime(2026, 1, 1, tzinfo=timezone.utc)
	cache = enable_ttl(_cache(), ttl_seconds=60, warming_enabled=True, now=now)
	# use_count == 0
	policy = WarmingPolicy(warm_when_remaining_seconds=300, min_use_count=2)
	ok, reason = should_warm(cache, policy, now=now)
	assert ok is False
	assert reason == WarmingSkipReason.INSUFFICIENT_USE.value
