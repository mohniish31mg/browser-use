"""Unit tests for cache health / degraded-mode state machine (Phase 2)."""

from __future__ import annotations

import copy

import pytest

from browser_use.learning.health import (
	apply_repair_result_to_health,
	begin_repairing,
	build_execution_failure,
	can_attempt_automatic_repair,
	can_attempt_llm_repair,
	explicit_recover_to_healthy,
	mark_degraded,
	note_repair_attempt,
	policy_from_env,
	record_unrecovered_failure,
	request_relearn,
)
from browser_use.learning.models import (
	ActionCache,
	CacheAction,
	CacheActionResult,
	CacheActionSource,
	CacheHealthPolicy,
	CacheHealthState,
	CacheLocator,
	CacheTarget,
	RepairOutcome,
	RepairResult,
)


def _cache(**kwargs) -> ActionCache:
	action = CacheAction(
		sequence=1,
		step_id='step_1',
		action_type='click',
		target=CacheTarget(
			node_name='BUTTON',
			attributes={'id': 'x'},
			xpath='html/body/button',
			element_hash=1,
		),
		result=CacheActionResult(success=True),
		status='kept',
		source=CacheActionSource(history_step=0, action_index=0),
		locator=CacheLocator(strategy='id', command='locator("#x")', evidence={}),
	)
	base = dict(
		version=1,
		source={'url': 'https://example.com'},
		workflow_id='wf_health_test',
		actions=[action],
	)
	base.update(kwargs)
	return ActionCache(**base)


def test_policy_from_env_defaults(monkeypatch):
	monkeypatch.delenv('OPTEXITY_CACHE_FAILURES_TO_DEGRADE', raising=False)
	monkeypatch.delenv('OPTEXITY_CACHE_MAX_REPAIR_ATTEMPTS', raising=False)
	policy = policy_from_env()
	assert policy.failures_to_degrade == 2
	assert policy.max_repair_attempts_per_window == 2
	assert policy.max_llm_repairs_per_window == 1


def test_policy_from_env_overrides(monkeypatch):
	monkeypatch.setenv('OPTEXITY_CACHE_FAILURES_TO_DEGRADE', '3')
	monkeypatch.setenv('OPTEXITY_CACHE_MAX_LLM_REPAIRS', '0')
	monkeypatch.setenv('OPTEXITY_CACHE_FAILURE_WINDOW_SECONDS', '120')
	policy = policy_from_env()
	assert policy.failures_to_degrade == 3
	assert policy.max_llm_repairs_per_window == 0
	assert policy.failure_window_seconds == 120.0


def test_one_failure_stays_healthy_under_threshold():
	policy = CacheHealthPolicy(failures_to_degrade=2)
	cache = begin_repairing(_cache(), step_id='step_1', reason='locator_miss')
	assert cache.health.state == CacheHealthState.REPAIRING
	cache = record_unrecovered_failure(
		cache, policy, step_id='step_1', reason='unrecoverable'
	)
	assert cache.health.state == CacheHealthState.HEALTHY
	assert len(cache.health.failure_timestamps) == 1
	assert cache.health.recommended_action is not None


def test_successful_repair_returns_healthy():
	policy = CacheHealthPolicy()
	cache = begin_repairing(_cache(), step_id='step_1', reason='fail')
	result = RepairResult(
		outcome=RepairOutcome.DETERMINISTIC_REPAIR,
		step_id='step_1',
		new_command='locator("#y")',
	)
	cache, failure = apply_repair_result_to_health(
		cache, policy, result, llm_attempted=False, deterministic_attempted=True
	)
	assert failure is None
	assert cache.health.state == CacheHealthState.HEALTHY


def test_repeated_failure_marks_degraded():
	policy = CacheHealthPolicy(failures_to_degrade=2, max_repair_attempts_per_window=5)
	cache = _cache()
	cache = record_unrecovered_failure(cache, policy, step_id='step_1', reason='fail_1')
	assert cache.health.state == CacheHealthState.HEALTHY
	cache = record_unrecovered_failure(cache, policy, step_id='step_1', reason='fail_2')
	assert cache.health.state == CacheHealthState.DEGRADED
	assert cache.health.recommended_action


def test_degraded_blocks_automatic_repair_and_llm():
	policy = CacheHealthPolicy(allow_automatic_repair_when_degraded=False)
	cache = mark_degraded(_cache(), step_id='step_1', reason='exhausted')
	ok, reason = can_attempt_automatic_repair(cache, policy)
	assert ok is False
	assert reason == 'cache_degraded'
	ok_llm, reason_llm = can_attempt_llm_repair(cache, policy)
	assert ok_llm is False
	assert reason_llm == 'cache_degraded'


def test_rate_limit_blocks_repeated_llm_without_full_agentic():
	policy = CacheHealthPolicy(max_repair_attempts_per_window=2, max_llm_repairs_per_window=1)
	cache = _cache()
	cache = note_repair_attempt(cache, llm=True)
	# One LLM repair already used in window
	ok_llm, reason = can_attempt_llm_repair(cache, policy)
	assert ok_llm is False
	assert reason == 'llm_repair_rate_limited'

	# Repair attempts exhausted → blocked (no full LLM path exposed here)
	cache = note_repair_attempt(cache, llm=False)
	ok, reason = can_attempt_automatic_repair(cache, policy)
	assert ok is False
	assert reason == 'repair_rate_limited'

	failure = build_execution_failure(
		cache,
		step_id='step_1',
		failure_reason='rate_limited',
		deterministic_repair_attempted=True,
		llm_repair_attempted=True,
	)
	assert failure.full_agentic_fallback is False
	assert failure.cache_state in {
		CacheHealthState.HEALTHY,
		CacheHealthState.REPAIRING,
		CacheHealthState.DEGRADED,
	}


def test_explicit_recovery_path():
	cache = mark_degraded(_cache(), step_id='step_1', reason='broken')
	cache = request_relearn(cache)
	assert cache.health.relearn_requested is True
	assert cache.health.state == CacheHealthState.DEGRADED
	# After operator relearns offline, explicit recover clears circuit.
	before_actions = copy.deepcopy([a.model_dump(mode='json') for a in cache.actions])
	cache = explicit_recover_to_healthy(cache)
	assert cache.health.state == CacheHealthState.HEALTHY
	assert cache.health.relearn_requested is False
	assert cache.health.failure_timestamps == []
	assert [a.model_dump(mode='json') for a in cache.actions] == before_actions


def test_structured_failure_never_enables_full_agentic():
	cache = mark_degraded(_cache(), step_id='step_1', reason='x')
	failure = build_execution_failure(
		cache,
		step_id='step_1',
		failure_reason='locator_failed',
		deterministic_repair_attempted=True,
		llm_repair_attempted=True,
		repair_outcome=RepairOutcome.UNRECOVERABLE,
	)
	assert failure.full_agentic_fallback is False
	assert failure.workflow_id == 'wf_health_test'
	assert failure.step_id == 'step_1'
	assert failure.deterministic_repair_attempted is True
	assert failure.llm_repair_attempted is True
	assert failure.cache_state == CacheHealthState.DEGRADED
	assert 'relearn' in failure.recommended_next_action or 'repair' in failure.recommended_next_action


@pytest.mark.asyncio
async def test_no_repeated_full_llm_on_degraded_runtime(monkeypatch, tmp_path):
	"""Runtime must not invoke targeted LLM (let alone full workflow) when DEGRADED."""
	from optexity.inference.core import cache_repair_runtime as runtime

	cache = mark_degraded(_cache(), step_id='step_1', reason='prior')
	path = tmp_path / 'action_cache.json'
	from browser_use.learning.cache import save_action_cache

	save_action_cache(cache, path)

	monkeypatch.setenv('OPTEXITY_CACHE_REPAIR', '1')
	monkeypatch.setenv('OPTEXITY_ACTION_CACHE_PATH', str(path))
	monkeypatch.setenv('OPTEXITY_CACHE_ALLOW_REPAIR_WHEN_DEGRADED', '0')

	llm_calls = {'n': 0}

	async def boom_llm(*_a, **_k):
		llm_calls['n'] += 1
		raise AssertionError('LLM must not be called when degraded')

	monkeypatch.setattr(runtime, '_targeted_llm_repair', boom_llm)

	class _Action:
		command = 'locator("#x")'
		skip_prompt = True
		prompt_instructions = '[cache_step:step_1] Click'

	class _Browser:
		pass

	result = await runtime.handle_cached_command_failure(
		_Action(),  # type: ignore[arg-type]
		browser=_Browser(),  # type: ignore[arg-type]
		task=None,  # type: ignore[arg-type]
		memory=None,  # type: ignore[arg-type]
		failure_error='locator_failed',
	)
	assert result.new_command is None
	assert result.failure is not None
	assert result.failure.full_agentic_fallback is False
	assert result.failure.cache_state == CacheHealthState.DEGRADED
	assert llm_calls['n'] == 0
	assert result.failure.llm_repair_attempted is False
