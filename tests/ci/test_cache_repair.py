"""Unit tests for minimal single-step cache repair."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from browser_use.learning.models import (
	ActionCache,
	CacheAction,
	CacheActionResult,
	CacheActionSource,
	CacheLocator,
	CacheTarget,
	RepairOutcome,
)
from browser_use.learning.repair import (
	alternatives_for_action,
	apply_locator_update,
	build_targeted_repair_prompt,
	repair_failed_step,
)


def _action(
	*,
	step_id: str,
	sequence: int,
	command: str,
	strategy: str = 'id',
	attrs: dict | None = None,
	ax_name: str | None = None,
	xpath: str = 'html/body/button',
	action_type: str = 'click',
	value: str | None = None,
	element_hash: int = 1,
) -> CacheAction:
	return CacheAction(
		sequence=sequence,
		step_id=step_id,
		action_type=action_type,
		value=value,
		target=CacheTarget(
			node_name='BUTTON' if action_type == 'click' else 'INPUT',
			attributes=attrs or {'id': 'primary', 'data-testid': 'primary-btn'},
			xpath=xpath,
			element_hash=element_hash,
			stable_hash=element_hash,
			ax_name=ax_name or 'Primary',
		),
		result=CacheActionResult(success=True),
		status='kept',
		source=CacheActionSource(history_step=0, action_index=0),
		locator=CacheLocator(strategy=strategy, command=command, evidence={'id': 'primary'}),
		page_url='https://example.com/app',
		prompt_instructions=f'Do {action_type}',
		workflow_id='wf_test',
	)


def _cache(actions: list[CacheAction]) -> ActionCache:
	return ActionCache(version=1, source={'url': 'https://example.com/app'}, actions=actions, workflow_id='wf_test')


@pytest.mark.asyncio
async def test_cached_action_succeeds_as_cache_hit():
	action = _action(step_id='step_a', sequence=1, command='locator("#primary")')
	cache = _cache([action])

	async def probe(cmd: str) -> bool:
		return cmd == 'locator("#primary")'

	new_cache, result = await repair_failed_step(
		cache,
		'step_a',
		failure_error='transient',
		probe=probe,
		allow_llm=False,
	)
	assert result.outcome == RepairOutcome.CACHE_HIT
	assert result.new_command == 'locator("#primary")'
	assert new_cache.actions[0].success_count == 1
	assert new_cache.version == 1  # no locator rewrite


@pytest.mark.asyncio
async def test_deterministic_alternative_repairs_only_failed_step():
	# Primary id locator "broken"; data-testid alternative still in evidence.
	failed = _action(
		step_id='step_fail',
		sequence=2,
		command='locator("#gone")',
		strategy='id',
		attrs={'id': 'gone', 'data-testid': 'still-there', 'aria-label': 'Go'},
		ax_name='Go',
		element_hash=20,
	)
	ok_before = _action(
		step_id='step_ok1',
		sequence=1,
		command='locator("#login")',
		attrs={'id': 'login'},
		element_hash=10,
	)
	ok_after = _action(
		step_id='step_ok3',
		sequence=3,
		command='locator("#cart")',
		attrs={'id': 'cart'},
		element_hash=30,
	)
	# Make failed locator evidence include test id (override attrs already set).
	failed.target.attributes = {'id': 'stale-id', 'data-testid': 'still-there', 'aria-label': 'Go'}
	# Current command is stale id; alternatives should include test_id / role.
	failed.locator = CacheLocator(
		strategy='id',
		command='locator("#stale-id")',
		evidence={'id': 'stale-id'},
	)

	cache = _cache([ok_before, failed, ok_after])
	before_dump = [a.model_dump(mode='json') for a in cache.actions]

	async def probe(cmd: str) -> bool:
		# Broken primary; working alternative from captured evidence.
		return cmd == 'get_by_test_id("still-there")'

	new_cache, result = await repair_failed_step(
		cache,
		'step_fail',
		failure_error='Timeout: locator("#stale-id")',
		probe=probe,
		allow_llm=False,
	)

	assert result.outcome == RepairOutcome.DETERMINISTIC_REPAIR
	assert result.new_command == 'get_by_test_id("still-there")'
	assert new_cache.version == 2

	# Only failed entry changed (locator + history/stats/timestamps).
	repaired = next(a for a in new_cache.actions if a.step_id == 'step_fail')
	assert repaired.locator is not None
	assert repaired.locator.command == 'get_by_test_id("still-there")'
	assert len(repaired.locator_history) == 1
	assert repaired.locator_history[0].locator.command == 'locator("#stale-id")'

	untouched_ids = {'step_ok1', 'step_ok3'}
	for action in new_cache.actions:
		if action.step_id not in untouched_ids:
			continue
		original = next(x for x in before_dump if x['step_id'] == action.step_id)
		# failure_count only increments on the failed step
		assert action.model_dump(mode='json') == original


@pytest.mark.asyncio
async def test_targeted_llm_repair_updates_only_failed_entry():
	failed = _action(
		step_id='step_llm',
		sequence=1,
		command='locator("#missing")',
		attrs={'id': 'missing'},  # no useful alternatives → LLM path
		element_hash=99,
	)
	# Force no deterministic alternatives: strip attrs/xpath/ax so list is empty-ish
	failed.target.attributes = {}
	failed.target.xpath = ''
	failed.target.ax_name = None
	failed.locator = CacheLocator(strategy='id', command='locator("#missing")', evidence={})

	peer = _action(step_id='step_peer', sequence=2, command='locator("#peer")', element_hash=2)
	cache = _cache([failed, peer])
	peer_before = copy.deepcopy(peer.model_dump(mode='json'))

	async def probe(cmd: str) -> bool:
		return cmd == 'locator("#recovered")'

	async def llm_repair(action, prompt: str):
		assert 'Do NOT rediscover' in prompt or 'ONLY' in prompt.upper() or 'only' in prompt.lower()
		assert action.step_id == 'step_llm'
		assert 'step_peer' in prompt  # nearby context present
		# Must not ask to redo peer as the objective — peer appears only as context.
		assert 'rediscover the whole workflow' in prompt.lower() or 'NOT rediscover' in prompt
		return CacheLocator(strategy='id', command='locator("#recovered")', evidence={'observed': True})

	new_cache, result = await repair_failed_step(
		cache,
		'step_llm',
		failure_error='not found',
		probe=probe,
		llm_repair=llm_repair,
		allow_llm=True,
	)

	assert result.outcome == RepairOutcome.LLM_REPAIR
	assert result.new_command == 'locator("#recovered")'
	assert new_cache.version == 2
	peer_after = next(a for a in new_cache.actions if a.step_id == 'step_peer')
	assert peer_after.model_dump(mode='json') == peer_before


@pytest.mark.asyncio
async def test_unrecoverable_does_not_rediscover_workflow():
	failed = _action(
		step_id='step_dead',
		sequence=1,
		command='locator("#gone")',
		attrs={},
		element_hash=1,
	)
	failed.target.attributes = {}
	failed.target.xpath = ''
	failed.target.ax_name = None
	others = [
		_action(step_id='step_2', sequence=2, command='locator("#b")', element_hash=2),
		_action(step_id='step_3', sequence=3, command='locator("#c")', element_hash=3),
	]
	cache = _cache([failed, *others])
	others_before = [copy.deepcopy(a.model_dump(mode='json')) for a in others]
	llm_calls = {'n': 0}

	async def probe(cmd: str) -> bool:
		return False

	async def llm_repair(action, prompt: str):
		llm_calls['n'] += 1
		return None  # no observed locator

	new_cache, result = await repair_failed_step(
		cache,
		'step_dead',
		failure_error='gone',
		probe=probe,
		llm_repair=llm_repair,
		allow_llm=True,
	)

	assert result.outcome == RepairOutcome.UNRECOVERABLE
	assert llm_calls['n'] == 1  # targeted attempt only — not one call per workflow step
	assert 'not rediscovered' in (result.message or '').lower() or result.error
	# Unaffected steps byte-equivalent
	for before in others_before:
		after = next(a for a in new_cache.actions if a.step_id == before['step_id'])
		assert after.model_dump(mode='json') == before
	# Version unchanged when no locator written
	assert new_cache.version == 1


def test_alternatives_do_not_include_current_command():
	action = _action(
		step_id='s',
		sequence=1,
		command='locator("#primary")',
		attrs={'id': 'primary', 'data-testid': 'primary-btn'},
	)
	alts = alternatives_for_action(action)
	cmds = [a.command for a in alts]
	assert 'locator("#primary")' not in cmds
	assert any('primary-btn' in c or 'test_id' in c or 'get_by_' in c for c in cmds)


def test_apply_locator_update_preserves_history_and_peers(tmp_path: Path):
	a1 = _action(step_id='s1', sequence=1, command='locator("#a")', element_hash=1)
	a2 = _action(step_id='s2', sequence=2, command='locator("#b")', element_hash=2)
	cache = _cache([a1, a2])
	updated = apply_locator_update(
		cache,
		's1',
		CacheLocator(strategy='name', command='locator("button[name=\'x\']")', evidence={}),
		reason='test',
	)
	assert updated.version == 2
	assert updated.actions[0].locator_history[0].locator.command == 'locator("#a")'
	assert updated.actions[1].locator.command == 'locator("#b")'

	# Round-trip JSON
	path = tmp_path / 'cache.json'
	path.write_text(json.dumps(updated.model_dump(mode='json')), encoding='utf-8')
	loaded = ActionCache.model_validate(json.loads(path.read_text(encoding='utf-8')))
	assert loaded.actions[0].step_id == 's1'


def test_targeted_prompt_scopes_to_one_step():
	failed = _action(step_id='step_4', sequence=4, command='locator("#x")')
	peers = [
		_action(step_id='step_3', sequence=3, command='locator("#y")', element_hash=3),
		failed,
		_action(step_id='step_5', sequence=5, command='locator("#z")', element_hash=5),
	]
	prompt = build_targeted_repair_prompt(
		failed, failure_error='boom', nearby_context=peers, page_url='https://example.com'
	)
	assert 'step_4' in prompt
	assert 'ONLY' in prompt.upper() or 'only' in prompt
	assert 'NOT rediscover' in prompt or 'Do NOT rediscover' in prompt
	assert '<<FAILED>>' in prompt
