"""Minimal single-step cache repair (additive; does not rewrite the extract pipeline).

Flow:
  cached step fails
    → try locator alternatives from captured evidence (deterministic)
    → else targeted LLM repair for THIS step only
    → validate replacement
    → update ONLY that cache entry (+ version bump)
    → leave all other steps unchanged

Never invents selectors. Never rediscovers the whole workflow.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from browser_use.learning.events import (
	CACHE_HIT,
	CACHE_MISS,
	CACHE_REPAIR_DETERMINISTIC_SUCCESS,
	CACHE_REPAIR_FAILED,
	CACHE_REPAIR_LLM_SUCCESS,
	CACHE_REPAIR_STARTED,
	log_cache_event,
)
from browser_use.learning.locator import (
	LocatorCandidate,
	list_locator_alternatives,
	locator_confidence,
)
from browser_use.learning.models import (
	ActionCache,
	CacheAction,
	CacheLocator,
	LocatorHistoryEntry,
	RepairOutcome,
	RepairResult,
	utc_now_iso,
)

# Probe: return True if command resolves + is actionable on the live page.
LocatorProbe = Callable[[str], Awaitable[bool] | bool]
# Targeted LLM: given a single-step repair prompt + failed action, return a
# locator observed from a successful one-step agent run (or None).
TargetedLlmRepair = Callable[[CacheAction, str], Awaitable[CacheLocator | None] | CacheLocator | None]


def build_targeted_repair_prompt(
	action: CacheAction,
	*,
	failure_error: str | None,
	nearby_context: Sequence[CacheAction] | None = None,
	page_url: str | None = None,
) -> str:
	"""Prompt that asks the LLM to replace ONLY this failed step — not the workflow."""
	prev = action.locator.command if action.locator else None
	context_lines: list[str] = []
	if nearby_context:
		for peer in nearby_context:
			marker = '<<FAILED>>' if peer.step_id == action.step_id else 'ok'
			cmd = peer.locator.command if peer.locator else None
			context_lines.append(
				f'- [{marker}] step_id={peer.step_id} type={peer.action_type} '
				f'value={peer.value!r} command={cmd!r}'
			)
	context_block = '\n'.join(context_lines) if context_lines else '(no nearby steps)'

	expected = action.prompt_instructions or _default_expected(action)
	url = page_url or action.page_url or '(unknown)'

	return (
		'You are repairing ONE failed step in an otherwise working automation.\n'
		'Do NOT rediscover or re-execute the whole workflow.\n'
		'Do NOT repeat earlier or later steps.\n'
		'Find and perform ONLY the replacement for this single failed step, then stop.\n\n'
		f'Failed step_id: {action.step_id}\n'
		f'Action type: {action.action_type}\n'
		f'Expected action: {expected}\n'
		f'Value (if input): {action.value!r}\n'
		f'Previous locator/command: {prev!r}\n'
		f'Failure error: {failure_error!r}\n'
		f'Current page URL context: {url}\n\n'
		f'Nearby workflow context (for orientation only — do not re-do other steps):\n'
		f'{context_block}\n\n'
		'Use the current page state to identify the correct element for THIS step only.'
	)


def _default_expected(action: CacheAction) -> str:
	if action.action_type == 'input':
		return f'Fill the target field with {action.value!r}'
	if action.action_type == 'click':
		label = action.target.ax_name or action.target.attributes.get('aria-label') or action.target.node_name
		return f'Click the element ({label})'
	return f'Perform {action.action_type}'


def alternatives_for_action(
	action: CacheAction,
	*,
	sibling_attribute_counts: dict[tuple[str, str], int] | None = None,
) -> list[LocatorCandidate]:
	"""Cheap deterministic candidates from existing captured evidence only."""
	exclude: set[str] = set()
	if action.locator and action.locator.command:
		exclude.add(action.locator.command)
	return list_locator_alternatives(
		node_name=action.target.node_name,
		attributes=action.target.attributes,
		xpath=action.target.xpath,
		ax_name=action.target.ax_name,
		sibling_attribute_counts=sibling_attribute_counts,
		exclude_commands=exclude,
	)


def find_action_by_step_id(cache: ActionCache, step_id: str) -> CacheAction | None:
	for action in cache.actions:
		if action.step_id == step_id:
			return action
	return None


def apply_locator_update(
	cache: ActionCache,
	step_id: str,
	new_locator: CacheLocator,
	*,
	reason: str,
	bump_version: bool = True,
) -> ActionCache:
	"""Update ONLY the matching step; retain prior locator in locator_history."""
	updated_actions: list[CacheAction] = []
	found = False
	for action in cache.actions:
		if action.step_id != step_id:
			updated_actions.append(action)
			continue
		found = True
		history = list(action.locator_history)
		if action.locator is not None:
			history.append(
				LocatorHistoryEntry(locator=action.locator, reason=reason, replaced_at=utc_now_iso())
			)
		updated_actions.append(
			action.model_copy(
				update={
					'locator': new_locator,
					'locator_history': history,
					'updated_at': utc_now_iso(),
					'confidence': locator_confidence(new_locator.strategy),
					'prompt_instructions': action.prompt_instructions
					or _default_expected(action),
				}
			)
		)
	if not found:
		raise KeyError(f'No cache action with step_id={step_id!r}')

	new_version = cache.version + 1 if bump_version else cache.version
	return cache.model_copy(update={'actions': updated_actions, 'version': new_version})


def record_step_success(cache: ActionCache, step_id: str) -> ActionCache:
	actions = [
		a.model_copy(update={'success_count': a.success_count + 1}) if a.step_id == step_id else a
		for a in cache.actions
	]
	return cache.model_copy(update={'actions': actions})


def record_step_failure(cache: ActionCache, step_id: str) -> ActionCache:
	actions = [
		a.model_copy(update={'failure_count': a.failure_count + 1}) if a.step_id == step_id else a
		for a in cache.actions
	]
	return cache.model_copy(update={'actions': actions})


async def _maybe_await(value: Any) -> Any:
	if hasattr(value, '__await__'):
		return await value
	return value


async def repair_failed_step(
	cache: ActionCache,
	step_id: str,
	*,
	failure_error: str | None,
	probe: LocatorProbe,
	llm_repair: TargetedLlmRepair | None = None,
	page_url: str | None = None,
	allow_llm: bool = True,
) -> tuple[ActionCache, RepairResult]:
	"""Repair a single failed step in-place on the cache.

	``probe(command)`` must return True only when the command works on the live page.
	``llm_repair`` must return a locator observed from a successful targeted agent run
	(never a fabricated selector).
	"""
	action = find_action_by_step_id(cache, step_id)
	if action is None:
		log_cache_event(CACHE_REPAIR_FAILED, step_id=step_id, reason='unknown_step_id')
		return cache, RepairResult(
			outcome=RepairOutcome.UNRECOVERABLE,
			step_id=step_id,
			workflow_id=cache.workflow_id,
			error='unknown_step_id',
			message='Failed step_id not found in cache; workflow not rediscovered',
		)

	prev_cmd = action.locator.command if action.locator else None
	version_before = cache.version
	cache = record_step_failure(cache, step_id)
	action = find_action_by_step_id(cache, step_id)
	assert action is not None

	log_cache_event(
		CACHE_REPAIR_STARTED,
		step_id=step_id,
		workflow_id=cache.workflow_id,
		action_type=action.action_type,
		previous_command=prev_cmd,
		error=failure_error,
	)

	# If primary still works, treat as cache hit (caller may have raced).
	# Never treat a known strict-mode collision as a hit — probe must reject it.
	strict_collision = bool(
		failure_error and 'strict mode violation' in failure_error.lower()
	)
	if (
		prev_cmd
		and not strict_collision
		and await _maybe_await(probe(prev_cmd))
	):
		log_cache_event(CACHE_HIT, step_id=step_id, command=prev_cmd)
		cache = record_step_success(cache, step_id)
		return cache, RepairResult(
			outcome=RepairOutcome.CACHE_HIT,
			step_id=step_id,
			workflow_id=cache.workflow_id,
			previous_command=prev_cmd,
			new_command=prev_cmd,
			new_locator=action.locator,
			cache_version_before=version_before,
			cache_version_after=cache.version,
			message='Original locator succeeded on re-probe',
		)

	tried: list[str] = []
	alternatives = alternatives_for_action(action)
	for cand in alternatives:
		tried.append(cand.command)
		ok = await _maybe_await(probe(cand.command))
		if not ok:
			continue
		new_locator = CacheLocator(
			strategy=cand.strategy,
			command=cand.command,
			evidence=cand.evidence,
		)
		cache = apply_locator_update(
			cache, step_id, new_locator, reason='deterministic_alternative'
		)
		cache = record_step_success(cache, step_id)
		log_cache_event(
			CACHE_REPAIR_DETERMINISTIC_SUCCESS,
			step_id=step_id,
			new_command=cand.command,
			strategy=cand.strategy,
		)
		return cache, RepairResult(
			outcome=RepairOutcome.DETERMINISTIC_REPAIR,
			step_id=step_id,
			workflow_id=cache.workflow_id,
			previous_command=prev_cmd,
			new_command=cand.command,
			new_locator=new_locator,
			tried_alternatives=tried,
			cache_version_before=version_before,
			cache_version_after=cache.version,
			message='Repaired via captured locator alternative (no LLM)',
		)

	if not allow_llm or llm_repair is None:
		log_cache_event(
			CACHE_REPAIR_FAILED,
			step_id=step_id,
			reason='deterministic_exhausted',
			tried=len(tried),
		)
		log_cache_event(CACHE_MISS, step_id=step_id)
		return cache, RepairResult(
			outcome=RepairOutcome.UNRECOVERABLE,
			step_id=step_id,
			workflow_id=cache.workflow_id,
			previous_command=prev_cmd,
			error=failure_error or 'deterministic_exhausted',
			tried_alternatives=tried,
			cache_version_before=version_before,
			cache_version_after=cache.version,
			message='Deterministic alternatives exhausted; LLM repair disabled or unavailable',
		)

	nearby = [a for a in cache.actions if a.status == 'kept']
	prompt = build_targeted_repair_prompt(
		action,
		failure_error=failure_error,
		nearby_context=nearby,
		page_url=page_url,
	)
	observed = await _maybe_await(llm_repair(action, prompt))
	if observed is None or not observed.command:
		log_cache_event(CACHE_REPAIR_FAILED, step_id=step_id, reason='llm_no_locator')
		log_cache_event(CACHE_MISS, step_id=step_id)
		return cache, RepairResult(
			outcome=RepairOutcome.UNRECOVERABLE,
			step_id=step_id,
			workflow_id=cache.workflow_id,
			previous_command=prev_cmd,
			error='llm_repair_returned_no_locator',
			tried_alternatives=tried,
			cache_version_before=version_before,
			cache_version_after=cache.version,
			message='Targeted LLM repair produced no observed locator; workflow not rediscovered',
		)

	# Validate LLM-observed locator on the live page before accepting.
	tried.append(observed.command)
	if not await _maybe_await(probe(observed.command)):
		log_cache_event(CACHE_REPAIR_FAILED, step_id=step_id, reason='llm_locator_probe_failed')
		log_cache_event(CACHE_MISS, step_id=step_id)
		return cache, RepairResult(
			outcome=RepairOutcome.UNRECOVERABLE,
			step_id=step_id,
			workflow_id=cache.workflow_id,
			previous_command=prev_cmd,
			error='llm_locator_failed_validation',
			tried_alternatives=tried,
			cache_version_before=version_before,
			cache_version_after=cache.version,
			message='LLM locator did not validate on page; not writing invented/unproven command',
		)

	cache = apply_locator_update(cache, step_id, observed, reason='targeted_llm_repair')
	cache = record_step_success(cache, step_id)
	log_cache_event(
		CACHE_REPAIR_LLM_SUCCESS,
		step_id=step_id,
		new_command=observed.command,
		strategy=observed.strategy,
	)
	return cache, RepairResult(
		outcome=RepairOutcome.LLM_REPAIR,
		step_id=step_id,
		workflow_id=cache.workflow_id,
		previous_command=prev_cmd,
		new_command=observed.command,
		new_locator=observed,
		tried_alternatives=tried,
		cache_version_before=version_before,
		cache_version_after=cache.version,
		message='Repaired via targeted single-step LLM; other steps unchanged',
	)


def extract_locator_from_targeted_history(
	history: Any,
	*,
	expected_action_type: str | None = None,
) -> CacheLocator | None:
	"""Derive a locator from a one-step agent history (observed only — no invention).

	Uses the same offline extractor as the main pipeline, then picks the last
	successful kept action (optionally filtered by type).
	"""
	from browser_use.learning.extractor import build_action_cache

	mini = build_action_cache(history)
	kept = [a for a in mini.actions if a.status == 'kept' and a.locator and a.locator.command]
	if expected_action_type:
		typed = [a for a in kept if a.action_type == expected_action_type]
		if typed:
			kept = typed
	if not kept:
		return None
	return kept[-1].locator
