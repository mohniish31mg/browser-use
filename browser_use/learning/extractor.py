"""AgentHistoryList → CachedAction[] (ActionCache).

Pure offline extraction: no LLM calls, no network I/O.

Pipeline:

    AgentHistoryList
         ↓
    normalize replayable DOM actions (+ record skipped non-DOM)
         ↓
    classify_actions (redundancy / exploration / retry)
         ↓
    derive Playwright locators from captured evidence
         ↓
    ActionCache

Non-DOM exploration (scroll, navigate, done, …) is recorded as
``skipped_actions`` for explainability and never converted to Automation.

Redundancy classification lives in ``learning.redundancy`` (multi-signal,
conservative: when unsure → keep).
"""

from __future__ import annotations

from typing import Any

from browser_use.agent.views import ActionResult, AgentHistory, AgentHistoryList
from browser_use.dom.views import DOMInteractedElement
from browser_use.learning.locator import (
	attribute_value_counts,
	derive_locator,
	locator_confidence,
)
from browser_use.learning.models import (
	ActionCache,
	CacheAction,
	CacheActionResult,
	CacheActionSource,
	CacheLocator,
	CacheStatus,
	CacheTarget,
	SkippedHistoryAction,
	new_step_id,
	utc_now_iso,
)
from browser_use.learning.redundancy import (
	build_classification_report,
	classify_actions,
	same_element,
)

_REPLAYABLE_ACTION_TYPES = frozenset(
	{
		'click',
		'input',
		'select_dropdown',
		'upload_file',
	}
)

_EXPLORATORY_ACTION_TYPES = frozenset(
	{
		'scroll',
		'navigate',
		'go_back',
		'extract',
		'find_text',
		'done',
		'wait',
		'send_keys',
		'switch',
		'close',
		'search',
		'evaluate',
	}
)


def build_action_cache(history: AgentHistoryList) -> ActionCache:
	"""Build an ordered action cache from agent history."""
	raw_entries: list[CacheAction] = []
	skipped: list[SkippedHistoryAction] = []
	sequence = 0

	step_urls: list[str | None] = []
	for history_item in history.history:
		step_urls.append(_page_url_from_history_item(history_item))

	for step_idx, history_item in enumerate(history.history):
		next_url = step_urls[step_idx + 1] if step_idx + 1 < len(step_urls) else None
		for entry in _iter_replayable_actions(history_item, step_idx, next_url=next_url):
			sequence += 1
			entry.sequence = sequence
			raw_entries.append(entry)
		skipped.extend(_iter_skipped_actions(history_item, step_idx))

	classify_actions(raw_entries)
	_attach_derived_locators(raw_entries)
	_finalize_action_metadata(raw_entries)

	source = _source_from_history(history)
	workflow_id = _workflow_id_from_source(source)
	for entry in raw_entries:
		entry.workflow_id = workflow_id

	report = build_classification_report(raw_entries, skipped=skipped)
	return ActionCache(
		version=1,
		source=source,
		actions=raw_entries,
		skipped_actions=skipped,
		workflow_id=workflow_id,
		classification_report=report,
	)


def enrich_cache_locators(cache: ActionCache) -> ActionCache:
	"""Re-derive locators on an existing cache (e.g. after loading JSON)."""
	_attach_derived_locators(cache.actions)
	return cache


def classification_report_from_cache(cache: ActionCache):
	"""Build the interview-facing classification report from a cache."""
	if cache.classification_report is not None:
		return cache.classification_report
	return build_classification_report(cache.actions, skipped=cache.skipped_actions)


def _source_from_history(history: AgentHistoryList) -> dict[str, Any]:
	urls = [u for u in history.urls() if u]
	return {'url': urls[0] if urls else None}


def _page_url_from_history_item(history_item: AgentHistory) -> str | None:
	if history_item.state and history_item.state.url:
		return history_item.state.url
	return None


def _iter_skipped_actions(
	history_item: AgentHistory, step_idx: int
) -> list[SkippedHistoryAction]:
	if not history_item.model_output:
		return []
	page_url = _page_url_from_history_item(history_item)
	out: list[SkippedHistoryAction] = []
	for action_idx, action in enumerate(history_item.model_output.action):
		action_type, _params = _split_action(action)
		if action_type is None or action_type in _REPLAYABLE_ACTION_TYPES:
			continue
		out.append(
			SkippedHistoryAction(
				history_step=step_idx,
				action_index=action_idx,
				action_type=action_type,
				page_url=page_url,
				classification='exploratory',
				classification_reasons=[
					f'non-replayable browser-use action ({action_type})',
					'useful during agentic discovery; not a deterministic Automation node',
				],
			)
		)
	return out


def _iter_replayable_actions(
	history_item: AgentHistory,
	step_idx: int,
	*,
	next_url: str | None,
) -> list[CacheAction]:
	if not history_item.model_output:
		return []

	actions = history_item.model_output.action
	interacted = history_item.state.interacted_element if history_item.state else None
	if interacted is None:
		interacted = [None] * len(actions)
	results = history_item.result or []
	page_url = _page_url_from_history_item(history_item)

	entries: list[CacheAction] = []
	for action_idx, action in enumerate(actions):
		action_type, params = _split_action(action)
		if action_type is None or action_type not in _REPLAYABLE_ACTION_TYPES:
			continue

		element = interacted[action_idx] if action_idx < len(interacted) else None
		if not isinstance(element, DOMInteractedElement):
			continue

		result = results[action_idx] if action_idx < len(results) else None
		parsed_result = _result_from_action_result(result)
		status: CacheStatus = 'kept' if parsed_result.success else 'dropped_failed'
		entries.append(
			CacheAction(
				sequence=0,
				action_type=action_type,
				value=_extract_value(action_type, params),
				target=_target_from_element(element),
				result=parsed_result,
				status=status,
				source=CacheActionSource(
					history_step=step_idx,
					action_index=action_idx,
					backend_node_id=element.backend_node_id,
				),
				page_url=page_url,
				next_page_url=next_url,
				step_id=new_step_id(),
				created_at=utc_now_iso(),
			)
		)
	return entries


def _split_action(action: Any) -> tuple[str | None, dict[str, Any]]:
	if hasattr(action, 'model_dump'):
		action_dict = action.model_dump(exclude_none=True, mode='json')
	elif isinstance(action, dict):
		action_dict = action
	else:
		action_dict = vars(action)

	if not action_dict:
		return None, {}

	action_type = next(iter(action_dict.keys()), None)
	params = action_dict.get(action_type) if action_type else None
	if not isinstance(params, dict):
		params = {}
	return action_type, params


def _extract_value(action_type: str, params: dict[str, Any]) -> str | None:
	if action_type == 'input':
		text = params.get('text')
		return text if isinstance(text, str) else None
	if action_type == 'select_dropdown':
		text = params.get('text')
		return text if isinstance(text, str) else None
	if action_type == 'upload_file':
		path = params.get('path')
		return path if isinstance(path, str) else None
	return None


def _target_from_element(element: DOMInteractedElement) -> CacheTarget:
	return CacheTarget(
		node_name=element.node_name,
		attributes=dict(element.attributes or {}),
		xpath=element.x_path,
		element_hash=element.element_hash,
		stable_hash=element.stable_hash,
		ax_name=element.ax_name,
	)


def _result_from_action_result(result: ActionResult | None) -> CacheActionResult:
	if result is None:
		return CacheActionResult(success=False, error='missing_result')
	if result.error:
		return CacheActionResult(success=False, error=result.error)
	return CacheActionResult(success=True, error=None)


def _attach_derived_locators(entries: list[CacheAction]) -> None:
	unique_targets: list[CacheTarget] = []
	for entry in entries:
		if any(same_element(entry.target, t) for t in unique_targets):
			continue
		unique_targets.append(entry.target)

	counts = attribute_value_counts(
		[{'attributes': t.attributes} for t in unique_targets],
	)

	for entry in entries:
		derived = derive_locator(
			node_name=entry.target.node_name,
			attributes=entry.target.attributes,
			xpath=entry.target.xpath,
			ax_name=entry.target.ax_name,
			sibling_attribute_counts=counts,
		)
		if derived is None:
			entry.locator = None
			if entry.status == 'kept':
				entry.status = 'cache_miss'
				if entry.classification is None:
					entry.classification = 'uncertain'
					entry.classification_reasons = [
						'no reliable locator derived from captured evidence',
					]
			continue

		entry.locator = CacheLocator(
			strategy=derived.strategy,
			command=derived.command,
			evidence=derived.evidence,
		)
		entry.confidence = locator_confidence(derived.strategy)


def _workflow_id_from_source(source: dict[str, Any]) -> str | None:
	url = source.get('url')
	if not url or not isinstance(url, str):
		return None
	import hashlib

	digest = hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]
	return f'wf_{digest}'


def _finalize_action_metadata(entries: list[CacheAction]) -> None:
	for entry in entries:
		if entry.prompt_instructions:
			continue
		if entry.action_type == 'input' and entry.value is not None:
			label = entry.target.ax_name or entry.target.attributes.get('name') or entry.target.node_name
			entry.prompt_instructions = f'Enter "{entry.value}" into the "{label}" field'
		elif entry.action_type == 'click':
			label = entry.target.ax_name or entry.target.attributes.get('aria-label') or entry.target.node_name
			entry.prompt_instructions = f'Click the "{label}" element'
		else:
			entry.prompt_instructions = f'Perform {entry.action_type}'
		if entry.locator and entry.confidence is None:
			entry.confidence = locator_confidence(entry.locator.strategy)
