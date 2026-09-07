"""Unit tests for action cache extraction from AgentHistoryList."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from browser_use.learning import (
	build_action_cache,
	build_and_save_action_cache,
	save_action_cache,
)
from browser_use.agent.service import Agent
from browser_use.agent.views import ActionResult, AgentHistory, AgentHistoryList, BrowserStateHistory, StepMetadata
from browser_use.dom.views import DOMInteractedElement, DOMRect, NodeType


@pytest.fixture
def AgentOutput():
	llm = MagicMock()
	agent = Agent(task='cache test', llm=llm)
	return agent.AgentOutput


def _element(
	*,
	backend_node_id: int = 1,
	node_name: str = 'INPUT',
	attributes: dict[str, str] | None = None,
	xpath: str = 'html/body/input',
	element_hash: int = 111,
	stable_hash: int | None = 222,
	ax_name: str | None = None,
) -> DOMInteractedElement:
	return DOMInteractedElement(
		node_id=1,
		backend_node_id=backend_node_id,
		frame_id=None,
		node_type=NodeType.ELEMENT_NODE,
		node_value='',
		node_name=node_name,
		attributes=attributes or {'name': 'field', 'type': 'text'},
		x_path=xpath,
		element_hash=element_hash,
		stable_hash=stable_hash,
		bounds=DOMRect(x=0, y=0, width=10, height=10),
		ax_name=ax_name,
	)


def _history_item(
	AgentOutput,
	*,
	actions: list[dict],
	elements: list[DOMInteractedElement | None],
	results: list[ActionResult],
	url: str = 'https://example.com/form',
	step_number: int = 1,
) -> AgentHistory:
	return AgentHistory(
		model_output=AgentOutput(
			evaluation_previous_goal=None,
			memory='test',
			next_goal=None,
			action=actions,  # type: ignore[arg-type]
		),
		result=results,
		state=BrowserStateHistory(
			url=url,
			title='Test',
			tabs=[],
			interacted_element=elements,
		),
		metadata=StepMetadata(
			step_start_time=0,
			step_end_time=1,
			step_number=step_number,
			step_interval=0.1,
		),
	)


def test_extracts_input_with_value_and_target_identity(AgentOutput):
	el = _element(attributes={'name': '04fullname', 'type': 'text', 'id': 'fn'}, xpath='html/body/form/input')
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'input': {'index': 20, 'text': 'myname', 'clear': True}}],
				elements=[el],
				results=[ActionResult(long_term_memory="Typed 'myname'")],
			)
		]
	)

	cache = build_action_cache(history)

	assert cache.version == 1
	assert cache.source['url'] == 'https://example.com/form'
	assert len(cache.actions) == 1
	action = cache.actions[0]
	assert action.sequence == 1
	assert action.action_type == 'input'
	assert action.value == 'myname'
	assert action.target.node_name == 'INPUT'
	assert action.target.attributes['name'] == '04fullname'
	assert action.target.xpath == 'html/body/form/input'
	assert action.target.element_hash == 111
	assert action.target.stable_hash == 222
	assert action.result.success is True
	assert action.status == 'kept'
	assert action.source.backend_node_id == 1
	assert 'index' not in action.model_dump()['target']
	# Biggest rule: never cache the ephemeral browser-use highlight index as identity.
	dumped = action.model_dump(mode='json')
	assert 'index' not in dumped
	assert dumped['locator']['command'].startswith('locator(')
	assert action.locator is not None
	assert action.locator.strategy == 'id'  # id preferred over name when both present
	assert action.locator.command == 'locator("#fn")'
	assert action.locator.evidence['id'] == 'fn'
	assert action.page_url == 'https://example.com/form'


def test_skips_non_dom_actions_and_preserves_order(AgentOutput):
	el_a = _element(element_hash=1, stable_hash=1, xpath='html/body/input[1]', attributes={'name': 'a'})
	el_b = _element(element_hash=2, stable_hash=2, xpath='html/body/input[2]', attributes={'name': 'b'})
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[
					{'navigate': {'url': 'https://example.com'}},
					{'input': {'index': 1, 'text': 'first'}},
					{'scroll': {'down': True}},
					{'input': {'index': 2, 'text': 'second'}},
					{'done': {'text': 'done', 'success': True}},
				],
				elements=[None, el_a, None, el_b, None],
				results=[
					ActionResult(long_term_memory='nav'),
					ActionResult(long_term_memory='a'),
					ActionResult(long_term_memory='scroll'),
					ActionResult(long_term_memory='b'),
					ActionResult(long_term_memory='done', is_done=True, success=True),
				],
			)
		]
	)

	cache = build_action_cache(history)
	assert [a.action_type for a in cache.actions] == ['input', 'input']
	assert [a.value for a in cache.actions] == ['first', 'second']
	assert [a.sequence for a in cache.actions] == [1, 2]


def test_records_failure(AgentOutput):
	el = _element()
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[el],
				results=[ActionResult(error='Element not found', long_term_memory=None)],
			)
		]
	)

	cache = build_action_cache(history)
	assert len(cache.actions) == 1
	assert cache.actions[0].result.success is False
	assert cache.actions[0].result.error == 'Element not found'
	assert cache.actions[0].status == 'dropped_failed'


def test_marks_redundant_input_retries_by_element_not_index(AgentOutput):
	el1 = _element(backend_node_id=20, element_hash=999, stable_hash=888, xpath='html/body/input')
	el2 = _element(backend_node_id=99, element_hash=999, stable_hash=888, xpath='html/body/input')

	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'input': {'index': 20, 'text': 'myname'}}],
				elements=[el1],
				results=[ActionResult(long_term_memory='ok')],
				step_number=1,
			),
			_history_item(
				AgentOutput,
				actions=[{'input': {'index': 99, 'text': 'myname'}}],
				elements=[el2],
				results=[ActionResult(long_term_memory='retry')],
				step_number=2,
			),
		]
	)

	cache = build_action_cache(history)
	assert len(cache.actions) == 2
	assert cache.actions[0].status == 'kept'
	assert cache.actions[1].status == 'dropped_redundant'


def test_different_value_on_same_element_keeps_last_only(AgentOutput):
	el = _element(element_hash=5, stable_hash=5, xpath='html/body/input')
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'input': {'index': 1, 'text': 'wrong'}}],
				elements=[el],
				results=[ActionResult(long_term_memory='ok')],
			),
			_history_item(
				AgentOutput,
				actions=[{'input': {'index': 1, 'text': 'right'}}],
				elements=[el],
				results=[ActionResult(long_term_memory='ok')],
			),
		]
	)

	cache = build_action_cache(history)
	assert [a.status for a in cache.actions] == ['dropped_superseded', 'kept']
	assert [a.value for a in cache.actions] == ['wrong', 'right']


def test_failed_then_success_keeps_only_success(AgentOutput):
	el = _element(element_hash=7, stable_hash=7, xpath='html/body/button', node_name='BUTTON')
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[el],
				results=[ActionResult(error='timeout', long_term_memory=None)],
			),
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[el],
				results=[ActionResult(long_term_memory='ok')],
			),
		]
	)
	cache = build_action_cache(history)
	assert [a.status for a in cache.actions] == ['dropped_failed', 'kept']


def test_skips_actions_without_interacted_element(AgentOutput):
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'input': {'index': 1, 'text': 'x'}}],
				elements=[None],
				results=[ActionResult(long_term_memory='ok')],
			)
		]
	)
	cache = build_action_cache(history)
	assert cache.actions == []


def test_save_action_cache_writes_json(AgentOutput, tmp_path: Path):
	el = _element(attributes={'name': 'city', 'type': 'text'})
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'input': {'index': 1, 'text': 'SF'}}],
				elements=[el],
				results=[ActionResult(long_term_memory='ok')],
				url='https://www.roboform.com/filling-test-all-fields',
			)
		]
	)
	out = tmp_path / 'action_cache.json'
	cache = build_and_save_action_cache(history, out)

	assert out.exists()
	loaded = json.loads(out.read_text(encoding='utf-8'))
	assert loaded['version'] == 1
	assert loaded['source']['url'] == 'https://www.roboform.com/filling-test-all-fields'
	assert loaded['actions'][0]['value'] == 'SF'
	assert loaded['actions'][0]['target']['attributes']['name'] == 'city'
	assert save_action_cache(cache, tmp_path / 'again.json').exists()


def test_click_redundant_retry_via_xpath_when_hashes_differ(AgentOutput):
	el1 = _element(
		node_name='BUTTON',
		element_hash=1,
		stable_hash=None,
		xpath='html/body/button',
		attributes={'id': 'go'},
	)
	el2 = _element(
		node_name='BUTTON',
		element_hash=2,
		stable_hash=None,
		xpath='html/body/button',
		attributes={'id': 'go'},
	)
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[el1],
				results=[ActionResult(long_term_memory='ok')],
			),
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 2}}],
				elements=[el2],
				results=[ActionResult(long_term_memory='retry')],
			),
		]
	)
	cache = build_action_cache(history)
	assert cache.actions[0].status == 'kept'
	assert cache.actions[1].status == 'dropped_redundant'
