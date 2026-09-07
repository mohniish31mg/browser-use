"""Tests for multi-signal redundancy / exploration classification."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from browser_use.agent.service import Agent
from browser_use.agent.views import ActionResult, AgentHistory, AgentHistoryList, BrowserStateHistory, StepMetadata
from browser_use.dom.views import DOMInteractedElement, DOMRect, NodeType
from browser_use.learning import (
	build_action_cache,
	classification_report_from_cache,
	validate_removal_candidates,
)
from browser_use.learning.models import CacheAction, CacheActionResult, CacheActionSource, CacheTarget
from browser_use.learning.redundancy import classify_actions


@pytest.fixture
def AgentOutput():
	llm = MagicMock()
	agent = Agent(task='redundancy test', llm=llm)
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


def test_1_duplicate_observation_recorded_as_exploratory_skipped(AgentOutput):
	el = _element(attributes={'name': '04fullname'})
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[
					{'scroll': {'down': True}},
					{'scroll': {'down': True}},
					{'input': {'index': 1, 'text': 'myname'}},
				],
				elements=[None, None, el],
				results=[
					ActionResult(long_term_memory='s1'),
					ActionResult(long_term_memory='s2'),
					ActionResult(long_term_memory='ok'),
				],
			)
		]
	)
	cache = build_action_cache(history)
	assert len(cache.skipped_actions) == 2
	assert all(s.classification == 'exploratory' for s in cache.skipped_actions)
	assert [a.status for a in cache.actions] == ['kept']
	report = classification_report_from_cache(cache)
	assert report.exploratory_actions >= 2
	assert report.retained_actions == 1


def test_2_duplicate_click_must_not_remove_when_page_changes(AgentOutput):
	btn = _element(
		node_name='BUTTON',
		element_hash=9,
		stable_hash=9,
		xpath='html/body/button',
		attributes={'id': 'next'},
	)
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[btn],
				results=[ActionResult(long_term_memory='ok')],
				url='https://example.com/page/1',
				step_number=1,
			),
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[btn],
				results=[ActionResult(long_term_memory='ok')],
				url='https://example.com/page/2',
				step_number=2,
			),
			_history_item(
				AgentOutput,
				actions=[{'done': {'text': 'done', 'success': True}}],
				elements=[None],
				results=[ActionResult(is_done=True, success=True, long_term_memory='done')],
				url='https://example.com/page/3',
				step_number=3,
			),
		]
	)
	cache = build_action_cache(history)
	clicks = [a for a in cache.actions if a.action_type == 'click']
	assert len(clicks) == 2
	assert all(a.status == 'kept' for a in clicks)
	assert all(a.classification == 'required' for a in clicks)


def test_3_required_navigation_click_retained(AgentOutput):
	link = _element(
		node_name='A',
		element_hash=3,
		stable_hash=3,
		xpath='html/body/a',
		attributes={'href': '/products'},
		ax_name='Products',
	)
	item = _element(
		node_name='A',
		element_hash=4,
		stable_hash=4,
		xpath='html/body/div/a',
		attributes={'href': '/iphone'},
		ax_name='iPhone',
	)
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[link],
				results=[ActionResult(long_term_memory='ok')],
				url='https://shop.example/home',
				step_number=1,
			),
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 2}}],
				elements=[item],
				results=[ActionResult(long_term_memory='ok')],
				url='https://shop.example/products',
				step_number=2,
			),
		]
	)
	cache = build_action_cache(history)
	assert [a.status for a in cache.actions] == ['kept', 'kept']
	assert cache.actions[0].classification == 'required'
	assert 'URL' in ' '.join(cache.actions[0].classification_reasons) or 'page' in ' '.join(
		cache.actions[0].classification_reasons
	).lower() or cache.actions[0].next_page_url == 'https://shop.example/products'


def test_4_required_menu_opening_click_retained(AgentOutput):
	menu = _element(node_name='BUTTON', element_hash=1, stable_hash=1, xpath='html/body/button[1]', attributes={'id': 'menu'})
	settings = _element(node_name='A', element_hash=2, stable_hash=2, xpath='html/body/a', attributes={'id': 'settings'})
	# Same page URL (menu opens in-place) — must KEEP menu click because no coalescing input.
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[menu],
				results=[ActionResult(long_term_memory='ok')],
				url='https://example.com/app',
				step_number=1,
			),
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 2}}],
				elements=[settings],
				results=[ActionResult(long_term_memory='ok')],
				url='https://example.com/app',
				step_number=2,
			),
		]
	)
	cache = build_action_cache(history)
	assert [a.status for a in cache.actions] == ['kept', 'kept']
	assert cache.actions[0].classification == 'required'
	assert any(
		'enables subsequent' in r or 'enable' in r.lower()
		for r in cache.actions[0].classification_reasons
	)


def test_5_exploratory_focus_click_removed_when_input_captures_target(AgentOutput):
	el = _element(element_hash=5, stable_hash=5, xpath='html/body/input', attributes={'name': '04fullname'})
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}, {'input': {'index': 1, 'text': 'myname'}}],
				elements=[el, el],
				results=[ActionResult(long_term_memory='click'), ActionResult(long_term_memory='ok')],
			)
		]
	)
	cache = build_action_cache(history)
	assert cache.actions[0].action_type == 'click'
	assert cache.actions[0].status == 'dropped_redundant'
	assert cache.actions[0].classification == 'exploratory'
	assert cache.actions[1].status == 'kept'
	assert cache.actions[1].classification == 'required'


def test_6_retry_not_incorrectly_classified_as_redundant(AgentOutput):
	el = _element(node_name='BUTTON', element_hash=7, stable_hash=7, xpath='html/body/button', attributes={'id': 'login'})
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[el],
				results=[ActionResult(error='timeout', long_term_memory=None)],
				step_number=1,
			),
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[el],
				results=[ActionResult(long_term_memory='ok')],
				step_number=2,
			),
		]
	)
	cache = build_action_cache(history)
	assert cache.actions[0].status == 'dropped_failed'
	assert cache.actions[0].classification == 'retry'
	assert 'failure recovery' in ' '.join(cache.actions[0].classification_reasons) or 'failed' in ' '.join(
		cache.actions[0].classification_reasons
	).lower()
	assert cache.actions[1].status == 'kept'
	assert cache.actions[1].classification == 'required'


def test_7_noop_duplicate_click_removed_when_safe(AgentOutput):
	el = _element(node_name='BUTTON', element_hash=8, stable_hash=8, xpath='html/body/button', attributes={'id': 'x'})
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[el],
				results=[ActionResult(long_term_memory='ok')],
				url='https://example.com/same',
				step_number=1,
			),
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[el],
				results=[ActionResult(long_term_memory='ok')],
				url='https://example.com/same',
				step_number=2,
			),
			_history_item(
				AgentOutput,
				actions=[{'done': {'text': 'done', 'success': True}}],
				elements=[None],
				results=[ActionResult(is_done=True, success=True, long_term_memory='done')],
				url='https://example.com/same',
				step_number=3,
			),
		]
	)
	cache = build_action_cache(history)
	clicks = [a for a in cache.actions if a.action_type == 'click']
	assert clicks[0].status == 'kept'
	assert clicks[1].status == 'dropped_redundant'
	assert clicks[1].classification == 'redundant'


def test_8_uncertain_action_retained_when_no_locator(AgentOutput):
	# DIV with no attributes / empty xpath → derivation fails → cache_miss / uncertain
	el = _element(node_name='DIV', attributes={}, xpath='', element_hash=99, stable_hash=99)
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[{'click': {'index': 1}}],
				elements=[el],
				results=[ActionResult(long_term_memory='ok')],
			)
		]
	)
	cache = build_action_cache(history)
	assert len(cache.actions) == 1
	# May be cache_miss if no locator; classification uncertain; not dropped_redundant
	assert cache.actions[0].status in ('kept', 'cache_miss')
	assert cache.actions[0].classification in ('required', 'uncertain')
	assert cache.actions[0].status != 'dropped_redundant'


def test_9_dependency_prevents_removal_of_nav_before_target(AgentOutput):
	# Same as test_3 — Products then iPhone
	test_3_required_navigation_click_retained(AgentOutput)


def test_10_deterministic_order_preserved_after_filtering(AgentOutput):
	a = _element(element_hash=1, stable_hash=1, xpath='html/body/input[1]', attributes={'name': 'a'})
	b = _element(element_hash=2, stable_hash=2, xpath='html/body/input[2]', attributes={'name': 'b'})
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[
					{'input': {'index': 1, 'text': 'myname'}},
					{'input': {'index': 1, 'text': 'myname'}},
					{'input': {'index': 2, 'text': 'xyz'}},
				],
				elements=[a, a, b],
				results=[
					ActionResult(long_term_memory='ok'),
					ActionResult(long_term_memory='retry'),
					ActionResult(long_term_memory='ok'),
				],
			)
		]
	)
	cache = build_action_cache(history)
	kept = [x for x in cache.actions if x.status == 'kept']
	assert [x.value for x in kept] == ['myname', 'xyz']
	assert kept[0].sequence < kept[1].sequence


def test_11_removal_validation_restores_required_candidate():
	req = CacheAction(
		sequence=1,
		action_type='click',
		value=None,
		target=CacheTarget(node_name='A', attributes={}, xpath='a', element_hash=1, stable_hash=1),
		result=CacheActionResult(success=True),
		status='kept',
		source=CacheActionSource(history_step=0, action_index=0),
		classification='required',
		classification_reasons=['nav'],
		step_id='step_required',
	)
	cand = CacheAction(
		sequence=2,
		action_type='click',
		value=None,
		target=CacheTarget(node_name='BUTTON', attributes={}, xpath='b', element_hash=2, stable_hash=2),
		result=CacheActionResult(success=True),
		status='dropped_redundant',
		source=CacheActionSource(history_step=1, action_index=0),
		classification='redundant',
		classification_reasons=['looked duplicate'],
		step_id='step_needed',
	)
	entries = [req, cand]

	def probe(actions: list[CacheAction]) -> bool:
		ids = {a.step_id for a in actions}
		return 'step_required' in ids and 'step_needed' in ids

	restored = validate_removal_candidates(entries, probe=probe, max_candidates=3)
	assert cand in restored
	assert cand.status == 'kept'
	assert cand.classification == 'required'


def test_12_removal_validation_keeps_redundant_when_probe_ok():
	req = CacheAction(
		sequence=1,
		action_type='input',
		value='x',
		target=CacheTarget(node_name='INPUT', attributes={'name': 'a'}, xpath='i', element_hash=1, stable_hash=1),
		result=CacheActionResult(success=True),
		status='kept',
		source=CacheActionSource(history_step=0, action_index=0),
		classification='required',
		classification_reasons=['fill'],
		step_id='step_fill',
	)
	cand = CacheAction(
		sequence=2,
		action_type='input',
		value='x',
		target=CacheTarget(node_name='INPUT', attributes={'name': 'a'}, xpath='i', element_hash=1, stable_hash=1),
		result=CacheActionResult(success=True),
		status='dropped_redundant',
		source=CacheActionSource(history_step=1, action_index=0),
		classification='redundant',
		classification_reasons=['duplicate'],
		step_id='step_dup',
	)

	def probe(actions: list[CacheAction]) -> bool:
		return any(a.step_id == 'step_fill' for a in actions)

	restored = validate_removal_candidates([req, cand], probe=probe)
	assert restored == []
	assert cand.status == 'dropped_redundant'
	assert any('validation succeeded' in r for r in cand.classification_reasons)


def test_13_multiple_redundant_without_reordering_required(AgentOutput):
	a = _element(element_hash=1, stable_hash=1, xpath='html/body/input[1]', attributes={'name': 'a'})
	b = _element(element_hash=2, stable_hash=2, xpath='html/body/input[2]', attributes={'name': 'b'})
	c = _element(element_hash=3, stable_hash=3, xpath='html/body/input[3]', attributes={'name': 'c'})
	history = AgentHistoryList(
		history=[
			_history_item(
				AgentOutput,
				actions=[
					{'input': {'index': 1, 'text': 'myname'}},
					{'input': {'index': 1, 'text': 'myname'}},
					{'input': {'index': 2, 'text': 'xyz'}},
					{'input': {'index': 2, 'text': 'xyz'}},
					{'input': {'index': 3, 'text': 'abc'}},
				],
				elements=[a, a, b, b, c],
				results=[ActionResult(long_term_memory='ok')] * 5,
			)
		]
	)
	cache = build_action_cache(history)
	kept = [x for x in cache.actions if x.status == 'kept']
	assert [x.value for x in kept] == ['myname', 'xyz', 'abc']
	assert [x.sequence for x in kept] == sorted(x.sequence for x in kept)
	report = classification_report_from_cache(cache)
	assert report.redundant_actions >= 2
	assert report.retained_actions == 3
