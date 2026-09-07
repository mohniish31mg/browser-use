"""Unit tests for analytics aggregation and learning reports."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from browser_use.learning.analytics import (
	ExecutionMode,
	ExecutionRecord,
	aggregate_workflow_report,
	apply_cache_snapshot,
	format_duration,
	format_learning_report,
	format_performance_report,
	ingest_benchmark_summary,
	learning_breakdown_from_cache,
	new_workflow_analytics,
	note_cache_failure,
	note_cache_repair,
	record_execution,
)
from browser_use.learning.models import (
	ActionCache,
	CacheAction,
	CacheActionResult,
	CacheActionSource,
	CacheLocator,
	CacheTarget,
)


def _action(status: str, *, conf: float | None = 0.9, strategy: str = 'id') -> CacheAction:
	return CacheAction(
		sequence=1,
		action_type='input',
		value='x',
		target=CacheTarget(node_name='INPUT', attributes={'id': 'a'}, xpath='x', element_hash=1),
		result=CacheActionResult(success=True),
		status=status,  # type: ignore[arg-type]
		source=CacheActionSource(history_step=0, action_index=0),
		locator=CacheLocator(strategy=strategy, command='locator("#a")', evidence={})
		if status == 'kept'
		else None,
		confidence=conf if status == 'kept' else None,
	)


def test_first_run_no_baseline():
	store = new_workflow_analytics('wf_empty')
	report = aggregate_workflow_report(store)
	assert report.total_executions == 0
	assert report.has_agentic_baseline is False
	assert report.has_cached_runs is False
	assert report.latency.original_agentic_seconds is None
	assert report.latency.speed_improvement_pct is None
	assert report.latency.total_time_saved_seconds is None
	assert report.cache_hit_rate is None
	assert report.llm_steps_avoided is None
	assert report.tokens.measurable is False
	assert 'unavailable' in report.tokens.message.lower()


def test_division_by_zero_safe():
	store = new_workflow_analytics('wf_agentic_only')
	store = record_execution(
		store,
		ExecutionRecord(
			workflow_id='wf_agentic_only',
			mode=ExecutionMode.AGENTIC,
			success=True,
			wall_clock_seconds=10.0,
			agent_steps=5,
			recorded_at='2026-01-01T12:00:00+00:00',
		),
	)
	report = aggregate_workflow_report(
		store, now=datetime(2026, 1, 1, 15, tzinfo=timezone.utc)
	)
	assert report.cache_hit_rate is None  # no cached attempts
	assert report.latency.speed_improvement_pct is None
	assert report.latency.total_time_saved_seconds is None


def test_latency_aggregation_and_time_saved():
	store = new_workflow_analytics('wf_lat')
	store = ingest_benchmark_summary(
		store,
		agentic_seconds=[14.2, 15.0],
		cached_seconds=[2.1, 2.0, 2.2],
		agentic_successes=[True, True],
		cached_successes=[True, True, True],
		agentic_steps=[8, 8],
		recorded_at='2026-01-02T10:00:00+00:00',
	)
	report = aggregate_workflow_report(
		store, now=datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
	)
	assert report.total_executions == 5
	assert report.executions_today == 5
	assert report.agentic_executions == 2
	assert report.cached_executions == 3
	assert report.successful_executions == 5
	assert store.baseline_agentic_seconds == pytest.approx(14.2)
	assert report.latency.original_agentic_seconds == pytest.approx(14.2)
	assert report.latency.current_cached_seconds == pytest.approx(2.2)  # latest cached ok
	assert report.latency.average_cached_seconds == pytest.approx(2.1)
	# speed vs current: (14.2 - 2.2) / 14.2
	assert report.latency.speed_improvement_pct == pytest.approx((14.2 - 2.2) / 14.2 * 100)
	assert report.latency.latency_saved_per_cached_run_seconds == pytest.approx(14.2 - 2.1)
	expected_saved = (14.2 - 2.1) + (14.2 - 2.0) + (14.2 - 2.2)
	assert report.latency.total_time_saved_seconds == pytest.approx(expected_saved)
	assert report.latency.time_saved_today_seconds == pytest.approx(expected_saved)
	assert report.cache_hit_rate == pytest.approx(1.0)
	assert report.llm_steps_avoided == 8 * 3


def test_failed_cached_lowers_hit_rate():
	store = new_workflow_analytics('wf_hit')
	store = ingest_benchmark_summary(
		store,
		agentic_seconds=[10.0],
		cached_seconds=[2.0, 2.0],
		agentic_successes=[True],
		cached_successes=[True, False],
		agentic_steps=[4],
		recorded_at='2026-01-03T00:00:00+00:00',
	)
	report = aggregate_workflow_report(store)
	assert report.cache_hit_rate == pytest.approx(0.5)
	assert report.failed_executions == 1
	assert report.llm_steps_avoided == 4  # only successful cached


def test_tokens_unavailable_not_fabricated_as_zero_savings():
	store = new_workflow_analytics('wf_tok')
	store = ingest_benchmark_summary(
		store,
		agentic_seconds=[10.0],
		cached_seconds=[2.0],
		tokens_measurable=False,
		recorded_at='2026-01-04T00:00:00+00:00',
	)
	report = aggregate_workflow_report(store)
	assert report.tokens.measurable is False
	assert report.tokens.estimated_tokens_saved is None
	assert 'unavailable' in report.tokens.message.lower()
	text = format_performance_report(report)
	assert 'LLM token measurement unavailable' in text
	# Latency section still present and separate
	assert 'Latency saved' in text or 'Original agentic' in text


def test_learning_breakdown_and_report_text():
	cache = ActionCache(
		version=4,
		workflow_id='wf_learn',
		source={'url': 'https://example.com'},
		actions=[
			_action('kept', conf=0.9, strategy='id'),
			_action('kept', conf=0.8, strategy='name'),
			_action('dropped_redundant'),
			_action('dropped_redundant'),
			_action('dropped_failed'),
			_action('cache_miss'),
		],
	)
	# fix sequences uniqueness not required
	breakdown = learning_breakdown_from_cache(cache)
	assert breakdown.observed_replayable_actions == 6
	assert breakdown.deterministic_actions_retained == 2
	assert breakdown.redundant_actions_removed == 2
	assert breakdown.failed_actions_dropped == 1
	assert breakdown.cache_misses_unresolved == 1

	store = apply_cache_snapshot(new_workflow_analytics('wf_learn'), cache)
	store = note_cache_repair(store, n=3)
	store = note_cache_failure(store, n=1)
	assert store.cache_version == 4
	assert store.cache_confidence == pytest.approx(0.85)
	assert store.cache_repairs == 3

	text = format_learning_report(breakdown, workflow_id='wf_learn')
	assert 'Observed replayable actions' in text
	assert 'Deterministic actions retained' in text
	assert 'Redundant actions removed' in text
	assert '-> 2' in text or '→ 2' in text or 'retained (kept)' in text
	assert 'Final cached automation' in text
	assert str(breakdown.deterministic_actions_retained) in text


def test_executions_today_timezone_aware():
	store = new_workflow_analytics('wf_day')
	store = record_execution(
		store,
		ExecutionRecord(
			workflow_id='wf_day',
			mode=ExecutionMode.CACHED,
			success=True,
			wall_clock_seconds=1.0,
			recorded_at='2026-05-01T23:30:00+00:00',
		),
	)
	store = record_execution(
		store,
		ExecutionRecord(
			workflow_id='wf_day',
			mode=ExecutionMode.CACHED,
			success=True,
			wall_clock_seconds=1.0,
			recorded_at='2026-05-02T01:00:00+00:00',
		),
	)
	report = aggregate_workflow_report(
		store, now=datetime(2026, 5, 2, 12, tzinfo=timezone.utc)
	)
	assert report.executions_today == 1
	assert report.total_executions == 2


def test_format_duration_helpers():
	assert format_duration(None) == 'n/a'
	assert format_duration(14.2) == '14.2 sec'
	assert 'm' in format_duration(1546)  # ~25m
	assert 'h' in format_duration(31 * 3600 + 18 * 60)


def test_mismatched_workflow_id_raises():
	store = new_workflow_analytics('wf_a')
	with pytest.raises(ValueError):
		record_execution(
			store,
			ExecutionRecord(
				workflow_id='wf_b',
				mode=ExecutionMode.AGENTIC,
				success=True,
				wall_clock_seconds=1.0,
			),
		)
