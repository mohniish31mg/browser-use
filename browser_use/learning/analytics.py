"""Lightweight analytics for learned automations (Phase 4).

Designed for JSON/CLI reports now; fields are dashboard-ready later.

Rules:
- Use actual measured wall-clock / step counts only.
- Never fabricate token savings. If tokens are unmeasurable, say so explicitly.
- Keep latency metrics separate from token metrics.
- Timestamps are timezone-aware ISO-8601 (UTC).
- No credentials / page HTML / cookies stored.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from browser_use.learning.models import ActionCache, CacheHealthState, utc_now_iso


class ExecutionMode(str, Enum):
	AGENTIC = 'agentic'
	CACHED = 'cached'


class ExecutionRecord(BaseModel):
	"""One measured run of a workflow (agentic or cached)."""

	workflow_id: str
	mode: ExecutionMode
	success: bool
	wall_clock_seconds: float
	recorded_at: str = Field(default_factory=utc_now_iso)
	agent_steps: int | None = None
	llm_calls_approx: int | None = None
	# None means measurement unavailable — never treat missing as zero savings.
	total_tokens: int | None = None
	tokens_measurable: bool = False
	cache_version: int | None = None
	notes: str | None = None


class LearningBreakdown(BaseModel):
	"""Human-readable learning funnel for interview / reports."""

	observed_replayable_actions: int = 0
	deterministic_actions_retained: int = 0
	redundant_actions_removed: int = 0
	failed_actions_dropped: int = 0
	superseded_actions_dropped: int = 0
	cache_misses_unresolved: int = 0
	# Kept actions that conversion may still leave agentic / unsupported.
	unresolved_or_non_converted: int = 0
	deterministic_action_types: dict[str, int] = Field(default_factory=dict)
	locator_strategies: dict[str, int] = Field(default_factory=dict)


class WorkflowAnalytics(BaseModel):
	"""Per-workflow analytics store (persistable JSON)."""

	workflow_id: str
	created_at: str = Field(default_factory=utc_now_iso)
	updated_at: str = Field(default_factory=utc_now_iso)
	records: list[ExecutionRecord] = Field(default_factory=list)
	# Explicit baseline (first successful agentic measurement), if known.
	baseline_agentic_seconds: float | None = None
	baseline_agentic_steps: int | None = None
	cache_repairs: int = 0
	cache_failures: int = 0
	cache_version: int | None = None
	cache_health: str | None = None
	cache_confidence: float | None = None
	learning: LearningBreakdown | None = None


class LatencyMetrics(BaseModel):
	original_agentic_seconds: float | None = None
	current_cached_seconds: float | None = None
	average_agentic_seconds: float | None = None
	average_cached_seconds: float | None = None
	speed_improvement_pct: float | None = None
	latency_saved_per_cached_run_seconds: float | None = None
	time_saved_today_seconds: float | None = None
	total_time_saved_seconds: float | None = None


class TokenMetrics(BaseModel):
	measurable: bool = False
	message: str = 'LLM token measurement unavailable'
	total_tokens_observed: int | None = None
	estimated_tokens_saved: int | None = None


class WorkflowPerformanceReport(BaseModel):
	"""Aggregated dashboard-ready report for one workflow."""

	workflow_id: str
	generated_at: str = Field(default_factory=utc_now_iso)
	total_executions: int = 0
	executions_today: int = 0
	successful_executions: int = 0
	failed_executions: int = 0
	cached_executions: int = 0
	agentic_executions: int = 0
	cache_hit_rate: float | None = None
	latency: LatencyMetrics = Field(default_factory=LatencyMetrics)
	tokens: TokenMetrics = Field(default_factory=TokenMetrics)
	deterministic_actions_count: int = 0
	redundant_actions_removed: int = 0
	llm_steps_avoided: int | None = None
	cache_confidence: float | None = None
	cache_repairs: int = 0
	cache_failures: int = 0
	current_cache_version: int | None = None
	cache_health: str | None = None
	learning: LearningBreakdown | None = None
	# First-run / empty baselines
	has_agentic_baseline: bool = False
	has_cached_runs: bool = False


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------


def _parse_ts(value: str) -> datetime:
	v = value.replace('Z', '+00:00')
	dt = datetime.fromisoformat(v)
	if dt.tzinfo is None:
		dt = dt.replace(tzinfo=timezone.utc)
	return dt.astimezone(timezone.utc)


def _today_utc(now: datetime | None = None) -> date:
	now = now or datetime.now(timezone.utc)
	return now.astimezone(timezone.utc).date()


def new_workflow_analytics(workflow_id: str) -> WorkflowAnalytics:
	return WorkflowAnalytics(workflow_id=workflow_id)


def record_execution(store: WorkflowAnalytics, record: ExecutionRecord) -> WorkflowAnalytics:
	"""Append a measured execution and update baselines when appropriate."""
	if record.workflow_id != store.workflow_id:
		raise ValueError(
			f'record workflow_id={record.workflow_id!r} does not match store {store.workflow_id!r}'
		)
	records = list(store.records) + [record]
	baseline_s = store.baseline_agentic_seconds
	baseline_steps = store.baseline_agentic_steps
	if (
		record.mode == ExecutionMode.AGENTIC
		and record.success
		and baseline_s is None
		and record.wall_clock_seconds >= 0
	):
		baseline_s = record.wall_clock_seconds
		baseline_steps = record.agent_steps
	return store.model_copy(
		update={
			'records': records,
			'baseline_agentic_seconds': baseline_s,
			'baseline_agentic_steps': baseline_steps,
			'updated_at': utc_now_iso(),
		}
	)


def learning_breakdown_from_cache(cache: ActionCache) -> LearningBreakdown:
	"""Derive interview-friendly learning funnel from an ActionCache."""
	observed = len(cache.actions)
	by_status: dict[str, int] = {}
	types: dict[str, int] = {}
	strategies: dict[str, int] = {}
	for action in cache.actions:
		by_status[action.status] = by_status.get(action.status, 0) + 1
		if action.status == 'kept':
			types[action.action_type] = types.get(action.action_type, 0) + 1
			if action.locator and action.locator.strategy:
				strategies[action.locator.strategy] = strategies.get(action.locator.strategy, 0) + 1

	kept = by_status.get('kept', 0)
	miss = by_status.get('cache_miss', 0)
	return LearningBreakdown(
		observed_replayable_actions=observed,
		deterministic_actions_retained=kept,
		redundant_actions_removed=by_status.get('dropped_redundant', 0),
		failed_actions_dropped=by_status.get('dropped_failed', 0),
		superseded_actions_dropped=by_status.get('dropped_superseded', 0),
		cache_misses_unresolved=miss,
		unresolved_or_non_converted=miss,
		deterministic_action_types=types,
		locator_strategies=strategies,
	)


def apply_cache_snapshot(store: WorkflowAnalytics, cache: ActionCache) -> WorkflowAnalytics:
	"""Attach current cache health/version/confidence + learning breakdown."""
	learning = learning_breakdown_from_cache(cache)
	confidences = [
		a.confidence for a in cache.actions if a.status == 'kept' and a.confidence is not None
	]
	confidence = (sum(confidences) / len(confidences)) if confidences else None
	health = cache.health.state.value if cache.health else CacheHealthState.HEALTHY.value
	return store.model_copy(
		update={
			'learning': learning,
			'cache_version': cache.version,
			'cache_health': health,
			'cache_confidence': confidence,
			'updated_at': utc_now_iso(),
		}
	)


def note_cache_repair(store: WorkflowAnalytics, *, n: int = 1) -> WorkflowAnalytics:
	return store.model_copy(
		update={'cache_repairs': store.cache_repairs + n, 'updated_at': utc_now_iso()}
	)


def note_cache_failure(store: WorkflowAnalytics, *, n: int = 1) -> WorkflowAnalytics:
	return store.model_copy(
		update={'cache_failures': store.cache_failures + n, 'updated_at': utc_now_iso()}
	)


# ---------------------------------------------------------------------------
# Aggregation (pure / testable)
# ---------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
	if not values:
		return None
	return sum(values) / len(values)


def _safe_pct(numerator: float, denominator: float) -> float | None:
	if denominator == 0:
		return None
	return (numerator / denominator) * 100.0


def _safe_rate(numerator: float, denominator: float) -> float | None:
	if denominator == 0:
		return None
	return numerator / denominator


def aggregate_workflow_report(
	store: WorkflowAnalytics,
	*,
	now: datetime | None = None,
) -> WorkflowPerformanceReport:
	"""Compute dashboard metrics from recorded executions + cache snapshot."""
	now = now or datetime.now(timezone.utc)
	today = _today_utc(now)

	records = store.records
	total = len(records)
	today_records = [r for r in records if _parse_ts(r.recorded_at).date() == today]
	successes = [r for r in records if r.success]
	failures = [r for r in records if not r.success]
	cached = [r for r in records if r.mode == ExecutionMode.CACHED]
	agentic = [r for r in records if r.mode == ExecutionMode.AGENTIC]
	cached_ok = [r for r in cached if r.success]
	cached_fail = [r for r in cached if not r.success]
	agentic_ok = [r for r in agentic if r.success]

	avg_agentic = _mean([r.wall_clock_seconds for r in agentic_ok])
	avg_cached = _mean([r.wall_clock_seconds for r in cached_ok])
	baseline = store.baseline_agentic_seconds
	original = baseline if baseline is not None else avg_agentic
	# "Current" = most recent successful cached run, else average.
	current_cached = cached_ok[-1].wall_clock_seconds if cached_ok else avg_cached

	latency_saved = None
	if original is not None and avg_cached is not None:
		latency_saved = max(0.0, original - avg_cached)

	speed_pct = None
	if original is not None and current_cached is not None and original > 0:
		speed_pct = _safe_pct(original - current_cached, original)

	# Time saved: for each successful cached run, (baseline_or_avg_agentic - run_duration).
	ref = original
	total_saved = None
	today_saved = None
	if ref is not None and cached_ok:
		saves = [max(0.0, ref - r.wall_clock_seconds) for r in cached_ok]
		total_saved = sum(saves)
		today_cached_ok = [
			r for r in cached_ok if _parse_ts(r.recorded_at).date() == today
		]
		today_saved = sum(max(0.0, ref - r.wall_clock_seconds) for r in today_cached_ok)

	# Cache hit rate: successful cached / all cached attempts (None if no cached attempts).
	cache_hit_rate = _safe_rate(len(cached_ok), len(cached))

	# LLM steps avoided: cached successes × baseline/mean agentic steps.
	steps_ref = store.baseline_agentic_steps
	if steps_ref is None and agentic_ok:
		step_vals = [r.agent_steps for r in agentic_ok if r.agent_steps is not None]
		steps_ref = int(round(sum(step_vals) / len(step_vals))) if step_vals else None
	llm_steps_avoided = None
	if steps_ref is not None and cached_ok:
		llm_steps_avoided = steps_ref * len(cached_ok)

	# Tokens: only if at least one record marked measurable with a non-None total.
	measurable_records = [r for r in records if r.tokens_measurable and r.total_tokens is not None]
	if measurable_records:
		token_metrics = TokenMetrics(
			measurable=True,
			message='Token totals are observed measurements (not fabricated).',
			total_tokens_observed=sum(r.total_tokens or 0 for r in measurable_records),
			estimated_tokens_saved=None,  # do not invent savings without a real baseline delta
		)
	else:
		token_metrics = TokenMetrics(
			measurable=False,
			message='LLM token measurement unavailable',
			total_tokens_observed=None,
			estimated_tokens_saved=None,
		)

	learning = store.learning
	det_count = learning.deterministic_actions_retained if learning else 0
	redundant = learning.redundant_actions_removed if learning else 0

	return WorkflowPerformanceReport(
		workflow_id=store.workflow_id,
		generated_at=now.isoformat(),
		total_executions=total,
		executions_today=len(today_records),
		successful_executions=len(successes),
		failed_executions=len(failures),
		cached_executions=len(cached),
		agentic_executions=len(agentic),
		cache_hit_rate=cache_hit_rate,
		latency=LatencyMetrics(
			original_agentic_seconds=original,
			current_cached_seconds=current_cached,
			average_agentic_seconds=avg_agentic,
			average_cached_seconds=avg_cached,
			speed_improvement_pct=speed_pct,
			latency_saved_per_cached_run_seconds=latency_saved,
			time_saved_today_seconds=today_saved,
			total_time_saved_seconds=total_saved,
		),
		tokens=token_metrics,
		deterministic_actions_count=det_count,
		redundant_actions_removed=redundant,
		llm_steps_avoided=llm_steps_avoided,
		cache_confidence=store.cache_confidence,
		cache_repairs=store.cache_repairs,
		cache_failures=store.cache_failures,
		current_cache_version=store.cache_version,
		cache_health=store.cache_health,
		learning=learning,
		has_agentic_baseline=original is not None,
		has_cached_runs=bool(cached),
	)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_duration(seconds: float | None) -> str:
	if seconds is None:
		return 'n/a'
	if seconds < 0:
		seconds = 0.0
	if seconds < 60:
		return f'{seconds:.1f} sec'
	total = int(round(seconds))
	hours, rem = divmod(total, 3600)
	minutes, secs = divmod(rem, 60)
	if hours:
		return f'{hours}h {minutes:02d}m'
	return f'{minutes}m {secs:02d}s'


def format_pct(value: float | None, *, digits: int = 1) -> str:
	if value is None:
		return 'n/a'
	return f'{value:.{digits}f}%'


def format_rate_as_pct(rate: float | None, *, digits: int = 1) -> str:
	if rate is None:
		return 'n/a'
	return f'{rate * 100:.{digits}f}%'


def format_performance_report(report: WorkflowPerformanceReport) -> str:
	"""Human-readable Automation Performance block (interview-friendly)."""
	lat = report.latency
	lines = [
		'Automation Performance',
		'',
		f'Original agentic run:     {format_duration(lat.original_agentic_seconds):>10}',
		f'Current cached run:       {format_duration(lat.current_cached_seconds):>10}',
		f'Speed improvement:        {format_pct(lat.speed_improvement_pct):>10}',
		'',
		f'Executions today:         {report.executions_today:>10,}',
		f'Total executions:         {report.total_executions:>10,}',
		'',
		f'Time saved today:         {format_duration(lat.time_saved_today_seconds):>10}',
		f'Total time saved:         {format_duration(lat.total_time_saved_seconds):>10}',
		'',
		f'LLM steps avoided:        {report.llm_steps_avoided if report.llm_steps_avoided is not None else "n/a":>10}',
		f'Cached actions:           {report.deterministic_actions_count:>10,}',
		f'Redundant actions removed:{report.redundant_actions_removed:>10,}',
		'',
		f'Cache hit rate:           {format_rate_as_pct(report.cache_hit_rate):>10}',
		f'Cache confidence:         {format_pct(report.cache_confidence * 100 if report.cache_confidence is not None else None):>10}',
		'',
		f'Cache repairs:            {report.cache_repairs:>10,}',
		f'Current cache version:    {report.current_cache_version if report.current_cache_version is not None else "n/a":>10}',
		f'Status:                   {(report.cache_health or "unknown").upper():>10}',
		'',
		'--- Latency vs tokens ---',
		f'Latency saved / cached run: {format_duration(lat.latency_saved_per_cached_run_seconds)}',
		f'Token measurement:          {report.tokens.message}',
	]
	if not report.has_agentic_baseline:
		lines.append('')
		lines.append('Note: no agentic baseline yet (first-run / learning phase).')
	if not report.has_cached_runs:
		lines.append('Note: no cached executions recorded yet.')
	return '\n'.join(lines)


def format_learning_report(learning: LearningBreakdown, *, workflow_id: str | None = None) -> str:
	"""Explain Observed → retained / removed → final cached automation."""
	header = 'Learning Report'
	if workflow_id:
		header = f'Learning Report ({workflow_id})'
	lines = [
		header,
		'',
		'Observed replayable actions',
		f'  -> {learning.observed_replayable_actions}',
		'Deterministic actions retained (kept)',
		f'  -> {learning.deterministic_actions_retained}'
		+ (
			f'  types={dict(learning.deterministic_action_types)}'
			if learning.deterministic_action_types
			else ''
		),
		'Redundant actions removed',
		f'  -> {learning.redundant_actions_removed}',
		'Failed actions dropped',
		f'  -> {learning.failed_actions_dropped}',
		'Superseded corrections dropped',
		f'  -> {learning.superseded_actions_dropped}',
		'Unresolved / cache_miss (no invented locator)',
		f'  -> {learning.cache_misses_unresolved}',
		'Final cached automation',
		f'  -> {learning.deterministic_actions_retained} deterministic Playwright steps'
		+ (
			f'  strategies={dict(learning.locator_strategies)}'
			if learning.locator_strategies
			else ''
		),
	]
	return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_analytics(store: WorkflowAnalytics, path: str | Path) -> Path:
	out = Path(path)
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(json.dumps(store.model_dump(mode='json'), indent=2), encoding='utf-8')
	return out


def load_analytics(path: str | Path) -> WorkflowAnalytics:
	data = json.loads(Path(path).read_text(encoding='utf-8'))
	return WorkflowAnalytics.model_validate(data)


def ingest_benchmark_summary(
	store: WorkflowAnalytics,
	*,
	agentic_seconds: list[float],
	cached_seconds: list[float],
	agentic_successes: list[bool] | None = None,
	cached_successes: list[bool] | None = None,
	agentic_steps: list[int] | None = None,
	recorded_at: str | None = None,
	tokens_measurable: bool = False,
) -> WorkflowAnalytics:
	"""Helper to turn measured trial lists into ExecutionRecords (no fabrication)."""
	ts = recorded_at or utc_now_iso()
	out = store
	for i, sec in enumerate(agentic_seconds):
		ok = True if agentic_successes is None else bool(agentic_successes[i])
		steps = agentic_steps[i] if agentic_steps and i < len(agentic_steps) else None
		out = record_execution(
			out,
			ExecutionRecord(
				workflow_id=store.workflow_id,
				mode=ExecutionMode.AGENTIC,
				success=ok,
				wall_clock_seconds=sec,
				agent_steps=steps,
				llm_calls_approx=steps,
				tokens_measurable=tokens_measurable,
				total_tokens=None if not tokens_measurable else 0,
				recorded_at=ts,
			),
		)
	for i, sec in enumerate(cached_seconds):
		ok = True if cached_successes is None else bool(cached_successes[i])
		out = record_execution(
			out,
			ExecutionRecord(
				workflow_id=store.workflow_id,
				mode=ExecutionMode.CACHED,
				success=ok,
				wall_clock_seconds=sec,
				agent_steps=0,
				llm_calls_approx=0,
				tokens_measurable=tokens_measurable,
				total_tokens=None if not tokens_measurable else 0,
				recorded_at=ts,
			),
		)
	return out
