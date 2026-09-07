"""Pydantic models for the action cache (CachedAction[] / ActionCache).

Mental checklist for each cached action (not every field is required forever —
the repo schema is the source of truth):

    ACTION
    ├── What?          action_type   (input / click / select / …)
    ├── What value?    value         (e.g. "myname")
    ├── Where?         target        (CacheTarget: tag, attrs, hashes, xpath)
    ├── How identify?  locator.strategy + evidence  (role / label / id / …)
    ├── What locator?  locator.command             (Playwright, no page. prefix)
    ├── On which page? page_url                    (from history BrowserState)
    ├── Did it succeed? result.success / result.error
    ├── Where in flow? sequence + step_id
    └── Repair stats   success_count / failure_count / locator_history
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

CacheStatus = Literal[
	'kept',
	'dropped_redundant',
	'dropped_failed',
	'dropped_superseded',
	'cache_miss',
]

# Interview-facing semantic labels (mapped onto CacheStatus for replay filtering).
# Conservative rule: ambiguous evidence → 'uncertain' and KEEP (status=kept).
ActionClassification = Literal[
	'required',
	'redundant',
	'exploratory',
	'retry',
	'uncertain',
]


def utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat()


def new_step_id() -> str:
	"""Stable internal id assigned once per learned action (survives repairs)."""
	return f'step_{uuid4().hex[:12]}'


class CacheTarget(BaseModel):
	"""Stable-ish identity for the interacted DOM node (not the transient index)."""

	node_name: str
	attributes: dict[str, str] = Field(default_factory=dict)
	xpath: str
	element_hash: int
	stable_hash: int | None = None
	ax_name: str | None = None


class CacheActionResult(BaseModel):
	success: bool
	error: str | None = None


class CacheActionSource(BaseModel):
	"""Debug provenance only — never use as identity or as a locator.

	``action_index`` is the position inside one history step's action list, and
	``backend_node_id`` is a CDP node id for that run. Both are ephemeral (like
	browser-use highlight indexes): Run 1's [37] can be a different element in
	Run 2. Identity for replay is ``target`` + ``locator.command`` only.
	"""

	history_step: int
	action_index: int
	backend_node_id: int | None = None


class CacheLocator(BaseModel):
	"""Optexity-ready Playwright locator (no ``page.`` prefix; no action method suffix)."""

	strategy: str
	command: str
	evidence: dict[str, Any] = Field(default_factory=dict)


class LocatorHistoryEntry(BaseModel):
	"""Previous locator retained when a step is repaired (historical metadata)."""

	locator: CacheLocator
	replaced_at: str = Field(default_factory=utc_now_iso)
	reason: str | None = None


class RepairOutcome(str, Enum):
	"""Result of attempting to repair a single failed cache step."""

	CACHE_HIT = 'cache_hit'
	DETERMINISTIC_REPAIR = 'deterministic_repair'
	LLM_REPAIR = 'llm_repair'
	UNRECOVERABLE = 'unrecoverable'


class RepairResult(BaseModel):
	"""Structured outcome for one-step cache repair (never whole-workflow rediscovery)."""

	outcome: RepairOutcome
	step_id: str
	workflow_id: str | None = None
	previous_command: str | None = None
	new_command: str | None = None
	new_locator: CacheLocator | None = None
	error: str | None = None
	tried_alternatives: list[str] = Field(default_factory=list)
	cache_version_before: int | None = None
	cache_version_after: int | None = None
	message: str | None = None


class CacheAction(BaseModel):
	"""One cached DOM interaction in sequence order (alias: CachedAction)."""

	sequence: int
	action_type: str
	value: str | None = None
	target: CacheTarget
	result: CacheActionResult
	status: CacheStatus
	source: CacheActionSource
	locator: CacheLocator | None = None
	# Page context at the history step that produced this action (may be None).
	page_url: str | None = None
	# Stable id for single-step repair (survives locator updates).
	step_id: str = Field(default_factory=new_step_id)
	workflow_id: str | None = None
	prompt_instructions: str | None = None
	created_at: str = Field(default_factory=utc_now_iso)
	updated_at: str | None = None
	success_count: int = 0
	failure_count: int = 0
	# Higher = more preferred locator strategy when first derived (0..1).
	confidence: float | None = None
	locator_history: list[LocatorHistoryEntry] = Field(default_factory=list)
	# Redundancy / learning classification (explainability; does not invent selectors).
	classification: ActionClassification | None = None
	classification_reasons: list[str] = Field(default_factory=list)
	# URL observed on the *next* history step (state-change signal). None if last step.
	next_page_url: str | None = None


# Interview-facing alias matching the design doc naming.
CachedAction = CacheAction


class SkippedHistoryAction(BaseModel):
	"""Non-replayable history gesture (scroll/navigate/done/…) kept for explainability.

	These never become Automation nodes; they document exploration during the
	first agentic run.
	"""

	history_step: int
	action_index: int
	action_type: str
	page_url: str | None = None
	classification: ActionClassification = 'exploratory'
	classification_reasons: list[str] = Field(default_factory=list)


class ClassificationReportEntry(BaseModel):
	step: int
	step_id: str | None = None
	action_type: str | None = None
	classification: ActionClassification
	reason: list[str] = Field(default_factory=list)
	status: CacheStatus | None = None
	command: str | None = None
	value: str | None = None
	removed: bool = False


class ClassificationReport(BaseModel):
	"""Machine-readable learning report for interviews / harnesses."""

	observed_actions: int = 0
	required_actions: int = 0
	redundant_actions: int = 0
	exploratory_actions: int = 0
	uncertain_actions: int = 0
	retry_actions: int = 0
	retained_actions: int = 0
	removed_actions: int = 0
	skipped_non_dom: int = 0
	entries: list[ClassificationReportEntry] = Field(default_factory=list)
	removed_detail: list[ClassificationReportEntry] = Field(default_factory=list)
	retained_detail: list[ClassificationReportEntry] = Field(default_factory=list)


class CacheHealthState(str, Enum):
	"""Workflow/cache operational state (degraded mode state machine)."""

	HEALTHY = 'healthy'
	REPAIRING = 'repairing'
	DEGRADED = 'degraded'
	DISABLED = 'disabled'


class CacheHealthPolicy(BaseModel):
	"""Configurable circuit-breaker thresholds (env-overridable; no magic prod constants)."""

	# Unrecovered failures inside the window before marking DEGRADED.
	failures_to_degrade: int = 2
	# Max repair attempts (deterministic and/or LLM) allowed inside the window.
	max_repair_attempts_per_window: int = 2
	# Cap targeted LLM repairs inside the window (token protection).
	max_llm_repairs_per_window: int = 1
	# Sliding window for the counters above.
	failure_window_seconds: float = 3600.0
	# When DEGRADED, automatic repair is off unless explicit recovery is used.
	allow_automatic_repair_when_degraded: bool = False


class CacheHealth(BaseModel):
	state: CacheHealthState = CacheHealthState.HEALTHY
	updated_at: str = Field(default_factory=utc_now_iso)
	last_failure_at: str | None = None
	last_failure_step_id: str | None = None
	last_failure_reason: str | None = None
	# ISO timestamps used for sliding-window rate limits (not secrets).
	failure_timestamps: list[str] = Field(default_factory=list)
	repair_attempt_timestamps: list[str] = Field(default_factory=list)
	llm_repair_timestamps: list[str] = Field(default_factory=list)
	recommended_action: str | None = None
	# Set when operator requests explicit relearn/repair.
	relearn_requested: bool = False


class CacheExecutionFailure(BaseModel):
	"""Structured failure — never implies full-workflow LLM rediscovery."""

	workflow_id: str | None = None
	step_id: str | None = None
	failure_reason: str
	deterministic_repair_attempted: bool = False
	llm_repair_attempted: bool = False
	cache_state: CacheHealthState
	recommended_next_action: str
	repair_outcome: RepairOutcome | None = None
	full_agentic_fallback: bool = False
	cache_version: int | None = None
	message: str | None = None


class CacheValidationStatus(str, Enum):
	"""Result of the last lightweight warm/validation pass."""

	UNKNOWN = 'unknown'
	VALID = 'valid'
	REPAIRED = 'repaired'
	FAILED = 'failed'


class CacheLifetime(BaseModel):
	"""Optional TTL / warming metadata (enhancement — not required by the base assignment).

	When ``ttl_seconds`` is None, TTL is disabled and warming is a no-op unless
	explicitly forced. Optexity does not ship a platform TTL; this is local to
	the learning/cache layer.
	"""

	ttl_seconds: float | None = None
	created_at: str = Field(default_factory=utc_now_iso)
	expires_at: str | None = None
	last_warmed_at: str | None = None
	warming_enabled: bool = False
	validation_status: CacheValidationStatus = CacheValidationStatus.UNKNOWN
	# Relevance signal for warming policy (incremented by callers on successful use).
	use_count: int = 0
	# Soft lock for idempotent warming (also guarded in-process by a mutex).
	warming_in_progress: bool = False
	last_warm_skip_reason: str | None = None


class ActionCache(BaseModel):
	version: int = 1
	source: dict[str, Any] = Field(default_factory=dict)
	actions: list[CacheAction] = Field(default_factory=list)
	# Non-DOM exploration recorded for reports only (never converted to Automation).
	skipped_actions: list[SkippedHistoryAction] = Field(default_factory=list)
	# Optional workflow fingerprint (URL + task hash, etc.) — set by callers.
	workflow_id: str | None = None
	# Interview / harness facing classification summary (recomputed on load if absent).
	classification_report: ClassificationReport | None = None
	# Phase 2: HEALTHY → REPAIRING → HEALTHY | DEGRADED (never silent full-LLM).
	health: CacheHealth = Field(default_factory=CacheHealth)
	# Phase 3 enhancement: optional TTL + warming (disabled unless configured).
	lifetime: CacheLifetime = Field(default_factory=CacheLifetime)
