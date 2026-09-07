"""Redundancy / exploration classification for cached DOM actions.

Architecture (keeps extractor thin):

    AgentHistoryList
         ↓
    normalize (extractor)
         ↓
    classify_actions (this module)
         ↓
    attach locators → ActionCache

Conservative rule: when signals conflict or evidence is weak → UNCERTAIN and KEEP.

Statuses mapped for replay compatibility:
  required / uncertain / retry(success)  → kept
  redundant / exploratory                → dropped_redundant
  failed attempts                        → dropped_failed (set earlier)
  value corrections                      → dropped_superseded (set earlier)

No website-specific logic. No invented selectors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from browser_use.learning.models import (
	ActionClassification,
	CacheAction,
	CacheStatus,
	CacheTarget,
	ClassificationReport,
	ClassificationReportEntry,
	SkippedHistoryAction,
)


def same_element(a: CacheTarget, b: CacheTarget) -> bool:
	"""Same identity strategy as Agent._is_redundant_retry_step."""
	if a.element_hash == b.element_hash:
		return True
	if a.stable_hash is not None and b.stable_hash is not None and a.stable_hash == b.stable_hash:
		return True
	if a.xpath and b.xpath and a.xpath == b.xpath:
		return True
	return False


def action_signature(entry: CacheAction) -> tuple[Any, ...]:
	return (entry.action_type, entry.value)


def _urls_equivalent(a: str | None, b: str | None) -> bool:
	if a is None or b is None:
		return False
	return a.rstrip('/') == b.rstrip('/')


def _url_changed(entry: CacheAction) -> bool | None:
	"""True/False if known; None if unknown (conservative → not a no-op)."""
	if entry.page_url is None or entry.next_page_url is None:
		return None
	return not _urls_equivalent(entry.page_url, entry.next_page_url)


def _set_class(
	entry: CacheAction,
	classification: ActionClassification,
	reasons: list[str],
	*,
	status: CacheStatus | None = None,
) -> None:
	entry.classification = classification
	entry.classification_reasons = list(reasons)
	if status is not None:
		entry.status = status


def mark_failed_and_retries(entries: list[CacheAction]) -> None:
	"""Failed gestures stay dropped_failed; classify as retry when later success exists."""
	for i, entry in enumerate(entries):
		if entry.status != 'dropped_failed':
			continue
		later_success = any(
			e.result.success
			and same_element(e.target, entry.target)
			and action_signature(e) == action_signature(entry)
			for e in entries[i + 1 :]
		)
		if later_success:
			_set_class(
				entry,
				'retry',
				[
					'prior attempt failed',
					'later successful action on same target',
					'not treated as redundant duplicate — failure recovery',
				],
			)
		else:
			_set_class(
				entry,
				'retry',
				['action failed; excluded from deterministic replay'],
			)


def mark_superseded_corrections(entries: list[CacheAction]) -> None:
	"""Keep only the last successful value per target for input/select/upload."""
	latest_by_target: list[CacheAction] = []

	for entry in entries:
		if entry.action_type not in ('input', 'select_dropdown', 'upload_file'):
			continue
		if entry.status != 'kept' or not entry.result.success:
			continue

		replaced = False
		for i, prior in enumerate(latest_by_target):
			if not same_element(prior.target, entry.target):
				continue
			if prior.value != entry.value:
				_set_class(
					prior,
					'redundant',
					[
						'value corrected by a later successful input on the same target',
						'earlier value would pollute deterministic replay',
					],
					status='dropped_superseded',
				)
			latest_by_target[i] = entry
			replaced = True
			break
		if not replaced:
			latest_by_target.append(entry)


def mark_exact_retries_after_success(entries: list[CacheAction]) -> None:
	"""Later exact retries after a successful earlier action → redundant."""
	last_success: list[CacheAction] = []

	for entry in entries:
		if entry.status in ('dropped_failed', 'dropped_superseded'):
			continue

		redundant = False
		for prior in reversed(last_success):
			if not same_element(prior.target, entry.target):
				continue
			if action_signature(prior) != action_signature(entry):
				break
			# Pagination: same click type on a *different* page_url → keep.
			if (
				entry.action_type == 'click'
				and prior.page_url
				and entry.page_url
				and not _urls_equivalent(prior.page_url, entry.page_url)
			):
				break
			redundant = True
			break

		if redundant:
			_set_class(
				entry,
				'redundant',
				[
					'duplicate action after prior success on same target',
					'same action type and value',
					'no evidence this retry is required for a new page state',
				],
				status='dropped_redundant',
			)
		elif entry.result.success and entry.status == 'kept':
			last_success.append(entry)


def coalesce_focus_click_before_input(entries: list[CacheAction]) -> None:
	"""click → input on same element: click is discovery/focus once fill exists."""
	for i, entry in enumerate(entries):
		if entry.action_type != 'click' or entry.status != 'kept':
			continue
		if _url_changed(entry) is True:
			continue
		for later in entries[i + 1 :]:
			if later.status != 'kept' or not later.result.success:
				continue
			if later.action_type != 'input':
				continue
			if not same_element(entry.target, later.target):
				continue
			_set_class(
				entry,
				'exploratory',
				[
					'focus/discovery click before successful input on same element',
					'deterministic fill captures the target; click not required for replay',
				],
				status='dropped_redundant',
			)
			break


def mark_noop_duplicate_clicks(entries: list[CacheAction]) -> None:
	"""Duplicate successful clicks with known no URL change."""
	seen: list[CacheAction] = []
	for entry in entries:
		if entry.action_type != 'click' or entry.status != 'kept' or not entry.result.success:
			continue
		changed = _url_changed(entry)
		dup = False
		for prior in seen:
			if not same_element(prior.target, entry.target):
				continue
			if action_signature(prior) != action_signature(entry):
				continue
			prior_changed = _url_changed(prior)
			if changed is False and prior_changed is False:
				dup = True
				break
			# Unknown URL delta → keep (do not mark redundant).
			if changed is None or prior_changed is None:
				dup = False
				break
		if dup:
			_set_class(
				entry,
				'redundant',
				[
					'duplicate click on same element',
					'no URL/page state change detected',
					'prior equivalent click already retained',
				],
				status='dropped_redundant',
			)
		else:
			seen.append(entry)


def mark_enabling_clicks(entries: list[CacheAction]) -> None:
	"""Retain successful clicks that enable a later different-target action on the same page.

	Example: click Menu → click Settings (no URL change). Menu is REQUIRED.
	Does not override already-classified exploratory focus-before-input clicks.
	"""
	for i, entry in enumerate(entries):
		if entry.classification is not None:
			continue
		if entry.action_type != 'click' or entry.status != 'kept' or not entry.result.success:
			continue
		# Navigation already proven via URL — leave for assign_default_required.
		if _url_changed(entry) is True:
			continue
		enables = False
		for later in entries[i + 1 :]:
			if later.status not in ('kept',) or not later.result.success:
				continue
			if same_element(entry.target, later.target):
				# Same target later → focus coalescing / retries handle this.
				continue
			# Later successful interaction on a different element while still on
			# a comparable page (or unknown URL) → this click may have enabled UI.
			if later.page_url and entry.page_url and not _urls_equivalent(
				later.page_url, entry.page_url
			):
				# Later action is on a different page; this click did not stay local.
				# If this click itself changed URL, navigation signal applies instead.
				continue
			if later.action_type in ('click', 'input', 'select_dropdown'):
				enables = True
				break
		if enables:
			_set_class(
				entry,
				'required',
				[
					'enables subsequent action on a different target',
					'no evidence this click is a no-op discovery-only step',
				],
			)


def assign_default_required(entries: list[CacheAction]) -> None:
	"""Anything still kept without classification → required (or uncertain)."""
	for entry in entries:
		if entry.classification is not None:
			continue
		if entry.status == 'cache_miss':
			_set_class(
				entry,
				'uncertain',
				[
					'no reliable locator derived from captured evidence',
					'retained for explainability; not convertible without repair',
				],
			)
			continue
		if entry.status != 'kept':
			continue
		reasons = ['successful DOM action retained for deterministic replay']
		if entry.action_type in ('input', 'select_dropdown', 'upload_file'):
			reasons.append('state-changing input contributes to form/application state')
		elif entry.action_type == 'click':
			if _url_changed(entry) is True:
				reasons.append('click changes URL/page state')
			else:
				reasons.append('click retained (may open UI or enable later actions)')
		_set_class(entry, 'required', reasons)


def classify_actions(entries: list[CacheAction]) -> None:
	"""Run multi-signal classification. Order matters; later passes refine earlier."""
	mark_failed_and_retries(entries)
	mark_superseded_corrections(entries)
	mark_exact_retries_after_success(entries)
	coalesce_focus_click_before_input(entries)
	mark_noop_duplicate_clicks(entries)
	mark_enabling_clicks(entries)
	assign_default_required(entries)


def validate_removal_candidates(
	entries: list[CacheAction],
	*,
	probe: Callable[[list[CacheAction]], bool],
	max_candidates: int = 3,
) -> list[CacheAction]:
	"""Bounded per-candidate validation (no exponential search).

	``probe(actions)`` returns True iff the proposed retained list is sufficient.

	- probe(kept) True → removal of already-dropped candidates stays OK
	- probe(kept) False while probe(kept + [candidate]) True → restore candidate
	"""
	restored: list[CacheAction] = []
	baseline = [e for e in entries if e.status == 'kept']
	candidates = [
		e
		for e in entries
		if e.status == 'dropped_redundant'
		and e.classification in ('redundant', 'exploratory')
	][: max(0, max_candidates)]

	for cand in candidates:
		ok_without = probe(baseline)
		ok_with = probe(baseline + [cand])
		if (not ok_without) and ok_with:
			_set_class(
				cand,
				'required',
				[
					'removal validation failed',
					'workflow probe succeeds only when this action is retained',
				],
				status='kept',
			)
			restored.append(cand)
			baseline = [e for e in entries if e.status == 'kept']
		elif ok_without:
			cand.classification_reasons = list(cand.classification_reasons) + [
				'removal validation succeeded (probe ok without this action)',
			]
	return restored


def build_classification_report(
	entries: list[CacheAction],
	*,
	skipped: list[SkippedHistoryAction] | None = None,
) -> ClassificationReport:
	skipped = skipped or []
	counts = {
		'required': 0,
		'redundant': 0,
		'exploratory': 0,
		'uncertain': 0,
		'retry': 0,
	}
	retained_detail: list[ClassificationReportEntry] = []
	removed_detail: list[ClassificationReportEntry] = []
	all_entries: list[ClassificationReportEntry] = []

	for entry in entries:
		cls: ActionClassification | None = entry.classification
		if cls is None:
			# Legacy caches predating classification labels.
			if entry.status == 'dropped_redundant':
				cls = 'redundant'
			elif entry.status == 'dropped_superseded':
				cls = 'redundant'
			elif entry.status == 'dropped_failed':
				cls = 'retry'
			elif entry.status == 'kept':
				cls = 'required'
			elif entry.status == 'cache_miss':
				cls = 'uncertain'
			else:
				cls = 'uncertain'
		if cls in counts:
			counts[cls] += 1
		removed = entry.status in (
			'dropped_redundant',
			'dropped_failed',
			'dropped_superseded',
		)
		row = ClassificationReportEntry(
			step=entry.sequence,
			step_id=entry.step_id,
			action_type=entry.action_type,
			classification=cls,
			reason=list(entry.classification_reasons)
			or (
				['legacy status mapped to classification']
				if entry.classification is None
				else []
			),
			status=entry.status,
			command=entry.locator.command if entry.locator else None,
			value=entry.value,
			removed=removed,
		)
		all_entries.append(row)
		if removed:
			removed_detail.append(row)
		elif entry.status == 'kept':
			retained_detail.append(row)

	for sk in skipped:
		counts['exploratory'] += 1
		row = ClassificationReportEntry(
			step=sk.history_step,
			action_type=sk.action_type,
			classification='exploratory',
			reason=list(sk.classification_reasons),
			removed=True,
		)
		all_entries.append(row)
		removed_detail.append(row)

	retained = sum(1 for e in entries if e.status == 'kept')
	return ClassificationReport(
		observed_actions=len(entries) + len(skipped),
		required_actions=counts['required'],
		redundant_actions=counts['redundant'],
		exploratory_actions=counts['exploratory'],
		uncertain_actions=counts['uncertain'],
		retry_actions=counts['retry'],
		retained_actions=retained,
		removed_actions=len(removed_detail),
		skipped_non_dom=len(skipped),
		entries=all_entries,
		removed_detail=removed_detail,
		retained_detail=retained_detail,
	)
