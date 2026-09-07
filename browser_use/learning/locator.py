"""DOMInteractedElement / captured attrs → LocatorCandidate → Playwright command.

Locators are derived only from fields present on the cached target (attributes,
xpath, ax_name, tag). Nothing is invented. Preference order matches the
assignment's robust-selector guidance; XPath is last resort.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

LocatorStrategy = Literal[
	'role+name',
	'label',
	'test_id',
	'id',
	'name',
	'attribute',
	'css',
	'xpath',
]

_TEST_ID_ATTRS = ('data-testid', 'data-test-id', 'data-test', 'data-cy', 'data-qa')
_STABLE_ATTR_CANDIDATES = (
	'placeholder',
	'title',
	'aria-labelledby',
	'for',
	'type',
	'href',
	'alt',
)

_UUID_RE = re.compile(
	r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
	re.I,
)
_DYNAMIC_PREFIX_RE = re.compile(
	r'^(ember\d|react-|ng-|css-|mui-|emotion-|svelte-)',
	re.I,
)
_HEX_HASH_RE = re.compile(r'^[a-f0-9]{10,}$', re.I)

# Transient / state classes that must not be baked into CSS locators.
_STATE_CLASS_EXACT = frozenset(
	{
		'active',
		'checked',
		'disabled',
		'error',
		'focus',
		'focused',
		'hover',
		'input_error',
		'invalid',
		'is-invalid',
		'is-valid',
		'loading',
		'open',
		'selected',
		'ng-dirty',
		'ng-invalid',
		'ng-pristine',
		'ng-touched',
		'ng-untouched',
		'ng-valid',
	}
)

# Implicit ARIA roles from captured tag/type — not invented selectors; evidence records this.
_INPUT_TYPE_TO_ROLE = {
	'checkbox': 'checkbox',
	'radio': 'radio',
	'button': 'button',
	'submit': 'button',
	'reset': 'button',
	'text': 'textbox',
	'search': 'searchbox',
	'email': 'textbox',
	'tel': 'textbox',
	'url': 'textbox',
	'password': 'textbox',
	'number': 'spinbutton',
}
_TAG_TO_ROLE = {
	'button': 'button',
	'a': 'link',
	'select': 'combobox',
	'textarea': 'textbox',
	'img': 'img',
}


class LocatorCandidate(BaseModel):
	"""Playwright locator suitable for Optexity BaseAction.command (no page. prefix)."""

	strategy: LocatorStrategy
	command: str
	evidence: dict[str, Any] = Field(default_factory=dict)


# Back-compat alias used in earlier stages / tests.
DerivedLocator = LocatorCandidate


def derive_locator(
	*,
	node_name: str,
	attributes: dict[str, str] | None,
	xpath: str | None,
	ax_name: str | None = None,
	sibling_attribute_counts: dict[tuple[str, str], int] | None = None,
) -> LocatorCandidate | None:
	"""Pick the best locator supported by the captured element data.

	Returns None when no non-invented locator can be formed (cache miss).
	``sibling_attribute_counts`` maps ``(attr, value) -> count`` across peers so
	non-unique id/name/test-id/label values are skipped when possible.

	When a unique id/test-id hook exists on the same element, role+name / label
	are deferred: accessible names like \"Add to cart\" often match many peers
	on a page, while id/test-id are purpose-built for uniqueness.
	"""
	candidates = list_locator_alternatives(
		node_name=node_name,
		attributes=attributes,
		xpath=xpath,
		ax_name=ax_name,
		sibling_attribute_counts=sibling_attribute_counts,
	)
	return candidates[0] if candidates else None


def list_locator_alternatives(
	*,
	node_name: str,
	attributes: dict[str, str] | None,
	xpath: str | None,
	ax_name: str | None = None,
	sibling_attribute_counts: dict[tuple[str, str], int] | None = None,
	exclude_commands: set[str] | None = None,
) -> list[LocatorCandidate]:
	"""All non-invented locator candidates from captured evidence (preference order).

	Used by minimal cache repair to try cheaper alternatives before LLM repair.
	Does not invent selectors — only strategies already supported by this module.
	"""
	attrs = {k: (v or '').strip() for k, v in (attributes or {}).items() if v is not None}
	tag = (node_name or '*').strip().lower() or '*'
	counts = sibling_attribute_counts or {}
	has_unique_hook = _has_unique_identity_hook(attrs, counts)
	excluded = exclude_commands or set()

	out: list[LocatorCandidate] = []
	seen: set[str] = set()
	for strategy, locator_expr, evidence in _ordered_candidates(
		tag=tag, attrs=attrs, xpath=xpath or '', ax_name=ax_name
	):
		if locator_expr in excluded or locator_expr in seen:
			continue
		if not _is_sufficiently_specific(
			strategy,
			evidence,
			counts,
			has_unique_identity_hook=has_unique_hook,
		):
			continue
		seen.add(locator_expr)
		out.append(LocatorCandidate(strategy=strategy, command=locator_expr, evidence=evidence))
	return out


def locator_confidence(strategy: str | None) -> float | None:
	"""Simple confidence prior from strategy rank (1.0 = strongest)."""
	if not strategy:
		return None
	rank = {
		'role+name': 1.0,
		'label': 0.95,
		'test_id': 0.9,
		'id': 0.85,
		'name': 0.7,
		'attribute': 0.55,
		'css': 0.4,
		'xpath': 0.25,
	}
	return rank.get(strategy)


def attribute_value_counts(targets: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
	"""Frequency of attribute values across targets for uniqueness checks."""
	counts: dict[tuple[str, str], int] = {}
	keys = ('id', 'name', 'aria-label', *_TEST_ID_ATTRS)
	for target in targets:
		attrs = target.get('attributes') or {}
		for key in keys:
			val = (attrs.get(key) or '').strip()
			if not val:
				continue
			counts[(key, val)] = counts.get((key, val), 0) + 1
	return counts


def _contains_private_use_glyphs(value: str) -> bool:
	"""True when accessible name includes icon-font Private Use Area code points.

	Sites often prefix labels with icon glyphs (e.g. ``\\uf0fe View Product``).
	Those names are poor role+name locators because every peer shares the same
	glyph+label and Playwright strict mode fails. Prefer href/attribute instead.
	"""
	return any(0xE000 <= ord(ch) <= 0xF8FF for ch in value)


def looks_dynamic(value: str) -> bool:
	"""Heuristic for auto-generated / unstable attribute values.

	Allows semantic form names with short numeric prefixes (e.g. ``04fullname``).
	Rejects UUIDs, framework prefixes, long hex hashes, and long digit runs.
	"""
	if not value or not value.strip():
		return True
	v = value.strip()
	if _UUID_RE.search(v) or _DYNAMIC_PREFIX_RE.match(v):
		return True
	if _HEX_HASH_RE.fullmatch(v):
		return True
	if re.search(r'\d{6,}', v):  # long counters / timestamps
		return True
	# css-module / styled-components hash tails (letter+digit soup segments)
	for seg in re.split(r'[\s_\-]+', v):
		if len(seg) < 6:
			continue
		digits = sum(c.isdigit() for c in seg)
		alphas = sum(c.isalpha() for c in seg)
		vowels = sum(c in 'aeiouAEIOU' for c in seg)
		if alphas and digits >= 3 and vowels == 0:
			return True
		if alphas and digits >= 4:
			return True
	return False


def quote_playwright(value: str, max_len: int | None = None) -> str:
	"""Quote a string for Playwright locator commands.

	Does not truncate by default — truncating breaks exact name/label matches.
	Pass ``max_len`` only for logging / display helpers.
	"""
	collapsed = ' '.join(value.split())
	escaped = collapsed.replace('\\', '\\\\').replace('"', '\\"')
	if max_len is not None and len(escaped) > max_len:
		escaped = escaped[: max_len - 3] + '...'
	return f'"{escaped}"'


def css_attr_selector(tag: str, attr: str, value: str) -> str:
	"""CSS attribute selector safe inside locator(\"...\")."""
	safe = value.replace('\\', '\\\\').replace("'", r'\'')
	return f"{tag}[{attr}='{safe}']"


def resolve_accessible_role(tag: str, attrs: dict[str, str]) -> tuple[str | None, dict[str, Any]]:
	"""Role from explicit attribute or implied by captured tag/type."""
	explicit = (attrs.get('role') or '').strip()
	if explicit and not looks_dynamic(explicit):
		return explicit, {'role_source': 'attribute', 'role': explicit}

	input_type = (attrs.get('type') or '').strip().lower()
	if tag == 'input':
		role = _INPUT_TYPE_TO_ROLE.get(input_type or 'text', 'textbox')
		return role, {'role_source': 'inferred_from_tag_type', 'tag': tag, 'type': input_type or 'text', 'role': role}

	role = _TAG_TO_ROLE.get(tag)
	if role:
		return role, {'role_source': 'inferred_from_tag', 'tag': tag, 'role': role}
	return None, {}


def resolve_accessible_name(attrs: dict[str, str], ax_name: str | None) -> tuple[str | None, dict[str, Any]]:
	if ax_name and ax_name.strip() and not looks_dynamic(ax_name):
		return ax_name.strip(), {'name_source': 'ax_name', 'accessible_name': ax_name.strip()}
	for key in ('aria-label', 'placeholder', 'title', 'alt'):
		val = (attrs.get(key) or '').strip()
		if val and not looks_dynamic(val):
			return val, {'name_source': key, 'accessible_name': val}
	return None, {}


def _is_unique(attr: str, value: str, counts: dict[tuple[str, str], int]) -> bool:
	if not counts:
		return True  # no peer info — treat as unique enough offline
	return counts.get((attr, value), 1) <= 1


def _has_unique_identity_hook(attrs: dict[str, str], counts: dict[tuple[str, str], int]) -> bool:
	"""True when captured attrs include a stable unique id or test-id style hook."""
	for attr in _TEST_ID_ATTRS:
		val = (attrs.get(attr) or '').strip()
		if val and not looks_dynamic(val) and _is_unique(attr, val, counts):
			return True
	el_id = (attrs.get('id') or '').strip()
	if el_id and not looks_dynamic(el_id) and _is_unique('id', el_id, counts):
		return True
	return False


def _is_sufficiently_specific(
	strategy: LocatorStrategy,
	evidence: dict[str, Any],
	counts: dict[tuple[str, str], int],
	*,
	has_unique_identity_hook: bool = False,
) -> bool:
	# Prefer purpose-built unique hooks over role/label when both exist.
	if strategy in ('role+name', 'label') and has_unique_identity_hook:
		return False
	if strategy in ('css', 'xpath', 'attribute'):
		return True
	if strategy == 'role+name':
		return True
	if strategy == 'label':
		val = evidence.get('aria-label') or evidence.get('accessible_name')
		return bool(val) and _is_unique('aria-label', str(val), counts)
	if strategy == 'test_id':
		attr = evidence.get('attr')
		val = evidence.get('value')
		return bool(attr and val) and _is_unique(str(attr), str(val), counts)
	if strategy == 'id':
		val = evidence.get('id')
		return bool(val) and _is_unique('id', str(val), counts)
	if strategy == 'name':
		val = evidence.get('name')
		return bool(val) and _is_unique('name', str(val), counts)
	return True


def _ordered_candidates(
	*,
	tag: str,
	attrs: dict[str, str],
	xpath: str,
	ax_name: str | None,
) -> list[tuple[LocatorStrategy, str, dict[str, Any]]]:
	"""Candidates in preference order; caller picks the first sufficiently specific one."""
	out: list[tuple[LocatorStrategy, str, dict[str, Any]]] = []

	# 1. accessible role + accessible name
	role, role_ev = resolve_accessible_role(tag, attrs)
	# Prefer true accessible name (ax / aria-label), not placeholder-only for role+name
	strong_name = None
	strong_ev: dict[str, Any] = {}
	if ax_name and ax_name.strip() and not looks_dynamic(ax_name):
		strong_name = ax_name.strip()
		strong_ev = {'name_source': 'ax_name', 'accessible_name': strong_name}
	elif (attrs.get('aria-label') or '').strip() and not looks_dynamic(attrs.get('aria-label', '')):
		strong_name = attrs['aria-label'].strip()
		strong_ev = {'name_source': 'aria-label', 'accessible_name': strong_name}

	# Skip icon-font-polluted names: they collide across product grids / nav icons.
	if role and strong_name and not _contains_private_use_glyphs(strong_name):
		out.append(
			(
				'role+name',
				f'get_by_role({quote_playwright(role)}, name={quote_playwright(strong_name)})',
				{**role_ev, **strong_ev},
			)
		)

	# 2. label
	aria_label = (attrs.get('aria-label') or '').strip()
	if aria_label and not looks_dynamic(aria_label):
		out.append(
			(
				'label',
				f'get_by_label({quote_playwright(aria_label)})',
				{'aria-label': aria_label},
			)
		)

	# 3. test id
	for attr in _TEST_ID_ATTRS:
		val = (attrs.get(attr) or '').strip()
		if not val or looks_dynamic(val):
			continue
		if attr == 'data-testid':
			cmd = f'get_by_test_id({quote_playwright(val)})'
		else:
			cmd = f'locator({quote_playwright(css_attr_selector(tag, attr, val), 400)})'
		out.append(('test_id', cmd, {'attr': attr, 'value': val}))
		break

	# 4. stable unique id
	el_id = (attrs.get('id') or '').strip()
	if el_id and not looks_dynamic(el_id):
		if re.match(r'^[A-Za-z][\w-]*$', el_id):
			sel = f'#{el_id}'
		else:
			sel = css_attr_selector(tag, 'id', el_id)
		out.append(('id', f'locator({quote_playwright(sel, 400)})', {'id': el_id}))

	# 5. stable unique name
	nm = (attrs.get('name') or '').strip()
	if nm and not looks_dynamic(nm):
		sel = css_attr_selector(tag, 'name', nm)
		out.append(('name', f'locator({quote_playwright(sel, 400)})', {'name': nm, 'tag': tag}))

	# 6. other stable attributes
	for attr in _STABLE_ATTR_CANDIDATES:
		if attr in ('type',) and tag == 'input':
			# type alone is almost never unique
			continue
		val = (attrs.get(attr) or '').strip()
		if not val or looks_dynamic(val):
			continue
		if attr == 'placeholder':
			out.append(
				(
					'attribute',
					f'get_by_placeholder({quote_playwright(val)})',
					{'attr': attr, 'value': val},
				)
			)
		else:
			sel = css_attr_selector(tag, attr, val)
			out.append(
				(
					'attribute',
					f'locator({quote_playwright(sel, 400)})',
					{'attr': attr, 'value': val},
				)
			)

	# 7. CSS locator from stable classes (exclude transient state classes)
	stable_classes = [
		c
		for c in (attrs.get('class') or '').split()
		if c
		and not looks_dynamic(c)
		and c.lower() not in _STATE_CLASS_EXACT
		and not c.lower().endswith('_error')
		and not c.lower().startswith('is-')
	]
	if stable_classes:
		sel = tag + ''.join(f'.{c}' for c in stable_classes[:3])
		out.append(('css', f'locator({quote_playwright(sel)})', {'classes': stable_classes[:3], 'tag': tag}))

	# 8. XPath last resort (still derived from capture — not invented)
	xp = (xpath or '').strip()
	if xp:
		raw = xp[len('xpath=') :] if xp.startswith('xpath=') else xp
		# Playwright xpath= needs a leading / (or //) for absolute paths from capture.
		if raw and not raw.startswith('/') and not raw.startswith('('):
			raw = '/' + raw
		xpath_expr = f'xpath={raw}'
		out.append(('xpath', f'locator({quote_playwright(xpath_expr)})', {'xpath': xp}))

	return out
