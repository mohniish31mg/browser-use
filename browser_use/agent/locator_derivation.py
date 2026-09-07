"""Compatibility shim — prefer ``browser_use.learning.locator``."""

from __future__ import annotations

from browser_use.learning.locator import (
	DerivedLocator,
	LocatorCandidate,
	LocatorStrategy,
	attribute_value_counts,
	css_attr_selector,
	derive_locator,
	looks_dynamic,
	quote_playwright,
	resolve_accessible_name,
	resolve_accessible_role,
)

__all__ = [
	'DerivedLocator',
	'LocatorCandidate',
	'LocatorStrategy',
	'attribute_value_counts',
	'css_attr_selector',
	'derive_locator',
	'looks_dynamic',
	'quote_playwright',
	'resolve_accessible_name',
	'resolve_accessible_role',
]
