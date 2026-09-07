"""Compatibility shim — prefer ``browser_use.learning``.

History → cache extraction lives in ``browser_use.learning.extractor`` /
``browser_use.learning.cache`` / ``browser_use.learning.models``.
"""

from __future__ import annotations

from browser_use.learning.cache import build_and_save_action_cache, save_action_cache
from browser_use.learning.extractor import build_action_cache, enrich_cache_locators
from browser_use.learning.models import (
	ActionCache,
	CacheAction,
	CacheActionResult,
	CacheActionSource,
	CacheLocator,
	CacheStatus,
	CacheTarget,
)

__all__ = [
	'ActionCache',
	'CacheAction',
	'CacheActionResult',
	'CacheActionSource',
	'CacheLocator',
	'CacheStatus',
	'CacheTarget',
	'build_action_cache',
	'build_and_save_action_cache',
	'enrich_cache_locators',
	'save_action_cache',
]
