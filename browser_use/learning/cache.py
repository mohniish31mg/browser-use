"""CachedAction[] ↔ cache.json persistence."""

from __future__ import annotations

import json
from pathlib import Path

from browser_use.agent.views import AgentHistoryList
from browser_use.learning.extractor import build_action_cache
from browser_use.learning.models import ActionCache


def save_action_cache(cache: ActionCache, filepath: str | Path) -> Path:
	"""Write cache JSON to disk. Creates parent directories as needed."""
	path = Path(filepath)
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(cache.model_dump(mode='json'), f, indent=2)
	return path


def load_action_cache(filepath: str | Path) -> ActionCache:
	"""Load and validate an ActionCache from JSON."""
	path = Path(filepath)
	with open(path, encoding='utf-8') as f:
		data = json.load(f)
	return ActionCache.model_validate(data)


def build_and_save_action_cache(history: AgentHistoryList, filepath: str | Path) -> ActionCache:
	"""Convenience: extract cache and write JSON in one call."""
	cache = build_action_cache(history)
	save_action_cache(cache, filepath)
	return cache
