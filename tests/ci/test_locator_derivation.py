"""Unit tests for deterministic locator derivation from captured DOM evidence."""

from __future__ import annotations

from browser_use.learning import (
	ActionCache,
	CacheAction,
	CacheActionResult,
	CacheActionSource,
	CacheTarget,
	attribute_value_counts,
	derive_locator,
	enrich_cache_locators,
	looks_dynamic,
	quote_playwright,
)


def test_prefers_role_and_name_when_ax_data_present():
	derived = derive_locator(
		node_name='BUTTON',
		attributes={'type': 'button'},
		xpath='html/body/button',
		ax_name='Sign In',
	)
	assert derived is not None
	assert derived.strategy == 'role+name'
	assert derived.command == 'get_by_role("button", name="Sign In")'
	assert derived.evidence['accessible_name'] == 'Sign In'
	assert derived.evidence['role'] == 'button'


def test_prefers_label_over_name_when_aria_label_present():
	derived = derive_locator(
		node_name='INPUT',
		attributes={'name': 'email', 'type': 'text', 'aria-label': 'Email address'},
		xpath='html/body/input',
		ax_name=None,
	)
	assert derived is not None
	# role+name uses aria-label as strong name first
	assert derived.strategy == 'role+name'
	assert 'Email address' in derived.command


def test_test_id_strategy():
	derived = derive_locator(
		node_name='INPUT',
		attributes={'data-testid': 'full-name', 'name': 'x'},
		xpath='html/body/input',
	)
	assert derived is not None
	assert derived.strategy == 'test_id'
	assert derived.command == 'get_by_test_id("full-name")'
	assert derived.evidence == {'attr': 'data-testid', 'value': 'full-name'}


def test_prefers_unique_test_hook_over_generic_role_name():
	"""Repeated labels like Add to cart need id/test-id when captured on the element."""
	derived = derive_locator(
		node_name='BUTTON',
		attributes={
			'id': 'add-to-cart-sauce-labs-backpack',
			'data-test': 'add-to-cart-sauce-labs-backpack',
			'class': 'btn btn_primary',
		},
		xpath='html/body/button',
		ax_name='Add to cart',
	)
	assert derived is not None
	assert derived.strategy == 'test_id'
	assert "data-test='add-to-cart-sauce-labs-backpack'" in derived.command


def test_stable_id_strategy():
	derived = derive_locator(
		node_name='INPUT',
		attributes={'id': 'user-email', 'type': 'text'},
		xpath='html/body/input',
	)
	assert derived is not None
	assert derived.strategy == 'id'
	assert derived.command == 'locator("#user-email")'
	assert derived.evidence['id'] == 'user-email'


def test_stable_name_from_roboform_style_attrs():
	"""Form names like 04fullname are captured evidence — not invented — and are usable."""
	derived = derive_locator(
		node_name='INPUT',
		attributes={'type': 'text', 'size': '20', 'name': '04fullname', 'value': '', 'style': ''},
		xpath='html/body/div[2]/form/div/div[1]/div[5]/div[2]/input',
	)
	assert derived is not None
	assert derived.strategy == 'name'
	assert derived.command == 'locator("input[name=\'04fullname\']")'
	assert derived.evidence['name'] == '04fullname'
	assert not looks_dynamic('04fullname')


def test_skips_non_unique_name_and_falls_back():
	counts = attribute_value_counts(
		[
			{'attributes': {'name': 'shared', 'type': 'text'}},
			{'attributes': {'name': 'shared', 'type': 'text'}},
		]
	)
	derived = derive_locator(
		node_name='INPUT',
		attributes={'name': 'shared', 'type': 'text'},
		xpath='html/body/form/input[1]',
		sibling_attribute_counts=counts,
	)
	assert derived is not None
	# name skipped as non-unique → xpath last resort
	assert derived.strategy == 'xpath'
	assert 'xpath=/html/' in derived.command


def test_xpath_last_resort_when_only_xpath_usable():
	derived = derive_locator(
		node_name='DIV',
		attributes={'class': 'a1b2c3d4e5f6'},  # dynamic-looking
		xpath='html/body/div[3]/div[2]',
	)
	assert derived is not None
	assert derived.strategy == 'xpath'
	assert derived.command.startswith('locator("xpath=/html/body/div[3]/div[2]")')


def test_css_excludes_state_classes():
	derived = derive_locator(
		node_name='INPUT',
		attributes={'class': 'form_input input_error focused', 'type': 'text'},
		xpath='',
	)
	# No id/name/test-id → css from stable classes only (input_error/focused stripped)
	assert derived is not None
	assert derived.strategy == 'css'
	assert 'input_error' not in derived.command
	assert 'focused' not in derived.command
	assert 'form_input' in derived.command


def test_cache_miss_when_no_evidence():
	derived = derive_locator(
		node_name='DIV',
		attributes={},
		xpath='',
		ax_name=None,
	)
	assert derived is None


def test_rejects_uuid_as_dynamic():
	assert looks_dynamic('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11')
	assert looks_dynamic('react-123abc')
	assert not looks_dynamic('submit-button')
	assert not looks_dynamic('10address1')


def test_skips_icon_font_role_name_prefers_href():
	"""Icon-font PUA glyphs in ax_name must not win over a captured href."""
	derived = derive_locator(
		node_name='A',
		attributes={'href': '/product_details/1', 'style': 'color: brown;'},
		xpath='html/body/a',
		ax_name='\uf0fe View Product',
	)
	assert derived is not None
	assert derived.strategy == 'attribute'
	assert "href='/product_details/1'" in derived.command or 'product_details/1' in derived.command


def test_does_not_invent_bare_locator_something():
	derived = derive_locator(
		node_name='INPUT',
		attributes={'name': 'city', 'type': 'text'},
		xpath='html/body/input',
	)
	assert derived is not None
	assert 'something' not in derived.command
	assert derived.command == 'locator("input[name=\'city\']")'


def test_enrich_marks_cache_miss_without_inventing():
	action = CacheAction(
		sequence=1,
		action_type='click',
		value=None,
		target=CacheTarget(
			node_name='DIV',
			attributes={},
			xpath='',
			element_hash=1,
			stable_hash=1,
		),
		result=CacheActionResult(success=True),
		status='kept',
		source=CacheActionSource(history_step=0, action_index=0, backend_node_id=1),
	)
	cache = ActionCache(actions=[action])
	enrich_cache_locators(cache)
	assert cache.actions[0].status == 'cache_miss'
	assert cache.actions[0].locator is None


def test_quote_playwright_escapes():
	assert quote_playwright('a"b') == '"a\\"b"'
