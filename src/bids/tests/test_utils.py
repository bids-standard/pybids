"""Tests for bids.utils (get_schema and related)."""

import os
from unittest.mock import patch

import pytest

from bids import config, utils


@pytest.fixture(autouse=True)
def reset_schema_cache():
    """Reset the global schema cache before each test so path/fail_silently are exercised."""
    utils.bids_schema = None
    yield
    utils.bids_schema = None


def _schema_has_expected_structure(schema):
    """Assert the returned object looks like a BIDS schema."""
    assert schema is not None
    assert hasattr(schema, "rules") and hasattr(schema.rules, "entities")
    assert hasattr(schema, "objects")
    assert hasattr(schema, "schema_version") and schema.schema_version
    assert hasattr(schema, "bids_version") and schema.bids_version


def test_get_schema_bundled():
    """path='bundled' loads the schema packaged with bidsschematools."""
    schema = utils.get_schema(path="bundled")
    print(f"test_get_schema_bundled: schema_version={schema.schema_version}")
    _schema_has_expected_structure(schema)


def test_get_schema_none_uses_bundled():
    """path=None uses the bundled schema (no remote fetch)."""
    schema = utils.get_schema(path=None)
    print(f"test_get_schema_none_uses_bundled: schema_version={schema.schema_version}")
    _schema_has_expected_structure(schema)


def test_get_schema_stable():
    """path='stable' loads the schema from the spec website (or bundled on failure)."""
    schema = utils.get_schema(path="stable", fail_silently=False)
    print(f"test_get_schema_stable: schema_version={schema.schema_version}")
    _schema_has_expected_structure(schema)


def test_get_schema_latest():
    """path='latest' loads the schema from the spec website (or bundled on failure)."""
    schema = utils.get_schema(path="latest", fail_silently=False)
    print(f"test_get_schema_latest: schema_version={schema.schema_version}")
    _schema_has_expected_structure(schema)


def test_get_schema_path_to_bundled_file():
    """path=<Path> to a valid schema file loads that schema."""
    from bidsschematools.data import load as bst_load

    path = bst_load.readable("schema.json")
    if not path or not getattr(path, "is_file", lambda: False)():
        pytest.skip("bidsschematools bundled schema.json not found")
    schema = utils.get_schema(path=path)
    print(f"test_get_schema_path_to_bundled_file: schema_version={schema.schema_version}")
    _schema_has_expected_structure(schema)


def test_get_schema_junk_url_raises():
    """path=<invalid URL> raises when fail_silently is False."""
    with pytest.raises(Exception):
        utils.get_schema(
            path="https://invalid.example.com/nonexistent/schema.json",
            fail_silently=False,
        )


def test_get_schema_junk_url_fail_silently_returns_bundled():
    """path=<invalid URL> with fail_silently=True falls back to bundled schema."""
    schema = utils.get_schema(
        path="https://invalid.example.com/nonexistent/schema.json",
        fail_silently=True,
    )
    print(f"test_get_schema_junk_url_fail_silently_returns_bundled: schema_version={schema.schema_version}")
    _schema_has_expected_structure(schema)


def test_get_schema_caching():
    """Second call returns the same cached schema regardless of path."""
    first = utils.get_schema(path="bundled")
    second = utils.get_schema(path="stable")
    print(f"test_get_schema_caching: schema_version={first.schema_version} (cached)")
    assert first is second


def test_get_schema_verify_ssl_disabled():
    """With schema_verify_ssl False (config) or BIDS_SCHEMA_VERIFY_SSL=0 (env), get_schema fetches the actual schema from the URL without SSL verification and emits InsecureRequestWarning."""
    import urllib3.exceptions

    from bidsschematools.schema import load_schema as bst_load_schema

    bundled = bst_load_schema()
    bundled_version = bundled.schema_version

    def _fetch_stable_with_verify_disabled_and_assert():
        utils.bids_schema = None
        try:
            with pytest.warns(urllib3.exceptions.InsecureRequestWarning):
                result = utils.get_schema(path="stable", fail_silently=False)
        except Exception as e:
            pytest.skip(f"Could not fetch schema from URL: {e}")
        _schema_has_expected_structure(result)
        assert result.schema_version != bundled_version, "expected schema from URL (stable), not bundled"
        return result

    # Test via config: disable SSL verification in pybids config
    with patch.object(config, "get_option", return_value=False):
        env_orig = os.environ.pop("BIDS_SCHEMA_VERIFY_SSL", None)
        try:
            _fetch_stable_with_verify_disabled_and_assert()
        finally:
            if env_orig is not None:
                os.environ["BIDS_SCHEMA_VERIFY_SSL"] = env_orig

    # Test via env: BIDS_SCHEMA_VERIFY_SSL=0
    os.environ["BIDS_SCHEMA_VERIFY_SSL"] = "0"
    try:
        _fetch_stable_with_verify_disabled_and_assert()
    finally:
        os.environ.pop("BIDS_SCHEMA_VERIFY_SSL", None)

