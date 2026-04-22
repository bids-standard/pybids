import pytest
from unittest.mock import patch

import requests
from types import SimpleNamespace

from bids import utils
from bids.utils import _allowed_bids_versions, collect_schema, matches_entities, convert_JSON


class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or []

    def json(self):
        return self._payload


def test_allowed_bids_versions_non_200(monkeypatch):
    """_allowed_bids_versions should return None on non-200 responses."""

    def fake_get(*args, **kwargs):
        return DummyResponse(status_code=500)

    monkeypatch.setattr(requests, "get", fake_get)
    assert _allowed_bids_versions() is None


def test_allowed_bids_versions_request_exception(monkeypatch):
    """_allowed_bids_versions should return None if requests.get raises."""

    def boom(*args, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr(requests, "get", boom)
    assert _allowed_bids_versions() is None


def test_allowed_bids_versions_success(monkeypatch):
    """_allowed_bids_versions returns versions >= min_version."""

    payload = [
        {"tag_name": "v1.7.0"},
        {"tag_name": "v1.8.0"},
        {"tag_name": "v1.9.0"},
        {"tag_name": "v1.10.1"},
    ]

    def fake_get(*args, **kwargs):
        return DummyResponse(status_code=200, payload=payload)

    monkeypatch.setattr(requests, "get", fake_get)
    allowed = _allowed_bids_versions(min_version="1.8.0")
    assert allowed == {"1.8.0", "1.9.0", "1.10.1"}


def test_collect_schema_uri_and_bids_version_mutually_exclusive():
    """collect_schema should reject simultaneous uri and bids_version."""
    with pytest.raises(ValueError, match="uri and bids_version are mutually exclusive"):
        collect_schema(uri="https://example.com/schema.json", bids_version="1.11.1")


def test_collect_schema_unparseable_bids_version():
    """collect_schema should fail cleanly when it cannot parse bids_version."""
    with pytest.raises(ValueError, match="Unable to determine version from bids_version"):
        collect_schema(bids_version="not-a-version")


@patch("bidsschematools.schema.load_schema")
def test_collect_schema_latest_and_stable(mock_load_schema):
    """collect_schema should resolve 'latest' and 'stable' labels."""

    # latest
    collect_schema(bids_version="latest")
    assert mock_load_schema.call_count == 1
    latest_arg = str(mock_load_schema.call_args_list[0].args[0])
    assert "latest/schema.json" in latest_arg

    mock_load_schema.reset_mock()

    # stable
    collect_schema(bids_version="stable")
    assert mock_load_schema.call_count == 1
    stable_arg = str(mock_load_schema.call_args_list[0].args[0])
    assert "stable/schema.json" in stable_arg


@patch.object(utils, "_allowed_bids_versions", return_value=None)
@patch("requests.head")
@patch("bidsschematools.schema.load_schema")
def test_collect_schema_allowed_versions_none(mock_load_schema, mock_head, mock_allowed):
    """If _allowed_bids_versions returns None, collect_schema should proceed without filtering."""
    mock_head.return_value = DummyResponse(status_code=200)
    collect_schema(bids_version="1.11.1")
    mock_load_schema.assert_called_once()


@patch.object(utils, "_allowed_bids_versions", return_value={"1.9.0"})
def test_collect_schema_invalid_numeric_version_filtered(mock_allowed):
    """Numeric bids_version not in allowed set should raise ValueError."""
    with pytest.raises(ValueError, match="is not an available BIDS release"):
        collect_schema(bids_version="1.0.0")


@patch("bidsschematools.schema.load_schema")
def test_collect_schema_default_latest_when_no_args(mock_load_schema):
    """With no uri or bids_version, collect_schema should default to latest schema URL."""
    collect_schema()
    mock_load_schema.assert_called_once()
    arg = str(mock_load_schema.call_args.args[0])
    assert "latest/schema.json" in arg

def test_match_entities():
    obj = SimpleNamespace(entities={'a': 1, 'b':2, 'c':3})
    entities = {'a': 1, 'b': 2, 'c': 3, 'd':4}
    assert matches_entities(obj, entities, strict=True) is False
    obj = SimpleNamespace(entities=entities)
    assert matches_entities(obj, entities) is True
    obj = SimpleNamespace(entities={'a': 2})
    entities = {'a': (1, 2, 3)}
    assert matches_entities(obj, entities) is True
    obj = SimpleNamespace(entities={'a': 4})
    assert matches_entities(obj, entities) is False
    entities = {'a': 1}
    assert matches_entities(obj, entities) is False

def test_convert_JSON():
    j = {
        "camelCase": "dontUseCamelCaseInPython",
        "array": [
            {
                "moreNonesense": [1, 2, 3]
            },
            {
                "evenMore": "a"
            }
        ],
        "notReplace": {
            "innerCamel": "value"
        },
        "Replace": {
            "DoNotTouchMe": "x"
        },
        "arrayedArray": [
            ["howDeepItGoes", "NoOneKnows"]
        ]
    }
    converted = convert_JSON(j)
    assert converted.get('camel_case', None) == 'dontUseCamelCaseInPython'
    assert isinstance(converted.get('array'), list)
    assert converted['array'][0]['more_nonesense'] == [1, 2, 3]
    assert converted['array'][1]['even_more'] == 'a'
    assert converted['not_replace'] == {'inner_camel': 'value'}
    assert converted['replace'] == {"DoNotTouchMe": "x"}
    assert converted['arrayed_array'] == [["howDeepItGoes", "NoOneKnows"]]