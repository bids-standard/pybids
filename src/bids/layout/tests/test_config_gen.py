"""Tests for the schema-driven config generator."""
import json
import re
from pathlib import Path

import pytest

from bids.layout.writing import build_path

try:
    from bidsschematools.schema import load_schema
    HAS_SCHEMA = True
except ImportError:
    HAS_SCHEMA = False

pytestmark = pytest.mark.skipif(
    not HAS_SCHEMA, reason="bidsschematools not installed"
)

from bids.layout.config_gen import (
    _build_datatype_entity,
    _build_extension_entity,
    _build_suffix_entity,
    _choose_default_extension,
    _format_entity_segment,
    _get_directory_entities,
    _get_format_patterns,
    _resolve_enum,
    bids_path,
    generate_entities,
    generate_path_patterns,
    rule_to_path_pattern,
    schema_entity_to_pybids,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def static_bids_config():
    config_path = Path(__file__).parent.parent / "config" / "bids.json"
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def static_deriv_config():
    config_path = Path(__file__).parent.parent / "config" / "derivatives.json"
    with open(config_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Schema-derived helpers tests
# ---------------------------------------------------------------------------


class TestGetFormatPatterns:
    """Tests for _get_format_patterns()."""

    def test_has_label_and_index(self, schema):
        patterns = _get_format_patterns(schema)
        assert "label" in patterns
        assert "index" in patterns

    def test_label_pattern_is_capture_group(self, schema):
        patterns = _get_format_patterns(schema)
        assert patterns["label"].startswith("(")
        assert patterns["label"].endswith(")")

    def test_index_pattern_matches_digits(self, schema):
        patterns = _get_format_patterns(schema)
        assert re.match(patterns["index"], "123")


class TestGetDirectoryEntities:
    """Tests for _get_directory_entities()."""

    def test_has_subject_and_session(self, schema):
        dirs = _get_directory_entities(schema)
        assert "subject" in dirs
        assert "session" in dirs

    def test_subject_template(self, schema):
        dirs = _get_directory_entities(schema)
        assert dirs["subject"] == "{subject}"

    def test_session_template(self, schema):
        dirs = _get_directory_entities(schema)
        assert dirs["session"] == "{subject}{session}"


# ---------------------------------------------------------------------------
# Entity conversion tests
# ---------------------------------------------------------------------------


class TestSchemaEntityToPybids:
    """Tests for schema_entity_to_pybids()."""

    def test_label_format(self, schema):
        """Label-format entities produce the correct regex pattern."""
        fmt_patterns = _get_format_patterns(schema)
        dir_entities = _get_directory_entities(schema)
        ent = schema_entity_to_pybids(
            "acquisition",
            schema["objects"]["entities"]["acquisition"],
            fmt_patterns, dir_entities,
        )
        assert ent["name"] == "acquisition"
        assert "acq-" in ent["pattern"]
        assert "dtype" not in ent

    def test_index_format(self, schema):
        """Index-format entities get dtype='int'."""
        fmt_patterns = _get_format_patterns(schema)
        dir_entities = _get_directory_entities(schema)
        ent = schema_entity_to_pybids(
            "run",
            schema["objects"]["entities"]["run"],
            fmt_patterns, dir_entities,
        )
        assert ent["name"] == "run"
        assert "run-" in ent["pattern"]
        assert ent["dtype"] == "int"

    def test_enum_entity(self, schema):
        """Enum-constrained entities put values directly in the pattern."""
        fmt_patterns = _get_format_patterns(schema)
        dir_entities = _get_directory_entities(schema)
        ent = schema_entity_to_pybids(
            "mtransfer",
            schema["objects"]["entities"]["mtransfer"],
            fmt_patterns, dir_entities,
        )
        assert ent["name"] == "mtransfer"
        assert "mt-" in ent["pattern"]
        assert "(on|off)" in ent["pattern"]

    def test_hemi_entity(self, schema):
        """Hemisphere entity has L|R enum."""
        fmt_patterns = _get_format_patterns(schema)
        dir_entities = _get_directory_entities(schema)
        ent = schema_entity_to_pybids(
            "hemisphere",
            schema["objects"]["entities"]["hemisphere"],
            fmt_patterns, dir_entities,
        )
        assert ent["name"] == "hemisphere"
        assert "hemi-" in ent["pattern"]
        assert "(L|R)" in ent["pattern"]

    def test_subject_directory(self, schema):
        """Subject entity has a directory field."""
        fmt_patterns = _get_format_patterns(schema)
        dir_entities = _get_directory_entities(schema)
        ent = schema_entity_to_pybids(
            "subject",
            schema["objects"]["entities"]["subject"],
            fmt_patterns, dir_entities,
        )
        assert ent["name"] == "subject"
        assert ent["directory"] == "{subject}"
        assert "sub-" in ent["pattern"]

    def test_session_directory_and_mandatory(self, schema):
        """Session entity has directory and mandatory=False."""
        fmt_patterns = _get_format_patterns(schema)
        dir_entities = _get_directory_entities(schema)
        ent = schema_entity_to_pybids(
            "session",
            schema["objects"]["entities"]["session"],
            fmt_patterns, dir_entities,
        )
        assert ent["name"] == "session"
        assert ent["directory"] == "{subject}{session}"
        assert ent["mandatory"] is False


class TestResolveEnum:
    """Tests for _resolve_enum()."""

    def test_no_enum(self):
        assert _resolve_enum({"name": "acq", "format": "label"}) is None

    def test_string_enum(self):
        result = _resolve_enum({"name": "mt", "enum": ["on", "off"]})
        assert result == ["on", "off"]

    def test_dict_enum(self):
        result = _resolve_enum({"name": "x", "enum": [{"name": "a"}, {"name": "b"}]})
        assert result == ["a", "b"]


class TestBuildSuffixEntity:
    """Tests for _build_suffix_entity()."""

    def test_has_known_suffixes(self, schema):
        ent = _build_suffix_entity(schema)
        assert ent["name"] == "suffix"
        for suffix in ["T1w", "T2w", "bold", "dwi"]:
            assert suffix in ent["pattern"]

    def test_pattern_is_regex(self, schema):
        ent = _build_suffix_entity(schema)
        # Should compile without error
        re.compile(ent["pattern"])


class TestBuildExtensionEntity:
    """Tests for _build_extension_entity()."""

    def test_has_known_extensions(self, schema):
        ent = _build_extension_entity(schema)
        assert ent["name"] == "extension"
        for ext in ["\\.nii\\.gz", "\\.nii", "\\.json", "\\.tsv"]:
            assert ext in ent["pattern"]

    def test_pattern_is_regex(self, schema):
        ent = _build_extension_entity(schema)
        re.compile(ent["pattern"])


class TestGenerateEntities:
    """Tests for generate_entities()."""

    def test_entity_order(self, schema):
        """Generated entities follow the canonical order from the schema."""
        entities = generate_entities(schema)
        entity_names = [e["name"] for e in entities]

        # Subject should be first
        assert entity_names[0] == "subject"

        # Session should be near the start
        assert entity_names.index("session") < entity_names.index("task")

        # Description should be last real entity (before pseudo-entities)
        real_entities = [
            e["name"] for e in entities
            if e["name"] not in {"suffix", "datatype", "extension"}
        ]
        assert real_entities[-1] == "description"

    def test_has_pseudo_entities(self, schema):
        """Schema-derived pseudo-entities are included at the end."""
        entities = generate_entities(schema)
        entity_names = [e["name"] for e in entities]

        for pseudo in ["suffix", "datatype", "extension"]:
            assert pseudo in entity_names

        # extension should be last
        assert entity_names[-1] == "extension"

    def test_no_hardcoded_fmap_pseudo_entity(self, schema):
        """The buggy fmap pseudo-entity should NOT be present."""
        entities = generate_entities(schema)
        entity_names = [e["name"] for e in entities]
        assert "fmap" not in entity_names

    def test_no_hardcoded_scans_pseudo_entity(self, schema):
        """The scans pseudo-entity should NOT be present."""
        entities = generate_entities(schema)
        entity_names = [e["name"] for e in entities]
        assert "scans" not in entity_names

    def test_datatype_entity_has_known_types(self, schema):
        """Datatype entity pattern includes known datatypes."""
        entities = generate_entities(schema)
        dt_entity = next(e for e in entities if e["name"] == "datatype")
        for dt in ["anat", "func", "dwi", "fmap"]:
            assert dt in dt_entity["pattern"]

    def test_all_schema_entities_present(self, schema):
        """Every entity in the schema entity order is in the generated list."""
        entities = generate_entities(schema)
        entity_names = {e["name"] for e in entities}
        for key in schema["rules"]["entities"]:
            assert key in entity_names


# ---------------------------------------------------------------------------
# Path pattern generation tests
# ---------------------------------------------------------------------------


class TestFormatEntitySegment:
    """Tests for _format_entity_segment()."""

    def test_optional(self):
        result = _format_entity_segment("acquisition", "acq", "optional")
        assert result == "[_acq-{acquisition}]"

    def test_required(self):
        result = _format_entity_segment("task", "task", "required")
        assert result == "_task-{task}"

    def test_with_enum(self):
        result = _format_entity_segment("hemisphere", "hemi", "optional", ["L", "R"])
        assert result == "[_hemi-{hemisphere<L|R>}]"

    def test_required_with_enum(self):
        result = _format_entity_segment("mtransfer", "mt", "required", ["on", "off"])
        assert result == "_mt-{mtransfer<on|off>}"


class TestChooseDefaultExtension:
    """Tests for _choose_default_extension()."""

    def test_prefers_nii_gz(self):
        assert _choose_default_extension([".nii", ".nii.gz", ".json"]) == ".nii.gz"

    def test_prefers_tsv(self):
        assert _choose_default_extension([".tsv", ".json"]) == ".tsv"

    def test_falls_back_to_non_json(self):
        assert _choose_default_extension([".json", ".csv"]) == ".csv"

    def test_single_extension(self):
        assert _choose_default_extension([".json"]) == ".json"


class TestRuleToPathPattern:
    """Tests for rule_to_path_pattern()."""

    def test_anat_nonparametric(self, schema):
        """Anat nonparametric rule produces a valid pattern."""
        rule = schema["rules"]["files"]["raw"]["anat"]["nonparametric"]
        patterns = rule_to_path_pattern(rule, schema)

        assert len(patterns) >= 1
        main = patterns[0]

        # Should start with directory structure
        assert main.startswith("sub-{subject}[/ses-{session}]/")

        # Should have datatype
        assert "{datatype<anat>|anat}" in main

        # Should have suffix
        assert "T1w" in main

        # Should have extension
        assert ".nii.gz" in main

    def test_func_rule(self, schema):
        """Func rule has required task entity."""
        rule = schema["rules"]["files"]["raw"]["func"]["func"]
        patterns = rule_to_path_pattern(rule, schema)

        main = patterns[0]
        # task should be required (no brackets)
        assert "_task-{task}" in main
        # bold should be a suffix option
        assert "bold" in main

    def test_sidecar_pattern_generated(self, schema):
        """Rules with imaging + json extensions produce sidecar patterns."""
        rule = schema["rules"]["files"]["raw"]["anat"]["nonparametric"]
        patterns = rule_to_path_pattern(rule, schema)

        # Should have main + sidecar
        assert len(patterns) == 2

        sidecar = patterns[1]
        # Sidecar should not start with sub-{subject}
        assert not sidecar.startswith("sub-")
        # Should contain .json
        assert ".json" in sidecar

    def test_no_sidecar_split(self, schema):
        """sidecar_split=False puts all extensions in one pattern."""
        rule = schema["rules"]["files"]["raw"]["anat"]["nonparametric"]
        patterns = rule_to_path_pattern(rule, schema, sidecar_split=False)

        assert len(patterns) == 1
        pattern = patterns[0]
        # Main pattern should contain both imaging and heritable extensions
        assert ".nii.gz" in pattern
        assert ".json" in pattern
        # Should still be a full path pattern
        assert pattern.startswith("sub-{subject}")

    def test_no_sidecar_split_build_path_json(self, schema):
        """sidecar_split=False patterns can build .json paths."""
        rule = schema["rules"]["files"]["raw"]["anat"]["nonparametric"]
        patterns = rule_to_path_pattern(rule, schema, sidecar_split=False)

        result = build_path(
            {"subject": "01", "datatype": "anat", "suffix": "T1w",
             "extension": ".json"},
            patterns,
        )
        assert result is not None
        assert result.endswith(".json")
        assert "sub-01" in result
        assert "anat" in result

    def test_pattern_usable_with_build_path(self, schema):
        """Generated patterns work with pybids build_path()."""
        rule = schema["rules"]["files"]["raw"]["anat"]["nonparametric"]
        patterns = rule_to_path_pattern(rule, schema)

        entities = {
            "subject": "01",
            "datatype": "anat",
            "suffix": "T1w",
            "extension": ".nii.gz",
        }
        result = build_path(entities, patterns)
        assert result is not None
        assert "sub-01" in result
        assert "anat" in result
        assert "T1w" in result
        assert result.endswith(".nii.gz")

    def test_func_pattern_with_build_path(self, schema):
        """Func pattern builds correctly with required task entity."""
        rule = schema["rules"]["files"]["raw"]["func"]["func"]
        patterns = rule_to_path_pattern(rule, schema)

        entities = {
            "subject": "01",
            "task": "rest",
            "datatype": "func",
            "suffix": "bold",
            "extension": ".nii.gz",
        }
        result = build_path(entities, patterns)
        assert result is not None
        assert "sub-01" in result
        assert "task-rest" in result
        assert "bold" in result


class TestGeneratePathPatterns:
    """Tests for generate_path_patterns()."""

    def test_raw_patterns_not_empty(self, schema):
        patterns = generate_path_patterns(schema, "raw")
        assert len(patterns) > 0

    def test_deriv_patterns_not_empty(self, schema):
        patterns = generate_path_patterns(schema, "deriv")
        assert len(patterns) > 0

    def test_patterns_are_strings(self, schema):
        patterns = generate_path_patterns(schema, "raw")
        for p in patterns:
            assert isinstance(p, str)

    def test_no_sidecar_split_all_full_path(self, schema):
        """sidecar_split=False produces only full-path patterns."""
        patterns = generate_path_patterns(schema, "raw", sidecar_split=False)
        for p in patterns:
            assert p.startswith("sub-{subject}"), (
                f"Expected full-path pattern, got sidecar: {p}"
            )


# ---------------------------------------------------------------------------
# Config comparison tests
# ---------------------------------------------------------------------------


class TestConfigComparison:
    """Compare generated entities/patterns against static JSON files."""

    def test_static_entity_names_covered(self, schema, static_bids_config):
        """All entity names from static bids.json appear in generated entities.

        The static bids.json uses some short names (proc, mt, inv, staining)
        while the schema-generated config uses the canonical long names
        (processing, mtransfer, inversion, stain). We map these for comparison.
        """
        # Known naming differences: static short name -> schema long name
        name_map = {
            "proc": "processing",
            "mt": "mtransfer",
            "inv": "inversion",
            "staining": "stain",
        }
        entities = generate_entities(schema)
        generated_names = {e["name"] for e in entities}
        static_names = {e["name"] for e in static_bids_config["entities"]}

        # Map static names to their schema equivalents
        mapped_static = {name_map.get(n, n) for n in static_names}

        # Pseudo-entities that were removed (fmap, scans) are expected to be missing
        removed_pseudos = {"fmap", "scans"}
        missing = mapped_static - generated_names - removed_pseudos
        assert not missing, f"Missing entities: {missing}"

    def test_generated_stain_entity_consistent(self, schema):
        """Generated config uses 'stain' consistently (unlike static bids.json)."""
        entities = generate_entities(schema)
        entity_names = {e["name"] for e in entities}
        assert "stain" in entity_names
        assert "staining" not in entity_names

    def test_round_trip_anat_t1w(self, schema, static_bids_config):
        """Building a T1w path gives comparable results."""
        entities = {
            "subject": "01",
            "session": "pre",
            "datatype": "anat",
            "suffix": "T1w",
            "extension": ".nii.gz",
        }
        gen_patterns = generate_path_patterns(schema, "raw")
        gen_result = build_path(entities, gen_patterns)
        static_result = build_path(
            entities, static_bids_config["default_path_patterns"]
        )
        assert gen_result is not None
        assert static_result is not None
        assert "sub-01" in gen_result
        assert "ses-pre" in gen_result
        assert "anat" in gen_result
        assert "T1w" in gen_result

    def test_round_trip_func_bold(self, schema, static_bids_config):
        """Building a BOLD path gives comparable results."""
        entities = {
            "subject": "02",
            "task": "rest",
            "run": 1,
            "datatype": "func",
            "suffix": "bold",
            "extension": ".nii.gz",
        }
        gen_patterns = generate_path_patterns(schema, "raw")
        gen_result = build_path(entities, gen_patterns)
        static_result = build_path(
            entities, static_bids_config["default_path_patterns"]
        )
        assert gen_result is not None
        assert static_result is not None
        assert "sub-02" in gen_result
        assert "task-rest" in gen_result
        assert "bold" in gen_result

    def test_round_trip_dwi(self, schema, static_bids_config):
        """Building a DWI path gives comparable results."""
        entities = {
            "subject": "03",
            "datatype": "dwi",
            "suffix": "dwi",
            "extension": ".nii.gz",
        }
        gen_patterns = generate_path_patterns(schema, "raw")
        gen_result = build_path(entities, gen_patterns)
        static_result = build_path(
            entities, static_bids_config["default_path_patterns"]
        )
        assert gen_result is not None
        assert static_result is not None
        assert "sub-03" in gen_result
        assert "dwi" in gen_result


# ---------------------------------------------------------------------------
# Standalone bids_path() tests
# ---------------------------------------------------------------------------


class TestBidsPath:
    """Tests for bids_path() convenience function."""

    def test_basic_anat(self):
        """Build a basic anatomical path."""
        result = bids_path({
            "subject": "01",
            "datatype": "anat",
            "suffix": "T1w",
            "extension": ".nii.gz",
        })
        assert result is not None
        assert result == "sub-01/anat/sub-01_T1w.nii.gz"

    def test_with_session(self):
        """Build a path with session."""
        result = bids_path({
            "subject": "01",
            "session": "pre",
            "datatype": "anat",
            "suffix": "T1w",
            "extension": ".nii.gz",
        })
        assert result is not None
        assert "ses-pre" in result

    def test_func_bold(self):
        """Build a functional BOLD path."""
        result = bids_path({
            "subject": "02",
            "task": "rest",
            "run": 1,
            "datatype": "func",
            "suffix": "bold",
            "extension": ".nii.gz",
        })
        assert result is not None
        assert "task-rest" in result
        assert "run-1" in result
        assert "bold" in result

    def test_with_custom_patterns(self):
        """Custom patterns override schema-generated ones."""
        result = bids_path(
            {"subject": "01", "suffix": "T1w", "extension": ".nii.gz"},
            patterns=["sub-{subject}_{suffix}{extension}"],
        )
        assert result == "sub-01_T1w.nii.gz"

    def test_no_match_returns_none(self):
        """Returns None when no pattern matches."""
        result = bids_path(
            {"subject": "01", "suffix": "nonexistent_suffix_xyz"},
            patterns=["sub-{subject}_{suffix<T1w>}{extension}"],
        )
        assert result is None


# ---------------------------------------------------------------------------
# Dict-based config paths tests
# ---------------------------------------------------------------------------


class TestDictConfigPaths:
    """Tests for passing in-memory config dicts to add_config_paths()."""

    def test_add_config_paths_accepts_dict(self):
        """add_config_paths() accepts a dict value without requiring a file."""
        import bids.config as cf
        from bids.layout.utils import add_config_paths

        config_dict = {
            "name": "_test_dict_config",
            "entities": [
                {"name": "subject", "pattern": "[/\\\\]+sub-([a-zA-Z0-9+]+)"},
            ],
            "default_path_patterns": [
                "sub-{subject}_{suffix}{extension}",
            ],
        }
        try:
            add_config_paths(_test_dict_config=config_dict)
            paths = cf.get_option("config_paths")
            assert "_test_dict_config" in paths
            assert paths["_test_dict_config"] is config_dict
        finally:
            # Clean up: remove the test config
            current = cf.get_option("config_paths")
            current.pop("_test_dict_config", None)
            cf.set_option("config_paths", current)

    def test_config_load_resolves_dict_from_config_paths(self):
        """Config.load() resolves a dict stored in config_paths by name."""
        import bids.config as cf
        from bids.layout.models import Config
        from bids.layout.utils import add_config_paths

        config_dict = {
            "name": "_test_load_dict",
            "entities": [
                {"name": "subject", "pattern": "[/\\\\]+sub-([a-zA-Z0-9+]+)"},
            ],
            "default_path_patterns": [],
        }
        try:
            add_config_paths(_test_load_dict=config_dict)
            loaded = Config.load("_test_load_dict")
            assert loaded.name == "_test_load_dict"
        finally:
            current = cf.get_option("config_paths")
            current.pop("_test_load_dict", None)
            cf.set_option("config_paths", current)
