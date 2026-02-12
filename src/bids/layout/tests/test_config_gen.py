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
    ConfigExtension,
    _build_datatype_entity,
    _choose_default_extension,
    _format_entity_segment,
    _resolve_enum,
    apply_extension,
    bids_path,
    generate_config,
    generate_entities,
    generate_extended_config,
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
def bids_config(schema):
    return generate_config("bids", schema=schema, rule_groups=["raw"])


@pytest.fixture(scope="module")
def deriv_config(schema):
    return generate_config("derivatives", schema=schema, rule_groups=["raw", "deriv"])


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
# Entity conversion tests
# ---------------------------------------------------------------------------


class TestSchemaEntityToPybids:
    """Tests for schema_entity_to_pybids()."""

    def test_label_format(self, schema):
        """Label-format entities produce the correct regex pattern."""
        ent = schema_entity_to_pybids(
            "acquisition",
            schema["objects"]["entities"]["acquisition"],
            schema["objects"]["formats"],
        )
        assert ent["name"] == "acquisition"
        assert "acq-" in ent["pattern"]
        assert "([a-zA-Z0-9+]+)" in ent["pattern"]
        assert "dtype" not in ent

    def test_index_format(self, schema):
        """Index-format entities get dtype='int'."""
        ent = schema_entity_to_pybids(
            "run",
            schema["objects"]["entities"]["run"],
            schema["objects"]["formats"],
        )
        assert ent["name"] == "run"
        assert "run-" in ent["pattern"]
        assert "(\\d+)" in ent["pattern"]
        assert ent["dtype"] == "int"

    def test_enum_entity(self, schema):
        """Enum-constrained entities put values directly in the pattern."""
        ent = schema_entity_to_pybids(
            "mtransfer",
            schema["objects"]["entities"]["mtransfer"],
            schema["objects"]["formats"],
        )
        assert ent["name"] == "mtransfer"
        assert "mt-" in ent["pattern"]
        assert "(on|off)" in ent["pattern"]

    def test_hemi_entity(self, schema):
        """Hemisphere entity has L|R enum."""
        ent = schema_entity_to_pybids(
            "hemisphere",
            schema["objects"]["entities"]["hemisphere"],
            schema["objects"]["formats"],
        )
        assert ent["name"] == "hemisphere"
        assert "hemi-" in ent["pattern"]
        assert "(L|R)" in ent["pattern"]

    def test_subject_directory(self, schema):
        """Subject entity has a directory field."""
        ent = schema_entity_to_pybids(
            "subject",
            schema["objects"]["entities"]["subject"],
            schema["objects"]["formats"],
        )
        assert ent["name"] == "subject"
        assert ent["directory"] == "{subject}"
        assert "sub-" in ent["pattern"]

    def test_session_directory_and_mandatory(self, schema):
        """Session entity has directory and mandatory=False."""
        ent = schema_entity_to_pybids(
            "session",
            schema["objects"]["entities"]["session"],
            schema["objects"]["formats"],
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
            if e["name"] not in {"suffix", "scans", "fmap", "datatype", "extension"}
        ]
        assert real_entities[-1] == "description"

    def test_has_pseudo_entities(self, schema):
        """Pseudo-entities are included at the end."""
        entities = generate_entities(schema)
        entity_names = [e["name"] for e in entities]

        for pseudo in ["suffix", "scans", "fmap", "datatype", "extension"]:
            assert pseudo in entity_names

        # extension should be last
        assert entity_names[-1] == "extension"

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


# ---------------------------------------------------------------------------
# Config generation tests
# ---------------------------------------------------------------------------


class TestGenerateConfig:
    """Tests for generate_config()."""

    def test_config_structure(self, bids_config):
        """Generated config has the expected top-level keys."""
        assert "name" in bids_config
        assert "entities" in bids_config
        assert "default_path_patterns" in bids_config
        assert bids_config["name"] == "bids"

    def test_entities_not_empty(self, bids_config):
        assert len(bids_config["entities"]) > 0

    def test_patterns_not_empty(self, bids_config):
        assert len(bids_config["default_path_patterns"]) > 0

    def test_core_entities_present(self, bids_config):
        """Core entities from the static bids.json are present."""
        entity_names = {e["name"] for e in bids_config["entities"]}
        for name in ["subject", "session", "task", "acquisition", "run",
                      "suffix", "extension", "datatype"]:
            assert name in entity_names

    def test_deriv_config_has_deriv_entities(self, deriv_config):
        """Derivatives config includes derivative-specific entities."""
        entity_names = {e["name"] for e in deriv_config["entities"]}
        # These come from the schema entity order
        for name in ["space", "description", "resolution", "density"]:
            assert name in entity_names


class TestStaticConfigConsistency:
    """Validate internal consistency of the static bids.json."""

    def test_entity_pattern_name_agreement(self, static_bids_config):
        """Path patterns should only reference entity names that exist in the
        entity definitions. Every {name} in a pattern must correspond to a
        defined entity name.
        """
        entity_names = {e["name"] for e in static_bids_config["entities"]}
        # Also include pseudo-entity-like names that appear in patterns
        entity_names.update({"subject", "session", "datatype", "suffix", "extension"})

        pattern_re = re.compile(r'\{(\w+)(?:<[^>]+>)?(?:\|[^}]*)?\}')

        for pattern in static_bids_config["default_path_patterns"]:
            referenced = set(pattern_re.findall(pattern))
            undefined = referenced - entity_names
            assert not undefined, (
                f"Pattern references undefined entity names {undefined}: "
                f"{pattern}"
            )


class TestConfigComparison:
    """Compare generated configs against static JSON files."""

    def test_static_entity_names_covered(self, bids_config, static_bids_config):
        """All entity names from static bids.json appear in generated config.

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
        generated_names = {e["name"] for e in bids_config["entities"]}
        static_names = {e["name"] for e in static_bids_config["entities"]}

        # Map static names to their schema equivalents
        mapped_static = {name_map.get(n, n) for n in static_names}

        missing = mapped_static - generated_names
        assert not missing, f"Missing entities: {missing}"

    def test_generated_stain_entity_consistent(self, bids_config):
        """Generated config uses 'stain' consistently (unlike static bids.json).

        The schema top-level key is 'stain', so all generated patterns should
        reference {stain}, not {staining}.
        """
        entity_names = {e["name"] for e in bids_config["entities"]}
        assert "stain" in entity_names
        assert "staining" not in entity_names

        # All patterns that reference stain should use {stain}, never {staining}
        for pattern in bids_config["default_path_patterns"]:
            assert "{staining}" not in pattern, (
                f"Generated pattern incorrectly uses '{{staining}}': {pattern}"
            )

    def test_generated_micr_sidecar_build_path_works(self, bids_config):
        """Generated config correctly builds microscopy sidecar paths with stain.

        This is the counterpart to TestStaticConfigConsistency's test — the
        generated config should NOT have the staining/stain mismatch bug.
        """
        entities = {
            "sample": "A",
            "stain": "LFB",
            "suffix": "TEM",
            "extension": ".json",
        }

        # Find sidecar patterns for microscopy
        sidecar_patterns = [
            p for p in bids_config["default_path_patterns"]
            if "sample-{sample}" in p
            and not p.startswith("sub-")
            and "{stain" in p
        ]
        if sidecar_patterns:
            result = build_path(entities, sidecar_patterns)
            assert result is not None, (
                "Generated sidecar pattern should work with 'stain' entity"
            )
            assert "stain-LFB" in result

    def test_round_trip_anat_t1w(self, bids_config, static_bids_config):
        """Building a T1w path gives comparable results."""
        entities = {
            "subject": "01",
            "session": "pre",
            "datatype": "anat",
            "suffix": "T1w",
            "extension": ".nii.gz",
        }
        gen_result = build_path(
            entities, bids_config["default_path_patterns"]
        )
        static_result = build_path(
            entities, static_bids_config["default_path_patterns"]
        )
        assert gen_result is not None
        assert static_result is not None
        # Both should produce a path containing the same key elements
        assert "sub-01" in gen_result
        assert "ses-pre" in gen_result
        assert "anat" in gen_result
        assert "T1w" in gen_result

    def test_round_trip_func_bold(self, bids_config, static_bids_config):
        """Building a BOLD path gives comparable results."""
        entities = {
            "subject": "02",
            "task": "rest",
            "run": 1,
            "datatype": "func",
            "suffix": "bold",
            "extension": ".nii.gz",
        }
        gen_result = build_path(
            entities, bids_config["default_path_patterns"]
        )
        static_result = build_path(
            entities, static_bids_config["default_path_patterns"]
        )
        assert gen_result is not None
        assert static_result is not None
        assert "sub-02" in gen_result
        assert "task-rest" in gen_result
        assert "bold" in gen_result

    def test_round_trip_dwi(self, bids_config, static_bids_config):
        """Building a DWI path gives comparable results."""
        entities = {
            "subject": "03",
            "datatype": "dwi",
            "suffix": "dwi",
            "extension": ".nii.gz",
        }
        gen_result = build_path(
            entities, bids_config["default_path_patterns"]
        )
        static_result = build_path(
            entities, static_bids_config["default_path_patterns"]
        )
        assert gen_result is not None
        assert static_result is not None
        assert "sub-03" in gen_result
        assert "dwi" in gen_result


# ---------------------------------------------------------------------------
# Extension mechanism tests
# ---------------------------------------------------------------------------


class TestConfigExtension:
    """Tests for the ConfigExtension and apply_extension()."""

    def test_extra_entity_at_end(self, bids_config):
        """Extra entities with position='end' go before pseudo-entities."""
        ext = ConfigExtension(
            name="test",
            extra_entities=[
                {"name": "hash", "pattern": "hash-([a-zA-Z0-9+]+)"},
            ],
        )
        result = apply_extension(bids_config, ext)
        names = [e["name"] for e in result["entities"]]

        assert "hash" in names
        # hash should be before suffix
        assert names.index("hash") < names.index("suffix")

    def test_extra_entity_after(self, bids_config):
        """Extra entity with position='after:session' appears after session."""
        ext = ConfigExtension(
            name="test",
            extra_entities=[
                {
                    "name": "hash",
                    "pattern": "hash-([a-zA-Z0-9+]+)",
                    "position": "after:session",
                },
            ],
        )
        result = apply_extension(bids_config, ext)
        names = [e["name"] for e in result["entities"]]

        assert names.index("hash") == names.index("session") + 1

    def test_extra_entity_before(self, bids_config):
        """Extra entity with position='before:task' appears before task."""
        ext = ConfigExtension(
            name="test",
            extra_entities=[
                {
                    "name": "mything",
                    "pattern": "mything-([a-zA-Z0-9+]+)",
                    "position": "before:task",
                },
            ],
        )
        result = apply_extension(bids_config, ext)
        names = [e["name"] for e in result["entities"]]

        assert names.index("mything") == names.index("task") - 1

    def test_entity_overrides(self, bids_config):
        """Entity overrides modify existing entity dicts."""
        ext = ConfigExtension(
            name="test",
            entity_overrides={
                "run": {"dtype": "str"},  # Override run from int to str
            },
        )
        result = apply_extension(bids_config, ext)
        run_ent = next(e for e in result["entities"] if e["name"] == "run")
        assert run_ent["dtype"] == "str"

    def test_extra_datatypes(self, bids_config):
        """Extra datatypes are added to the datatype entity pattern."""
        ext = ConfigExtension(
            name="test",
            extra_datatypes=["figures"],
        )
        result = apply_extension(bids_config, ext)
        dt_entity = next(e for e in result["entities"] if e["name"] == "datatype")
        assert "figures" in dt_entity["pattern"]

    def test_extra_path_patterns(self, bids_config):
        """Extra path patterns are appended."""
        ext = ConfigExtension(
            name="test",
            extra_path_patterns=[
                "sub-{subject}/{datatype<figures>}/sub-{subject}_{suffix}{extension}",
            ],
        )
        result = apply_extension(bids_config, ext)
        assert result["default_path_patterns"][-1].startswith("sub-{subject}/{datatype<figures>}")

    def test_does_not_mutate_input(self, bids_config):
        """apply_extension does not modify the input config."""
        original_len = len(bids_config["entities"])
        ext = ConfigExtension(
            name="test",
            extra_entities=[
                {"name": "hash", "pattern": "hash-([a-zA-Z0-9+]+)"},
            ],
        )
        apply_extension(bids_config, ext)
        assert len(bids_config["entities"]) == original_len


class TestGenerateExtendedConfig:
    """Tests for generate_extended_config()."""

    def test_with_nipreps_like_extension(self, schema):
        """A NiPreps-like extension produces a valid config."""
        ext = ConfigExtension(
            name="nipreps",
            extra_entities=[
                {
                    "name": "hash",
                    "pattern": "hash-([a-zA-Z0-9+]+)",
                    "position": "after:session",
                },
                {
                    "name": "fmapid",
                    "pattern": "[_/\\\\]+fmapid-([a-zA-Z0-9+]+)",
                    "position": "after:label",
                },
            ],
            extra_datatypes=["figures"],
            extra_path_patterns=[
                "sub-{subject}/{datatype<figures>}/sub-{subject}"
                "[_ses-{session}][_desc-{description}]"
                "_{suffix<T1w|T2w>}{extension<.html|.svg>|.svg}",
            ],
        )

        config = generate_extended_config(
            name="nipreps",
            schema=schema,
            extensions=[ext],
            rule_groups=["raw", "deriv"],
        )

        entity_names = {e["name"] for e in config["entities"]}
        assert "hash" in entity_names
        assert "fmapid" in entity_names

        # figures should be in datatype pattern
        dt_entity = next(e for e in config["entities"] if e["name"] == "datatype")
        assert "figures" in dt_entity["pattern"]

        # The extra pattern should be present
        assert any("figures" in p for p in config["default_path_patterns"])

    def test_build_path_with_extended_config(self, schema):
        """build_path works with an extended config for derivative files."""
        ext = ConfigExtension(
            name="nipreps",
            extra_path_patterns=[
                "sub-{subject}[/ses-{session}]/{datatype<anat>|anat}/"
                "sub-{subject}[_ses-{session}][_desc-{description}]"
                "_{suffix<boldref>}{extension<.nii|.nii.gz>|.nii.gz}",
            ],
        )
        config = generate_extended_config(
            name="nipreps",
            schema=schema,
            extensions=[ext],
            rule_groups=["raw", "deriv"],
        )

        entities = {
            "subject": "01",
            "description": "preproc",
            "datatype": "anat",
            "suffix": "boldref",
            "extension": ".nii.gz",
        }
        result = build_path(entities, config["default_path_patterns"])
        assert result is not None
        assert "sub-01" in result
        assert "desc-preproc" in result
        assert "boldref" in result


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
