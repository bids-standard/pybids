"""Generate pybids config dicts from the BIDS schema.

This module converts the structured BIDS schema (loaded via bidsschematools)
into pybids-format configuration dictionaries compatible with
``bids.json`` and ``derivatives.json``.

It also provides an extension mechanism for downstream tools (such as
NiPreps/niworkflows) to layer custom entities and path patterns on top
of the schema-generated config, and a standalone ``bids_path()``
convenience function.

Requires the optional ``bidsschematools`` package::

    pip install pybids[schema]
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from .writing import build_path

try:
    from bidsschematools.schema import load_schema

    HAS_SCHEMA = True
except ImportError:
    HAS_SCHEMA = False


def _require_schema():
    if not HAS_SCHEMA:
        raise ImportError(
            "bidsschematools is required for schema-driven config generation. "
            "Install it with: pip install pybids[schema]"
        )


# ---------------------------------------------------------------------------
# Entity conversion
# ---------------------------------------------------------------------------

# Entities that define directory structure
_DIRECTORY_ENTITIES = {
    "subject": "{subject}",
    "session": "{subject}{session}",
}

# Format → regex capture group mapping
_FORMAT_PATTERN = {
    "label": "([a-zA-Z0-9+]+)",
    "index": "(\\d+)",
}

# Index-format entities get dtype: int
_INDEX_DTYPE = "int"

# Pseudo-entities appended to every config (not from the schema entity list)
_PSEUDO_ENTITIES = [
    {
        "name": "suffix",
        "pattern": "(?:^|[_/\\\\])([a-zA-Z0-9+]+)\\.[^/\\\\]+$",
    },
    {
        "name": "scans",
        "pattern": "(.*\\_scans.tsv)$",
    },
    {
        "name": "fmap",
        "pattern": "(phasediff|magnitude[1-2]|phase[1-2]|fieldmap|epi)\\.nii",
    },
    # datatype is built dynamically
    {
        "name": "extension",
        "pattern": "[^./\\\\](\\.[^/\\\\]+)$",
    },
]


def schema_entity_to_pybids(
    entity_key: str,
    entity_def: dict,
    format_patterns: dict,
) -> dict:
    """Convert a single BIDS schema entity definition to a pybids entity dict.

    Parameters
    ----------
    entity_key : str
        The long entity name as it appears in the schema (e.g., ``"acquisition"``).
    entity_def : dict
        The entity definition from ``schema.objects.entities[entity_key]``.
    format_patterns : dict
        Format patterns from ``schema.objects.formats``.

    Returns
    -------
    dict
        A pybids entity dict with keys: ``name``, ``pattern``, and optionally
        ``dtype``, ``directory``, ``mandatory``.
    """
    short_name = entity_def["name"]
    fmt = entity_def.get("format", "label")

    # Build regex capture group
    enum_values = _resolve_enum(entity_def)
    if enum_values:
        capture = "(%s)" % "|".join(enum_values)
    else:
        capture = _FORMAT_PATTERN.get(fmt, _FORMAT_PATTERN["label"])

    # Build prefix: directory entities use path separator, others use _ or separator
    if entity_key == "subject":
        prefix = "[/\\\\]+"
    else:
        prefix = "[_/\\\\]+"

    pattern = "%s%s-%s" % (prefix, short_name, capture)

    result = {"name": entity_key, "pattern": pattern}

    # Directory entities
    if entity_key in _DIRECTORY_ENTITIES:
        result["directory"] = _DIRECTORY_ENTITIES[entity_key]

    # Session is not mandatory
    if entity_key == "session":
        result["mandatory"] = False

    # Index-format entities get int dtype
    if fmt == "index":
        result["dtype"] = _INDEX_DTYPE

    return result


def _resolve_enum(entity_def) -> list[str] | None:
    """Extract enum values from an entity definition, if present."""
    if "enum" not in entity_def:
        return None
    values = []
    for v in entity_def["enum"]:
        if isinstance(v, str):
            values.append(v)
        elif isinstance(v, Mapping) and "name" in v:
            values.append(v["name"])
    return values or None


def _build_datatype_entity(datatypes: set[str]) -> dict:
    """Build the pseudo-entity dict for the ``datatype`` entity."""
    dt_pattern = "|".join(sorted(datatypes))
    return {
        "name": "datatype",
        "pattern": "[/\\\\]+(%s)[/\\\\]+" % dt_pattern,
    }


def generate_entities(schema) -> list[dict]:
    """Generate the full ordered list of pybids entity dicts from the schema.

    Parameters
    ----------
    schema
        A loaded, dereferenced BIDS schema (from ``bidsschematools.schema.load_schema()``).

    Returns
    -------
    list of dict
        Ordered list of pybids entity dicts, including pseudo-entities at the end.
    """
    entity_order = list(schema["rules"]["entities"])
    entity_defs = schema["objects"]["entities"]
    format_defs = schema["objects"]["formats"]

    entities = []
    for key in entity_order:
        if key not in entity_defs:
            continue
        entities.append(schema_entity_to_pybids(key, entity_defs[key], format_defs))

    # Collect all datatypes from rules for the datatype pseudo-entity
    datatypes = _collect_datatypes(schema)

    # Append pseudo-entities
    for pseudo in _PSEUDO_ENTITIES:
        entities.append(dict(pseudo))

    # Insert datatype entity before extension (second-to-last)
    dt_entity = _build_datatype_entity(datatypes)
    entities.insert(-1, dt_entity)

    return entities


def _collect_datatypes(schema) -> set[str]:
    """Collect all datatype values referenced in file rules."""
    datatypes = set()
    for rule_group_key in ("raw", "deriv"):
        rule_group = schema["rules"]["files"].get(rule_group_key, {})
        _collect_datatypes_recursive(rule_group, datatypes)
    return datatypes


def _collect_datatypes_recursive(obj, datatypes: set):
    """Recursively walk rule groups to find datatypes."""
    if isinstance(obj, Mapping):
        if "datatypes" in obj and isinstance(obj["datatypes"], (list, tuple)):
            for dt in obj["datatypes"]:
                if dt:
                    datatypes.add(dt)
        for v in obj.values():
            if isinstance(v, Mapping):
                _collect_datatypes_recursive(v, datatypes)


# ---------------------------------------------------------------------------
# Path pattern generation
# ---------------------------------------------------------------------------


def _choose_default_extension(extensions: list[str]) -> str:
    """Pick a sensible default extension from a list."""
    preferred = [".nii.gz", ".nii", ".tsv", ".tsv.gz"]
    for ext in preferred:
        if ext in extensions:
            return ext
    # Fall back to first non-.json extension, or first overall
    non_json = [e for e in extensions if e != ".json"]
    return non_json[0] if non_json else extensions[0]


def _format_entity_segment(
    entity_key: str,
    short_name: str,
    level: str,
    enum_values: list[str] | None = None,
) -> str:
    """Format one entity as a pybids path pattern segment.

    Parameters
    ----------
    entity_key : str
        Long entity name (e.g., ``"acquisition"``).
    short_name : str
        Short entity key (e.g., ``"acq"``).
    level : str
        Either ``"required"`` or ``"optional"``.
    enum_values : list of str, optional
        If present, constrains valid values in the pattern.

    Returns
    -------
    str
        Pattern segment like ``"[_acq-{acquisition}]"`` or
        ``"_task-{task}"`` (no brackets for required).
    """
    inner = "%s-{%s" % (short_name, entity_key)
    if enum_values:
        inner += "<%s>" % "|".join(enum_values)
    inner += "}"

    if level == "required":
        return "_" + inner
    else:
        return "[_" + inner + "]"


def rule_to_path_pattern(
    rule: dict,
    schema,
) -> list[str]:
    """Convert a single dereferenced file rule to pybids path pattern(s).

    Parameters
    ----------
    rule : dict
        A dereferenced file rule with keys: ``entities``, ``datatypes``,
        ``suffixes``, ``extensions``.
    schema
        The loaded BIDS schema.

    Returns
    -------
    list of str
        One or more pybids path patterns. The first is the main file pattern;
        subsequent ones are sidecar/inheritance patterns for heritable
        extensions (``.json``, ``.tsv``, etc.).
    """
    entity_order = list(schema["rules"]["entities"])
    entity_defs = schema["objects"]["entities"]

    rule_entities = rule.get("entities", {})
    datatypes = rule.get("datatypes", [])
    suffixes = rule.get("suffixes", [])
    extensions = rule.get("extensions", [])

    if not suffixes or not extensions:
        return []

    # Separate heritable vs main extensions
    heritable_exts = {".tsv", ".json", ".bval", ".bvec"}
    all_exts = set(extensions)
    main_exts = sorted(all_exts - heritable_exts)
    sidecar_exts = sorted(all_exts & heritable_exts)
    has_main = bool(main_exts)

    patterns = []

    # --- Main pattern ---
    exts_for_main = main_exts if has_main else sorted(all_exts)
    patterns.append(
        _build_file_pattern(
            entity_order, entity_defs, rule_entities,
            datatypes, suffixes, exts_for_main,
        )
    )

    # --- Sidecar/inheritance pattern ---
    if sidecar_exts and has_main:
        patterns.append(
            _build_sidecar_pattern(
                entity_order, entity_defs, rule_entities,
                suffixes, sidecar_exts,
            )
        )

    return patterns


def _parse_entity_level(value) -> tuple[str, list[str] | None]:
    """Parse an entity level specification from a file rule.

    Entity values in rules can be:
    - A string like ``"required"`` or ``"optional"``
    - A Mapping with ``"level"`` and optionally ``"enum"`` keys

    Returns
    -------
    tuple of (str, list or None)
        The level string and any enum override.
    """
    if isinstance(value, str):
        return value, None
    if isinstance(value, Mapping):
        level = value.get("level", "optional")
        enum_raw = value.get("enum")
        enum_values = None
        if enum_raw:
            enum_values = []
            for v in enum_raw:
                if isinstance(v, str):
                    enum_values.append(v)
                elif isinstance(v, Mapping) and "name" in v:
                    enum_values.append(v["name"])
        return level, enum_values or None
    return "optional", None


def _build_file_pattern(
    entity_order: list[str],
    entity_defs: dict,
    rule_entities: dict,
    datatypes: list[str],
    suffixes: list[str],
    extensions: list[str],
) -> str:
    """Build the main (full-path) pattern for a file rule."""
    parts = []

    # Directory: sub-{subject}[/ses-{session}]/
    parts.append("sub-{subject}[/ses-{session}]/")

    # Datatype
    if datatypes:
        dt_vals = "|".join(datatypes)
        parts.append("{datatype<%s>|%s}/" % (dt_vals, datatypes[0]))

    # Filename: sub-{subject}[_ses-{session}]
    parts.append("sub-{subject}[_ses-{session}]")

    # Entity segments (in canonical order, skip subject/session)
    for ent_key in entity_order:
        if ent_key in ("subject", "session"):
            continue
        if ent_key not in rule_entities:
            continue

        level, rule_enum = _parse_entity_level(rule_entities[ent_key])

        ent_def = entity_defs.get(ent_key, {})
        short_name = ent_def.get("name", ent_key)

        # Use rule-level enum override if present, else entity-level enum
        enum_values = rule_enum or _resolve_enum(ent_def)

        parts.append(_format_entity_segment(ent_key, short_name, level, enum_values))

    # Suffix
    suffix_vals = "|".join(suffixes)
    parts.append("_{suffix<%s>}" % suffix_vals)

    # Extension
    ext_vals = "|".join(extensions)
    default_ext = _choose_default_extension(extensions)
    parts.append("{extension<%s>|%s}" % (ext_vals, default_ext))

    return "".join(parts)


def _build_sidecar_pattern(
    entity_order: list[str],
    entity_defs: dict,
    rule_entities: dict,
    suffixes: list[str],
    extensions: list[str],
) -> str:
    """Build a sidecar/inheritance pattern (no directory prefix, all optional)."""
    parts = []

    for ent_key in entity_order:
        if ent_key in ("subject", "session"):
            continue
        if ent_key not in rule_entities:
            continue

        ent_def = entity_defs.get(ent_key, {})
        short_name = ent_def.get("name", ent_key)
        enum_values = _resolve_enum(ent_def)

        # All entities are optional in sidecar patterns
        parts.append(_format_entity_segment(ent_key, short_name, "optional", enum_values))

    # Suffix
    suffix_vals = "|".join(suffixes)
    parts.append("_{suffix<%s>}" % suffix_vals)

    # Extension
    ext_vals = "|".join(extensions)
    default_ext = ".json" if ".json" in extensions else extensions[0]
    parts.append("{extension<%s>|%s}" % (ext_vals, default_ext))

    pattern = "".join(parts)

    # Fix leading bracket: first segment is [_key-{val}], but sidecar patterns
    # in pybids use [key-{val}_] style for the first entity. However, the
    # existing pybids convention is inconsistent here, so we use the simpler
    # [_key-{val}] style throughout (the _ is inside the optional bracket).
    # Strip the leading _ from the first optional bracket if present.
    if pattern.startswith("[_"):
        pattern = "[" + pattern[2:]

    return pattern


def generate_path_patterns(schema, rule_group: str = "raw") -> list[str]:
    """Generate all pybids path patterns from a schema rule group.

    Parameters
    ----------
    schema
        The loaded BIDS schema.
    rule_group : str
        Which rule group to process: ``"raw"`` or ``"deriv"``.

    Returns
    -------
    list of str
        List of pybids path patterns.
    """
    patterns = []
    seen = set()

    rule_groups = schema["rules"]["files"].get(rule_group, {})
    _generate_patterns_recursive(rule_groups, schema, patterns, seen)

    return patterns


def _generate_patterns_recursive(obj, schema, patterns, seen):
    """Recursively walk rule groups, generating patterns for leaf rules."""
    if isinstance(obj, Mapping):
        # A leaf rule has entities, suffixes, and extensions
        if "entities" in obj and "suffixes" in obj and "extensions" in obj:
            # Deduplicate: same rule may appear via multiple inheritance paths
            key = _make_rule_key(obj)
            if key not in seen:
                seen.add(key)
                new_patterns = rule_to_path_pattern(obj, schema)
                patterns.extend(new_patterns)
        else:
            for v in obj.values():
                if isinstance(v, Mapping):
                    _generate_patterns_recursive(v, schema, patterns, seen)


def _make_rule_key(rule) -> tuple:
    """Create a hashable dedup key from a rule, handling Namespace values."""
    # Entity values may be strings or Namespace dicts; convert to strings
    entities = rule.get("entities", {})
    ent_items = []
    for k, v in sorted(entities.items()):
        if isinstance(v, str):
            ent_items.append((k, v))
        elif isinstance(v, Mapping):
            ent_items.append((k, str(v.get("level", "optional"))))
        else:
            ent_items.append((k, str(v)))
    return (
        tuple(ent_items),
        tuple(sorted(rule.get("suffixes", []))),
        tuple(sorted(rule.get("extensions", []))),
        tuple(sorted(rule.get("datatypes", []))),
    )


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def generate_config(
    name: str = "bids",
    schema=None,
    schema_path: str | None = None,
    rule_groups: list[str] | None = None,
) -> dict:
    """Generate a complete pybids config dict from the BIDS schema.

    Parameters
    ----------
    name : str
        The config name (e.g., ``"bids"``, ``"derivatives"``).
    schema : optional
        A pre-loaded BIDS schema. If ``None``, loads via ``bidsschematools``.
    schema_path : str, optional
        Path to schema directory. Only used if ``schema`` is ``None``.
    rule_groups : list of str, optional
        Which rule groups to include. Defaults to ``["raw"]`` for ``"bids"``,
        ``["raw", "deriv"]`` for ``"derivatives"`` or any other name.

    Returns
    -------
    dict
        A config dict matching the ``bids.json`` format::

            {"name": ..., "entities": [...], "default_path_patterns": [...]}
    """
    _require_schema()

    if schema is None:
        kwargs = {}
        if schema_path is not None:
            kwargs["schema_dir"] = schema_path
        schema = load_schema(**kwargs)

    if rule_groups is None:
        if name == "bids":
            rule_groups = ["raw"]
        else:
            rule_groups = ["raw", "deriv"]

    # Generate entities
    entities = generate_entities(schema)

    # Generate path patterns
    all_patterns = []
    for rg in rule_groups:
        all_patterns.extend(generate_path_patterns(schema, rg))

    return {
        "name": name,
        "entities": entities,
        "default_path_patterns": all_patterns,
    }


# ---------------------------------------------------------------------------
# Extension mechanism
# ---------------------------------------------------------------------------


class ConfigExtension:
    """Defines custom entities and path patterns to layer on top of a
    schema-generated config.

    Parameters
    ----------
    name : str
        A name for this extension (e.g., ``"nipreps"``).
    extra_entities : list of dict, optional
        Additional entity dicts to add. Each must have at minimum
        ``"name"`` and ``"pattern"``. An optional ``"position"`` key
        controls insertion: ``"after:<entity_name>"`` inserts after the
        named entity, ``"before:<entity_name>"`` inserts before it, and
        ``"end"`` (default) appends before pseudo-entities.
    extra_path_patterns : list of str, optional
        Additional path patterns to append.
    entity_overrides : dict, optional
        A dict mapping entity long names to partial dicts that override
        fields in the generated entity.
    extra_datatypes : list of str, optional
        Additional datatype values to recognize in the ``datatype`` entity
        pattern (e.g., ``["figures"]``).
    inject_entity_segments : list of dict, optional
        Entity segments to inject into all generated path patterns.
        Each dict must have ``"segment"`` (the string to insert, e.g.,
        ``"[_hash-{hash}]"``) and ``"after"`` (the substring to insert
        after, e.g., ``"[_ses-{session}]"``). Patterns that do not
        contain the ``"after"`` string are left unchanged. Injection is
        skipped if the segment is already present (deduplication).
    """

    def __init__(
        self,
        name: str,
        extra_entities: list[dict] | None = None,
        extra_path_patterns: list[str] | None = None,
        entity_overrides: dict[str, dict] | None = None,
        extra_datatypes: list[str] | None = None,
        inject_entity_segments: list[dict] | None = None,
    ):
        self.name = name
        self.extra_entities = extra_entities or []
        self.extra_path_patterns = extra_path_patterns or []
        self.entity_overrides = entity_overrides or {}
        self.extra_datatypes = extra_datatypes or []
        self.inject_entity_segments = inject_entity_segments or []


def _rename_template_vars(
    patterns: list[str], old_name: str, new_name: str,
) -> list[str]:
    """Rename template variables in path patterns.

    Rewrites ``{old_name}``, ``{old_name<...>}``, ``{old_name|...}`` to use
    *new_name* instead.
    """
    regex = re.compile(r"\{" + re.escape(old_name) + r"(?=[<|}\s])")
    return [regex.sub("{" + new_name, p) for p in patterns]


def apply_extension(config: dict, extension: ConfigExtension) -> dict:
    """Apply a ConfigExtension to a generated config dict.

    Parameters
    ----------
    config : dict
        A config dict as returned by :func:`generate_config`.
    extension : ConfigExtension
        The extension to apply.

    Returns
    -------
    dict
        A new config dict with the extension applied (does not
        mutate the input).
    """
    config = copy.deepcopy(config)

    # 1. Apply entity overrides
    for ent in config["entities"]:
        if ent["name"] in extension.entity_overrides:
            ent.update(extension.entity_overrides[ent["name"]])

    # 2. Rename template variables in patterns to match overridden names
    for entity_key, overrides in extension.entity_overrides.items():
        if "name" in overrides and overrides["name"] != entity_key:
            config["default_path_patterns"] = _rename_template_vars(
                config["default_path_patterns"],
                entity_key,
                overrides["name"],
            )

    # 3. Inject entity segments into existing patterns
    for injection in extension.inject_entity_segments:
        segment = injection["segment"]
        after = injection["after"]
        config["default_path_patterns"] = [
            p.replace(after, after + segment)
            if (after in p and segment not in p) else p
            for p in config["default_path_patterns"]
        ]

    # 4. Insert extra entities at specified positions
    pseudo_names = {"suffix", "scans", "fmap", "datatype", "extension"}

    for extra_ent in extension.extra_entities:
        extra_ent = dict(extra_ent)  # Don't mutate the original
        position = extra_ent.pop("position", "end")

        if position == "end":
            # Insert before pseudo-entities
            idx = _find_first_pseudo_index(config["entities"], pseudo_names)
            config["entities"].insert(idx, extra_ent)
        elif position.startswith("after:"):
            target = position.split(":", 1)[1]
            for i, ent in enumerate(config["entities"]):
                if ent["name"] == target:
                    config["entities"].insert(i + 1, extra_ent)
                    break
        elif position.startswith("before:"):
            target = position.split(":", 1)[1]
            for i, ent in enumerate(config["entities"]):
                if ent["name"] == target:
                    config["entities"].insert(i, extra_ent)
                    break

    # 5. Extend datatype pattern
    if extension.extra_datatypes:
        for ent in config["entities"]:
            if ent["name"] == "datatype":
                m = re.search(r"\(([^)]+)\)", ent["pattern"])
                if m:
                    existing = set(m.group(1).split("|"))
                    existing.update(extension.extra_datatypes)
                    new_vals = "|".join(sorted(existing))
                    ent["pattern"] = ent["pattern"].replace(
                        m.group(1), new_vals
                    )
                break

    # 6. Append extra path patterns
    config["default_path_patterns"].extend(extension.extra_path_patterns)

    return config


def _find_first_pseudo_index(entities: list[dict], pseudo_names: set) -> int:
    """Find the index of the first pseudo-entity in the list."""
    for i, ent in enumerate(entities):
        if ent["name"] in pseudo_names:
            return i
    return len(entities)


def generate_extended_config(
    name: str = "bids",
    extensions: list[ConfigExtension] | None = None,
    schema=None,
    schema_path: str | None = None,
    rule_groups: list[str] | None = None,
) -> dict:
    """Generate config and apply extensions in one step.

    Parameters
    ----------
    name : str
        Config name.
    extensions : list of ConfigExtension, optional
        Extensions to apply in order.
    schema : optional
        Pre-loaded BIDS schema.
    schema_path : str, optional
        Path to schema directory.
    rule_groups : list of str, optional
        Rule groups to process.

    Returns
    -------
    dict
        The final config dict ready for use with ``Config.load()`` or
        ``BIDSLayout``.
    """
    config = generate_config(
        name=name, schema=schema, schema_path=schema_path,
        rule_groups=rule_groups,
    )

    if extensions:
        for ext in extensions:
            config = apply_extension(config, ext)

    return config


# ---------------------------------------------------------------------------
# Standalone path builder
# ---------------------------------------------------------------------------

# Module-level cache for the default schema-generated patterns
_default_patterns_cache: dict[str, list[str]] = {}


def bids_path(
    entities: dict[str, Any],
    patterns: list[str] | None = None,
    schema=None,
    strict: bool = False,
) -> str | None:
    """Build a BIDS-compliant path from an entity dictionary.

    A standalone convenience function that does not require a ``BIDSLayout``.

    Parameters
    ----------
    entities : dict
        Entity name-value pairs (e.g., ``{"subject": "01", "suffix": "T1w"}``).
    patterns : list of str, optional
        Path patterns to match against. If ``None``, generates patterns
        from the BIDS schema covering both raw and derivative files.
    schema : optional
        Pre-loaded BIDS schema for pattern generation.
    strict : bool
        If ``True``, all entities must match a pattern.

    Returns
    -------
    str or None
        The constructed path, or ``None`` if no pattern matched.

    Examples
    --------
    >>> bids_path({
    ...     "subject": "01",
    ...     "datatype": "anat",
    ...     "suffix": "T1w",
    ...     "extension": ".nii.gz",
    ... })
    'sub-01/anat/sub-01_T1w.nii.gz'
    """
    if patterns is None:
        patterns = _get_default_patterns(schema)

    return build_path(entities, patterns, strict=strict)


def _get_default_patterns(schema=None) -> list[str]:
    """Get or generate default path patterns (cached)."""
    cache_key = "default"
    if cache_key not in _default_patterns_cache:
        _require_schema()
        if schema is None:
            schema = load_schema()
        config = generate_config(
            name="all", schema=schema, rule_groups=["raw", "deriv"]
        )
        _default_patterns_cache[cache_key] = config["default_path_patterns"]
    return _default_patterns_cache[cache_key]
