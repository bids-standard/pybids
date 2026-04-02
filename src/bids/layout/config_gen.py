"""Generate pybids path patterns and entity lists from the BIDS schema.

This module converts the structured BIDS schema (loaded via bidsschematools)
into pybids-format entity lists and path patterns, and provides a standalone
``bids_path()`` convenience function.

Requires the optional ``bidsschematools`` package::

    pip install pybids[schema]
"""
from __future__ import annotations

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

# Index-format entities get dtype: int
_INDEX_DTYPE = "int"


def _get_format_patterns(schema) -> dict[str, str]:
    """Build format → regex capture group mapping from the schema.

    Reads ``schema['objects']['formats']`` and extracts the ``pattern``
    attribute for each format that has one.
    """
    formats = schema["objects"]["formats"]
    patterns = {}
    for key, fmt_def in formats.items():
        pat = getattr(fmt_def, "pattern", None) or fmt_def.get("pattern")
        if pat:
            patterns[key] = "(%s)" % pat
    return patterns


def _get_directory_entities(schema) -> dict[str, str]:
    """Derive directory entity mappings from ``schema['rules']['directories']``.

    Returns a dict mapping entity keys to their directory template strings,
    e.g. ``{"subject": "{subject}", "session": "{subject}{session}"}``.
    """
    raw_dirs = schema["rules"]["directories"].get("raw", {})
    result = {}

    # Subject is a top-level directory entity
    subj_dir = raw_dirs.get("subject", {})
    if hasattr(subj_dir, "entity") or (isinstance(subj_dir, Mapping) and "entity" in subj_dir):
        result["subject"] = "{subject}"

    # Session is nested under subject
    sess_dir = raw_dirs.get("session", {})
    if hasattr(sess_dir, "entity") or (isinstance(sess_dir, Mapping) and "entity" in sess_dir):
        result["session"] = "{subject}{session}"

    return result


def schema_entity_to_pybids(
    entity_key: str,
    entity_def: dict,
    format_patterns: dict,
    directory_entities: dict,
) -> dict:
    """Convert a single BIDS schema entity definition to a pybids entity dict.

    Parameters
    ----------
    entity_key : str
        The long entity name as it appears in the schema (e.g., ``"acquisition"``).
    entity_def : dict
        The entity definition from ``schema.objects.entities[entity_key]``.
    format_patterns : dict
        Format → regex capture group mapping from :func:`_get_format_patterns`.
    directory_entities : dict
        Directory entity mapping from :func:`_get_directory_entities`.

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
        # Fall back to label pattern if format not found
        label_pattern = format_patterns.get("label", "([a-zA-Z0-9+]+)")
        capture = format_patterns.get(fmt, label_pattern)

    # Build prefix: directory entities use path separator, others use _ or separator
    if entity_key == "subject":
        prefix = "[/\\\\]+"
    else:
        prefix = "[_/\\\\]+"

    pattern = "%s%s-%s" % (prefix, short_name, capture)

    result = {"name": entity_key, "pattern": pattern}

    # Directory entities
    if entity_key in directory_entities:
        result["directory"] = directory_entities[entity_key]

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


def _build_suffix_entity(schema) -> dict:
    """Build the suffix pseudo-entity from schema suffix definitions."""
    suffixes = schema["objects"].get("suffixes", {})
    if suffixes:
        suffix_names = sorted(suffixes.keys())
        capture = "(%s)" % "|".join(suffix_names)
    else:
        # Fallback: match alphanumeric
        capture = "([a-zA-Z0-9+]+)"
    return {
        "name": "suffix",
        "pattern": "(?:^|[_/\\\\])%s\\.[^/\\\\]+$" % capture,
    }


def _build_extension_entity(schema) -> dict:
    """Build the extension pseudo-entity from schema extension definitions."""
    extensions = schema["objects"].get("extensions", {})
    if extensions:
        # Get all extension values, escape dots for regex
        ext_values = []
        for key, ext_def in extensions.items():
            # Extension keys in the schema are the actual extensions (e.g., ".nii.gz")
            val = ext_def.get("value", key) if isinstance(ext_def, Mapping) else key
            ext_values.append(val.replace(".", "\\."))
        capture = "(%s)" % "|".join(sorted(ext_values, key=len, reverse=True))
    else:
        # Fallback: match any extension
        capture = "(\\.[^/\\\\]+)"
    return {
        "name": "extension",
        "pattern": "[^./\\\\]%s$" % capture,
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
    format_patterns = _get_format_patterns(schema)
    directory_entities = _get_directory_entities(schema)

    entities = []
    for key in entity_order:
        if key not in entity_defs:
            continue
        entities.append(
            schema_entity_to_pybids(key, entity_defs[key], format_patterns, directory_entities)
        )

    # Collect all datatypes from rules for the datatype pseudo-entity
    datatypes = _collect_datatypes(schema)

    # Append schema-derived pseudo-entities
    entities.append(_build_suffix_entity(schema))
    entities.append(_build_datatype_entity(datatypes))
    entities.append(_build_extension_entity(schema))

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
    default: str | None = None,
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
    default : str, optional
        Default value appended after the closing ``>`` or ``}``
        (e.g., ``"image"`` → ``{mode<…>|image}``).

    Returns
    -------
    str
        Pattern segment like ``"[_acq-{acquisition}]"`` or
        ``"_task-{task}"`` (no brackets for required).
    """
    inner = "%s-{%s" % (short_name, entity_key)
    if enum_values:
        inner += "<%s>" % "|".join(enum_values)
    if default:
        inner += "|%s" % default
    inner += "}"

    if level == "required":
        return "_" + inner
    else:
        return "[_" + inner + "]"


def rule_to_path_pattern(
    rule: dict,
    schema,
    sidecar_split: bool = True,
) -> list[str]:
    """Convert a single dereferenced file rule to pybids path pattern(s).

    Parameters
    ----------
    rule : dict
        A dereferenced file rule with keys: ``entities``, ``datatypes``,
        ``suffixes``, ``extensions``.
    schema
        The loaded BIDS schema.
    sidecar_split : bool, optional
        If ``True`` (default), heritable extensions (``.json``, ``.tsv``,
        etc.) are split into a separate sidecar/inheritance pattern.
        If ``False``, all extensions are included in the main pattern
        and no sidecar pattern is generated.

    Returns
    -------
    list of str
        One or more pybids path patterns. The first is the main file pattern;
        if *sidecar_split* is True, subsequent ones are sidecar/inheritance
        patterns for heritable extensions.
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
    if sidecar_split:
        exts_for_main = main_exts if has_main else sorted(all_exts)
    else:
        exts_for_main = sorted(all_exts)
    patterns.append(
        _build_file_pattern(
            entity_order, entity_defs, rule_entities,
            datatypes, suffixes, exts_for_main,
        )
    )

    # --- Sidecar/inheritance pattern ---
    if sidecar_split and sidecar_exts and has_main:
        patterns.append(
            _build_sidecar_pattern(
                entity_order, entity_defs, rule_entities,
                suffixes, sidecar_exts,
            )
        )

    return patterns


def _parse_entity_level(value) -> tuple[str, list[str] | None, str | None]:
    """Parse an entity level specification from a file rule.

    Entity values in rules can be:
    - A string like ``"required"`` or ``"optional"``
    - A Mapping with ``"level"``, optionally ``"enum"`` and ``"default"`` keys

    Returns
    -------
    tuple of (str, list or None, str or None)
        The level string, any enum override, and any default value.
    """
    if isinstance(value, str):
        return value, None, None
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
        default = value.get("default")
        return level, enum_values or None, default
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

        level, rule_enum, default = _parse_entity_level(rule_entities[ent_key])

        ent_def = entity_defs.get(ent_key, {})
        short_name = ent_def.get("name", ent_key)

        # Use rule-level enum override if present, else entity-level enum
        enum_values = rule_enum or _resolve_enum(ent_def)

        parts.append(_format_entity_segment(ent_key, short_name, level, enum_values, default))

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


def generate_path_patterns(
    schema, rule_group: str = "raw", sidecar_split: bool = True,
) -> list[str]:
    """Generate all pybids path patterns from a schema rule group.

    Parameters
    ----------
    schema
        The loaded BIDS schema.
    rule_group : str
        Which rule group to process: ``"raw"`` or ``"deriv"``.
    sidecar_split : bool, optional
        If ``True`` (default), heritable extensions are split into
        separate sidecar patterns.  If ``False``, all extensions are
        kept in a single main pattern per rule.

    Returns
    -------
    list of str
        List of pybids path patterns.
    """
    patterns = []
    seen = set()

    rule_groups = schema["rules"]["files"].get(rule_group, {})
    _generate_patterns_recursive(
        rule_groups, schema, patterns, seen, sidecar_split=sidecar_split,
    )

    return patterns


def _generate_patterns_recursive(obj, schema, patterns, seen, sidecar_split=True):
    """Recursively walk rule groups, generating patterns for leaf rules."""
    if isinstance(obj, Mapping):
        # A leaf rule has entities, suffixes, and extensions
        if "entities" in obj and "suffixes" in obj and "extensions" in obj:
            # Deduplicate: same rule may appear via multiple inheritance paths
            key = _make_rule_key(obj)
            if key not in seen:
                seen.add(key)
                new_patterns = rule_to_path_pattern(
                    obj, schema, sidecar_split=sidecar_split,
                )
                patterns.extend(new_patterns)
        else:
            for v in obj.values():
                if isinstance(v, Mapping):
                    _generate_patterns_recursive(
                        v, schema, patterns, seen, sidecar_split=sidecar_split,
                    )


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
        all_patterns = []
        for rg in ("raw", "deriv"):
            all_patterns.extend(
                generate_path_patterns(schema, rg, sidecar_split=True)
            )
        _default_patterns_cache[cache_key] = all_patterns
    return _default_patterns_cache[cache_key]
