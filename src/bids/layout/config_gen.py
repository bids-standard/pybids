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

def _rename_template_vars(
    patterns: list[str], old_name: str, new_name: str,
) -> list[str]:
    """Rename template variables in path patterns.

    Rewrites ``{old_name}``, ``{old_name<...>}``, ``{old_name|...}`` to use
    *new_name* instead.
    """
    regex = re.compile(r"\{" + re.escape(old_name) + r"(?=[<|}\s])")
    return [regex.sub("{" + new_name, p) for p in patterns]


def _build_extended_entity_order(schema, extension: ConfigExtension) -> list[str]:
    """Build an entity order that includes extension entities at their positions."""
    order = list(schema["rules"]["entities"])

    # Build a lookup so position targets can use either schema keys or
    # short BIDS names (e.g., "after:hemi" matches schema key "hemisphere").
    entity_defs = schema["objects"]["entities"]
    name_map = _build_entity_name_map(entity_defs)

    def _find_target(target):
        """Resolve a position target to an index in *order*."""
        # Direct match (schema key or previously inserted extra entity)
        if target in order:
            return order.index(target)
        # Resolve short name → schema key
        resolved = name_map.get(target)
        if resolved and resolved in order:
            return order.index(resolved)
        return None

    for extra_ent in extension.extra_entities:
        name = extra_ent["name"]
        if name in order:
            continue
        position = extra_ent.get("position", "end")
        if position == "end":
            order.append(name)
        elif position.startswith("after:"):
            target = position.split(":", 1)[1]
            idx = _find_target(target)
            order.insert((idx + 1) if idx is not None else len(order), name)
        elif position.startswith("before:"):
            target = position.split(":", 1)[1]
            idx = _find_target(target)
            order.insert(idx if idx is not None else len(order), name)
    return order


def _build_extended_entity_defs(schema, extension: ConfigExtension) -> dict:
    """Build entity definitions combining schema and extension entities."""
    defs = dict(schema["objects"]["entities"])
    for extra_ent in extension.extra_entities:
        name = extra_ent["name"]
        if name not in defs:
            defs[name] = {"name": name, "format": "label"}
    return defs


def _build_entity_name_map(entity_defs: dict) -> dict[str, str]:
    """Map both schema keys and short BIDS names to schema keys.

    For example, both ``"description"`` and ``"desc"`` map to
    ``"description"``.
    """
    name_map: dict[str, str] = {}
    for key, defn in entity_defs.items():
        name_map[key] = key
        short = defn.get("name", key)
        if short != key:
            name_map[short] = key
    return name_map


def _collect_deriv_entities_for_datatypes(
    schema, datatypes: list[str],
) -> dict[str, str]:
    """Collect the union of entities from deriv rules matching *datatypes*.

    All collected entities are marked ``"optional"``.
    """
    entities: dict[str, str] = {}
    deriv_rules = schema["rules"]["files"].get("deriv", {})
    _collect_rule_entities(deriv_rules, set(datatypes), entities)
    return entities


def _collect_rule_entities(obj, target_datatypes: set, entities: dict):
    """Recursively find rules matching *target_datatypes* and union entities."""
    if isinstance(obj, Mapping):
        if "entities" in obj and "datatypes" in obj:
            if set(obj.get("datatypes", [])) & target_datatypes:
                for ent_key in obj["entities"]:
                    if ent_key not in ("subject", "session"):
                        if ent_key not in entities:
                            entities[ent_key] = "optional"
        else:
            for v in obj.values():
                if isinstance(v, Mapping):
                    _collect_rule_entities(v, target_datatypes, entities)


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
