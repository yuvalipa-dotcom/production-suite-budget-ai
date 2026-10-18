"""Minimal JSON-Schema validation for the keywords used by ``contracts/dataset_contract.schema.json``.

Supported: type, required, properties, additionalProperties, items, minItems,
minimum, maximum, pattern, const, enum. An unsupported keyword is an error, so
the schema can never be silently under-enforced.
"""

from __future__ import annotations

import re

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}
_KNOWN = {"$schema", "title", "type", "required", "properties", "additionalProperties", "items",
          "minItems", "minimum", "maximum", "pattern", "const", "enum"}


def errors(value, schema: dict, path: str = "$") -> list[str]:
    unknown = set(schema) - _KNOWN
    if unknown:
        raise ValueError(f"unsupported schema keywords at {path}: {sorted(unknown)}")
    out: list[str] = []
    if "type" in schema and not _TYPES[schema["type"]](value):
        return [f"{path}: expected {schema['type']}"]
    if "const" in schema and value != schema["const"]:
        out.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path}: not one of {schema['enum']!r}")
    if "pattern" in schema and isinstance(value, str) and not re.search(schema["pattern"], value):
        out.append(f"{path}: does not match {schema['pattern']}")
    if "minimum" in schema and _TYPES["number"](value) and value < schema["minimum"]:
        out.append(f"{path}: below {schema['minimum']}")
    if "maximum" in schema and _TYPES["number"](value) and value > schema["maximum"]:
        out.append(f"{path}: above {schema['maximum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        out += [f"{path}: missing {k}" for k in schema.get("required", []) if k not in value]
        extra = schema.get("additionalProperties", True)
        for k in sorted(value):
            if k in props:
                out += errors(value[k], props[k], f"{path}.{k}")
            elif extra is False:
                out.append(f"{path}: unexpected {k}")
            elif isinstance(extra, dict):
                out += errors(value[k], extra, f"{path}.{k}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            out.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                out += errors(item, schema["items"], f"{path}[{i}]")
    return out
