"""Experimental, dependency-free Sylang core-v0.1 codecs.

These codecs translate a bounded semantic tree, not unrestricted natural language.
The names Prime and M identify proposed experimental serializations only.
"""

from .codec import (
    ASPECTS,
    EVIDENCES,
    FORMATS,
    MAX_DEPTH,
    MAX_LITERAL_SCALARS,
    MAX_NODES,
    MAX_TEXT_BYTES,
    POLARITIES,
    RELATIONS,
    SCHEMA_VERSION,
    TENSES,
    ParseError,
    SylangError,
    ValidationError,
    canonicalize,
    decode,
    encode,
    semantic_equal,
    validate,
)

__all__ = [
    "ASPECTS", "EVIDENCES", "FORMATS", "MAX_DEPTH", "MAX_LITERAL_SCALARS",
    "MAX_NODES", "MAX_TEXT_BYTES", "POLARITIES", "RELATIONS", "SCHEMA_VERSION",
    "TENSES", "ParseError", "SylangError", "ValidationError", "canonicalize",
    "decode", "encode", "semantic_equal", "validate",
]
