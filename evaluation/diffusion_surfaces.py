"""Lossless surface adapters for finite-world diffusion experiments.

The lexical and byte modes are explicitly surrogate tokenizations, useful for
isolating representation choices. Qwen modes use the reviewed, hash-verified
tokenizer loader from the existing evaluation harness. This module never
downloads assets, imports model code, or runs inference.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Iterable

from sylang_core import encode

from .harness import EvaluationError, load_tokenizers

DEFAULT_TOKENIZER_MANIFEST = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "tokenizers.lock.json"
)
LEXICAL_PATTERN = r"\w+|\s+|[^\w\s]"
_LEXICAL = re.compile(LEXICAL_PATTERN)
_ALIASES = {
    "qwen3": "Qwen/Qwen3-0.6B-Base",
    "qwen3.5": "Qwen/Qwen3.5-0.8B-Base",
}
SURFACE_FORMATS = ("english", "json", "m")


@lru_cache(maxsize=4)
def _local_tokenizers(manifest_path: str):
    # All version checks, asset hashes, and special-token configuration stay in
    # the shared loader. Assets must already exist; no implicit download occurs.
    return {item.metadata["id"]: item for item in load_tokenizers(manifest_path)}


def _tokenizer(name: str, manifest_path: str | Path | None):
    identifier = _ALIASES.get(name, name)
    manifest = Path(manifest_path or DEFAULT_TOKENIZER_MANIFEST).resolve()
    try:
        return _local_tokenizers(str(manifest))[identifier]
    except KeyError as error:
        raise EvaluationError(f"Unknown diffusion surface tokenizer: {name!r}") from error


def tokenizer_metadata(
    tokenizer: str = "lexical", *, tokenizer_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Describe token units without implying a trained diffusion model exists."""
    if tokenizer == "lexical":
        return dict(
            id="lossless-lexical-v1", kind="surrogate", native_model_tokens=False,
            unit="Unicode word run, whitespace run, or single punctuation/symbol",
            pattern=LEXICAL_PATTERN, whitespace_preserved=True,
            quoted_literals_atomic=False,
        )
    if tokenizer == "utf8_bytes":
        return dict(
            id="utf8-bytes-v1", kind="surrogate", native_model_tokens=False,
            unit="one UTF-8 byte", whitespace_preserved=True,
            quoted_literals_atomic=False,
        )
    local = _tokenizer(tokenizer, tokenizer_manifest)
    result = copy.deepcopy(local.metadata)
    result.update(kind="pretrained_tokenizer", native_model_tokens=True,
                  add_special_tokens=False, model_inference_run=False)
    return result


def tokenize_surface(
    text: str, tokenizer: str = "lexical", *,
    tokenizer_manifest: str | Path | None = None,
) -> tuple[str | int, ...]:
    """Return lossless position units; reject lossy/unknown/special Qwen output.

    The lexical rule deliberately splits the contents of quoted literals just
    like other text. Treating an entire literal as one semantic unit would
    silently introduce a fusion advantage into this surface comparison.
    """
    if not isinstance(text, str):
        raise TypeError("A surface must be a string")
    if tokenizer == "lexical":
        units = tuple(_LEXICAL.findall(text))
        if "".join(units) != text:
            raise EvaluationError("Lexical surface tokenization lost text")
        return units
    if tokenizer == "utf8_bytes":
        return tuple(text.encode("utf-8"))
    local = _tokenizer(tokenizer, tokenizer_manifest)
    encoded = local.tokenizer.encode(text, add_special_tokens=False)
    identifiers = set(encoded.ids)
    if identifiers & (local.unknown_ids | local.special_ids):
        raise EvaluationError("Surface produced unknown or reserved special token IDs")
    if local.tokenizer.decode(encoded.ids, skip_special_tokens=False) != text:
        raise EvaluationError("Surface tokenizer did not preserve text exactly")
    return tuple(encoded.ids)


def decode_surface_tokens(
    tokens: Iterable[str | int], tokenizer: str = "lexical", *,
    tokenizer_manifest: str | Path | None = None,
) -> str:
    """Invert a complete token stream; masked streams are not decoder inputs."""
    units = tuple(tokens)
    if tokenizer == "lexical":
        if any(type(unit) is not str for unit in units):
            raise EvaluationError("Lexical surface tokens must be strings")
        return "".join(units)
    if any(type(unit) is not int or unit < 0 for unit in units):
        raise EvaluationError("Byte/model surface tokens must be nonnegative integers")
    if tokenizer == "utf8_bytes":
        if any(unit > 255 for unit in units):
            raise EvaluationError("UTF-8 byte token is outside 0..255")
        try:
            return bytes(units).decode("utf-8")
        except UnicodeError as error:
            raise EvaluationError("Byte token stream is not valid UTF-8") from error
    local = _tokenizer(tokenizer, tokenizer_manifest)
    if any(unit > 0xFFFFFFFF or local.tokenizer.id_to_token(unit) is None for unit in units):
        raise EvaluationError("Model token ID is outside the vocabulary")
    if set(units) & (local.unknown_ids | local.special_ids):
        raise EvaluationError("Surface contains unknown or reserved special token IDs")
    return local.tokenizer.decode(list(units), skip_special_tokens=False)


def surface_text(ast: Any, format_name: str) -> str:
    """Use the existing canonical codec rather than a benchmark-only grammar."""
    if format_name not in SURFACE_FORMATS:
        raise EvaluationError(f"Expected one of {SURFACE_FORMATS}: {format_name!r}")
    return encode(ast, format_name)


def surface_tokens(
    ast: Any, format_name: str, tokenizer: str = "lexical", *,
    tokenizer_manifest: str | Path | None = None,
) -> tuple[str | int, ...]:
    return tokenize_surface(surface_text(ast, format_name), tokenizer,
                            tokenizer_manifest=tokenizer_manifest)
