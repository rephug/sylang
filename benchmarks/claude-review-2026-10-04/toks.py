"""Offline tokenizer adapters for the independent review (no model weights).

Every asset is an official vendor distribution obtained through an allowed
channel. Hugging Face was blocked by the session egress policy, so Qwen3 is
reconstructed from Alibaba's official BPE ranks plus the Qwen3 pre-tokenizer
regex and NFC normalizer recorded in the repository's committed results; the
reconstruction is validated count-for-count against those committed results.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import unicodedata
from pathlib import Path

TOK_DIR = Path(os.environ.get("SYLANG_REVIEW_TOK_DIR", Path(__file__).resolve().parents[2] / ".cache" / "review-tokenizers"))

QWEN_PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
LLAMA3_PAT = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
LLAMA4_PAT = r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# Qwen3 base added tokens (151643..151664); only these strings can be matched atomically.
QWEN3_ADDED = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
               "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
               "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", "<tool_call>",
               "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
               "<|repo_name|>", "<|file_sep|>"]
LLAMA3_SPECIAL = ["<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>",
                  "<|reserved_special_token_1|>", "<|finetune_right_pad_id|>", "<|step_id|>",
                  "<|start_header_id|>", "<|end_header_id|>", "<|eom_id|>", "<|eot_id|>", "<|python_tag|>"]
LLAMA4_SPECIAL = ["<|begin_of_text|>", "<|end_of_text|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
                  "<|header_start|>", "<|header_end|>", "<|eom|>", "<|eot|>", "<|step|>", "<|text_post_train_reserved_special_token_0|>",
                  "<|python_start|>", "<|python_end|>", "<|finetune_right_pad|>"]
O200K_SPECIAL = ["<|endoftext|>", "<|endofprompt|>", "<|start|>", "<|end|>", "<|message|>", "<|channel|>",
                 "<|return|>", "<|call|>", "<|constrain|>"]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Tok:
    """Uniform interface: count(text), pieces(text), roundtrip_exact(text), specials_hit(text)."""

    def __init__(self, name, family, source, path, normalizer, encode, decode, piece, specials, vocab):
        self.name, self.family, self.source, self.normalizer = name, family, source, normalizer
        self.path = Path(path)
        self.sha256 = _sha256(path)
        self._encode, self._decode, self._piece = encode, decode, piece
        self.specials = specials
        self.vocab = vocab

    def ids(self, text):
        return self._encode(text)

    def count(self, text):
        return len(self._encode(text))

    def pieces(self, text):
        return [self._piece(i) for i in self._encode(text)]

    def roundtrip_exact(self, text):
        return self._decode(self._encode(text)) == text

    def specials_hit(self, text):
        return [s for s in self.specials if s in text]

    def meta(self):
        return dict(name=self.name, family=self.family, source=self.source, file=self.path.name,
                    sha256=self.sha256, normalizer=self.normalizer,
                    adapter_ids=self.vocab)


def _tiktoken_tok(name, family, source, path, pat, specials, normalizer=None, ranks=None, offset=0):
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe
    ranks = ranks if ranks is not None else load_tiktoken_bpe(str(path))
    base = max(ranks.values()) + 1
    special_map = {s: base + i for i, s in enumerate(specials)}
    enc = tiktoken.Encoding(name=name, pat_str=pat, mergeable_ranks=ranks, special_tokens=special_map)
    norm = (lambda t: unicodedata.normalize(normalizer, t)) if normalizer else (lambda t: t)

    def encode(text):
        # Mirrors the repository policy encode_special_tokens=False: reserved strings in
        # the input are recognized as single control IDs (a transport hazard, reported).
        return enc.encode(norm(text), allowed_special="all")

    def decode(ids):
        return enc.decode_bytes(ids).decode("utf-8", errors="replace")

    def piece(i):
        return enc.decode_single_token_bytes(i).decode("utf-8", errors="backslashreplace")

    tok = Tok(name, family, source, path, normalizer, encode, decode, piece, list(special_map), base + len(specials))
    tok.encoding = enc
    return tok


def load_all():
    toks = []
    d = TOK_DIR
    toks.append(_tiktoken_tok(
        "qwen3-reconstructed", "Qwen3 (151,643 BPE + 22 added; NFC)",
        "dashscope wheel resources/qwen.tiktoken (Alibaba) + Qwen3 regex/NFC from committed results",
        d / "qwen.tiktoken", QWEN_PAT, QWEN3_ADDED, normalizer="NFC"))
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(d))
    import tiktoken
    o200k = tiktoken.get_encoding("o200k_base")  # verifies OpenAI expected_hash from the cache
    toks.append(_tiktoken_tok("o200k_base", "OpenAI o200k (GPT-4o/4.1/5, o-series; gpt-oss base)",
                              "OpenAI o200k_base.tiktoken via litellm wheel cache; SHA-256 = tiktoken expected_hash",
                              d / "fb374d419588a4632f3f557e76b4b70aebbca790", o200k._pat_str, O200K_SPECIAL,
                              ranks=o200k._mergeable_ranks))
    cl100k = tiktoken.get_encoding("cl100k_base")
    toks.append(_tiktoken_tok("cl100k_base", "OpenAI cl100k (GPT-4/3.5; basis of several open tokenizers)",
                              "OpenAI cl100k_base.tiktoken via litellm wheel cache; SHA-256 = tiktoken expected_hash",
                              d / "9b5ad71b2ce5302211f9c61530b329a4922fc6a4", cl100k._pat_str, ["<|endoftext|>", "<|endofprompt|>"],
                              ranks=cl100k._mergeable_ranks))
    toks.append(_tiktoken_tok("llama3", "Meta Llama 3.x (128k tiktoken BPE)", "llama-models wheel llama3/tokenizer.model",
                              d / "llama3" / "tokenizer.model", LLAMA3_PAT, LLAMA3_SPECIAL))
    toks.append(_tiktoken_tok("llama4", "Meta Llama 4 (200k tiktoken BPE)", "llama-models wheel llama4/tokenizer.model",
                              d / "llama4" / "tokenizer.model", LLAMA4_PAT, LLAMA4_SPECIAL))
    tekken = json.loads((d / "tekken_240911.json").read_text(encoding="utf-8"))
    cfg = tekken["config"]
    inner = cfg["default_vocab_size"] - cfg["default_num_special_tokens"]
    tranks = {base64.b64decode(v["token_bytes"]): v["rank"] for v in tekken["vocab"][:inner]}
    toks.append(_tiktoken_tok("tekken", "Mistral Tekken v3 (131k)", "mistral-common wheel data/tekken_240911.json",
                              d / "tekken_240911.json", cfg["pattern"], ["<s>", "</s>", "[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]"],
                              ranks=tranks))
    toks.append(_gemma(d / "gemma3.model"))
    return toks


def _gemma(path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(path))
    specials = ["<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>", "<pad>", "<unk>", "<mask>"]
    special_ids = {s: sp.piece_to_id(s) for s in specials}

    def encode(text):
        # Split on control strings first (as Hugging Face added-token handling does).
        out, rest = [], text
        while rest:
            hits = [(rest.find(s), s) for s in specials if rest.find(s) >= 0]
            if not hits:
                out += sp.encode(rest)
                break
            pos, s = min(hits)
            if pos:
                out += sp.encode(rest[:pos])
            out.append(special_ids[s])
            rest = rest[pos + len(s):]
        return out

    def decode(ids):
        return sp.decode(ids)

    def piece(i):
        return sp.id_to_piece(i)

    return Tok("gemma3", "Google Gemma 3 (262k SentencePiece, byte fallback)",
               "storage.googleapis.com/gemma-data/tokenizers/tokenizer_gemma3.model (Google)",
               path, None, encode, decode, piece, specials, sp.get_piece_size())
