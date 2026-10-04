"""Alternative reversible renderings of the same core-v0.1 tree, for comparison only.

Each variant has a renderer and a strict decoder so reversibility is tested, not
assumed. Literals keep the repository's JSON quoting. These are measurement
controls for the independent review; they do not change sylang_core.
"""
from __future__ import annotations

import json

from sylang_core import SCHEMA_VERSION, encode, validate
from sylang_core.codec import _EVIDENCE_ENGLISH, _VERB_LOOKUP, _tokens, _verb, _quote, _CODES

FIELDS = ("subject", "relation", "object", "polarity", "tense", "aspect", "evidence")
DEFAULTS = dict(polarity="positive", tense="present", aspect="simple", evidence="unspecified")
FEATURE_ORDER = ("tense", "aspect", "evidence")  # polarity is a prefix in cdsl


class DecodeError(ValueError):
    pass


# ---------- renderers (body only; no per-record version header) ----------

def r_english_concise(node):
    """Controlled English without the version header; unspecified evidence omitted."""
    if node["kind"] == "if":
        return f"If [{r_english_concise(node['condition'])}] then [{r_english_concise(node['consequence'])}]."
    prefix = "" if node["evidence"] == "unspecified" else _EVIDENCE_ENGLISH[node["evidence"]] + " "
    verb = _verb(node["relation"], node["polarity"], node["tense"], node["aspect"])
    return f"{prefix}{_quote(node['subject'])} {verb} {_quote(node['object'])}."


def r_english_natural(node):
    """Unquoted natural English. NOT reversible in general; a reference row only."""
    if node["kind"] == "if":
        return f"If {r_english_natural(node['condition']).rstrip('.')}, then {r_english_natural(node['consequence'])}"
    ev = {"unspecified": "", "direct": " (observed)", "reported": " (reportedly)", "inferred": " (inferred)"}[node["evidence"]]
    verb = _verb(node["relation"], node["polarity"], node["tense"], node["aspect"])
    return f"{node['subject']} {verb} {node['object']}{ev}."


def r_m_body(node):
    return encode({"version": SCHEMA_VERSION, "statement": node}, "m")[len("M0.1:"):]


def r_m_words(node):
    """M's positional layout with full enum words: isolates the effect of 1-letter codes."""
    if node["kind"] == "if":
        return f"i({r_m_words(node['condition'])},{r_m_words(node['consequence'])})"
    vals = [_quote(node[f]) if f in ("subject", "object") else
            (_CODES["polarity"][node[f]] if f == "polarity" else node[f]) for f in FIELDS]
    return "p(" + ",".join(vals) + ")"


def r_json_array(node):
    """Positional compact JSON array, full enum words."""
    return json.dumps(_arr(node), ensure_ascii=False, separators=(",", ":"))


def _arr(node):
    if node["kind"] == "if":
        return ["if", _arr(node["condition"]), _arr(node["consequence"])]
    return [node[f] for f in FIELDS]


def r_json_short(node):
    """Compact JSON with short keys and defaults omitted."""
    return json.dumps(_short(node), ensure_ascii=False, separators=(",", ":"))


def _short(node):
    if node["kind"] == "if":
        return {"if": _short(node["condition"]), "then": _short(node["consequence"])}
    out = {"s": node["subject"], "r": node["relation"], "o": node["object"]}
    for f in ("polarity", "tense", "aspect", "evidence"):
        if node[f] != DEFAULTS[f]:
            out[f] = node[f]
    return out


def r_cdsl(node):
    """Compact readable DSL: [not ]REL("S","O")[ FEATURE...]; defaults omitted; words not codes."""
    if node["kind"] == "if":
        return f"if({r_cdsl(node['condition'])};{r_cdsl(node['consequence'])})"
    s = ("not " if node["polarity"] == "negative" else "") + f"{node['relation']}({_quote(node['subject'])},{_quote(node['object'])})"
    for f in FEATURE_ORDER:
        if node[f] != DEFAULTS[f]:
            s += " " + node[f]
    return s


def r_cdsl_full(node):
    """Same compact DSL with every feature explicit (no defaults)."""
    if node["kind"] == "if":
        return f"if({r_cdsl_full(node['condition'])};{r_cdsl_full(node['consequence'])})"
    return (f"{node['relation']}({_quote(node['subject'])},{_quote(node['object'])}) "
            f"{node['polarity']} {node['tense']} {node['aspect']} {node['evidence']}")


RENDER = {
    "english_concise": r_english_concise,
    "m_body": r_m_body,
    "m_words": r_m_words,
    "json_array": r_json_array,
    "json_short": r_json_short,
    "cdsl": r_cdsl,
    "cdsl_full": r_cdsl_full,
}
NONREVERSIBLE = {"english_natural": r_english_natural}


# ---------- strict decoders (reuse the core tokenizer for JSON-quoted literals) ----------

class _R:
    def __init__(self, text):
        self.t = _tokens(text)
        self.i = 0

    def peek(self, k=0):
        return self.t[self.i + k] if self.i + k < len(self.t) else None

    def word(self, allowed):
        tok = self.peek()
        if tok is None or tok.kind == "string" or tok.value not in allowed:
            raise DecodeError(f"expected one of {sorted(allowed)[:6]} at {self.i}")
        self.i += 1
        return tok.value

    def expect(self, value):
        self.word({value})

    def string(self):
        tok = self.peek()
        if tok is None or tok.kind != "string":
            raise DecodeError("expected string")
        self.i += 1
        return tok.value

    def done(self):
        if self.i != len(self.t):
            raise DecodeError("trailing input")

    def phrase(self, table):
        for phrase in sorted(table, key=len, reverse=True):
            exp = _tokens(phrase)
            act = self.t[self.i:self.i + len(exp)]
            if len(act) == len(exp) and all(a.kind == b.kind and a.value == b.value for a, b in zip(act, exp)):
                self.i += len(exp)
                return table[phrase]
        return None


def _doc(node):
    ast = {"version": SCHEMA_VERSION, "statement": node}
    validate(ast)
    return ast


def d_english_concise(text):
    r = _R(text)

    def node():
        tok = r.peek()
        if tok is not None and tok.kind == "word" and tok.value == "If":
            r.expect("If"); r.expect("[")
            c = node(); r.expect("]"); r.expect("then"); r.expect("[")
            q = node(); r.expect("]"); r.expect(".")
            return {"kind": "if", "condition": c, "consequence": q}
        ev = r.phrase({v: k for k, v in _EVIDENCE_ENGLISH.items() if k != "unspecified"}) or "unspecified"
        s = r.string()
        vp = r.phrase(_VERB_LOOKUP)
        if vp is None:
            raise DecodeError("verb phrase")
        o = r.string(); r.expect(".")
        rel, pol, tense, aspect = vp
        return dict(kind="pred", subject=s, relation=rel, object=o, polarity=pol, tense=tense, aspect=aspect, evidence=ev)

    n = node(); r.done()
    return _doc(n)


def d_m_body(text):
    from sylang_core import decode
    return decode("M0.1:" + text, "m")


def d_m_words(text):
    from sylang_core.codec import ASPECTS, EVIDENCES, RELATIONS, TENSES
    r = _R(text)

    def node():
        k = r.word({"p", "i"}); r.expect("(")
        if k == "i":
            c = node(); r.expect(","); q = node(); r.expect(")")
            return {"kind": "if", "condition": c, "consequence": q}
        s = r.string(); r.expect(","); rel = r.word(set(RELATIONS)); r.expect(","); o = r.string(); r.expect(",")
        pol = {"+": "positive", "-": "negative"}[r.word({"+", "-"})]; r.expect(",")
        tense = r.word(set(TENSES)); r.expect(","); aspect = r.word(set(ASPECTS)); r.expect(",")
        ev = r.word(set(EVIDENCES)); r.expect(")")
        return dict(kind="pred", subject=s, relation=rel, object=o, polarity=pol, tense=tense, aspect=aspect, evidence=ev)

    n = node(); r.done()
    return _doc(n)


def d_json_array(text):
    def build(a):
        if not isinstance(a, list):
            raise DecodeError("array expected")
        if len(a) == 3 and a[0] == "if":
            return {"kind": "if", "condition": build(a[1]), "consequence": build(a[2])}
        if len(a) != 7:
            raise DecodeError("arity")
        return {"kind": "pred", **dict(zip(FIELDS, a))}
    return _doc(build(json.loads(text)))


def d_json_short(text):
    def build(o):
        if not isinstance(o, dict):
            raise DecodeError("object expected")
        if set(o) == {"if", "then"}:
            return {"kind": "if", "condition": build(o["if"]), "consequence": build(o["then"])}
        if not {"s", "r", "o"} <= set(o) or not set(o) <= {"s", "r", "o", *DEFAULTS}:
            raise DecodeError("keys")
        node = {"kind": "pred", "subject": o["s"], "relation": o["r"], "object": o["o"]}
        for f, dv in DEFAULTS.items():
            if f in o and o[f] == dv:
                raise DecodeError("non-canonical explicit default")
            node[f] = o.get(f, dv)
        return node
    return _doc(build(json.loads(text)))


def _d_cdsl(text, full):
    from sylang_core.codec import ASPECTS, EVIDENCES, RELATIONS, TENSES
    r = _R(text)
    rank = {**{t: ("tense", t) for t in TENSES}, **{a: ("aspect", a) for a in ASPECTS},
            **{e: ("evidence", e) for e in EVIDENCES}}

    def node():
        tok = r.peek()
        if tok is not None and tok.kind == "word" and tok.value == "if" and r.peek(1) and r.peek(1).value == "(":
            r.expect("if"); r.expect("(")
            c = node(); r.expect(";"); q = node(); r.expect(")")
            return {"kind": "if", "condition": c, "consequence": q}
        pol = "positive"
        if not full and tok is not None and tok.kind == "word" and tok.value == "not":
            r.expect("not"); pol = "negative"
        rel = r.word(set(RELATIONS)); r.expect("("); s = r.string(); r.expect(","); o = r.string(); r.expect(")")
        feats = dict(DEFAULTS, polarity=pol)
        if full:
            feats["polarity"] = r.word({"positive", "negative"})
            feats["tense"] = r.word(set(TENSES)); feats["aspect"] = r.word(set(ASPECTS)); feats["evidence"] = r.word(set(EVIDENCES))
        else:
            last = -1
            while r.peek() is not None and r.peek().kind == "word" and r.peek().value in rank:
                field, value = rank[r.word(set(rank))]
                order = FEATURE_ORDER.index(field)
                if order <= last or value == DEFAULTS[field]:
                    raise DecodeError("non-canonical feature order/default")
                feats[field] = value
                last = order
        return dict(kind="pred", subject=s, relation=rel, object=o, **feats)

    n = node(); r.done()
    return _doc(n)


DECODE = {
    "english_concise": d_english_concise,
    "m_body": d_m_body,
    "m_words": d_m_words,
    "json_array": d_json_array,
    "json_short": d_json_short,
    "cdsl": lambda t: _d_cdsl(t, False),
    "cdsl_full": lambda t: _d_cdsl(t, True),
}


def all_renderings(ast):
    """Existing canonical formats + body-only and alternative controls."""
    node = ast["statement"]
    out = {f: encode(ast, f) for f in ("english", "json", "dsl", "prime", "m")}
    out.update({k: fn(node) for k, fn in RENDER.items()})
    out.update({k: fn(node) for k, fn in NONREVERSIBLE.items()})
    return out


# ---------- best-effort optimized M (defaults omitted, relation as head) ----------
_OPT_REL = {"see": "s", "help": "h", "contain": "c"}
_OPT_FEAT = (("tense", {"past": "p", "future": "f"}), ("aspect", {"progressive": "g", "completed": "c"}),
             ("evidence", {"direct": "d", "reported": "r", "inferred": "i"}))


def r_m_opt(node):
    """Optimized M: [!]R(S,O)[/codes]; i(X,Y). Codes p,f,g,c,d,r,i are field-unique."""
    if node["kind"] == "if":
        return f"i({r_m_opt(node['condition'])},{r_m_opt(node['consequence'])})"
    codes = "".join(table[node[f]] for f, table in _OPT_FEAT if node[f] in table)
    return (("!" if node["polarity"] == "negative" else "") + _OPT_REL[node["relation"]]
            + f"({_quote(node['subject'])},{_quote(node['object'])})" + (f"/{codes}" if codes else ""))


def d_m_opt(text):
    import re
    # Literal-aware: tokenize with the core tokenizer, then interpret '!' and '/codes'.
    pos = {"text": text}
    toks = []
    i = 0
    from sylang_core.codec import _STRING_DECODER
    while i < len(text):
        ch = text[i]
        if ch == '"':
            v, j = _STRING_DECODER.raw_decode(text, i); toks.append(("s", v)); i = j; continue
        m = re.compile(r"[a-z]+").match(text, i)
        if m:
            toks.append(("w", m.group())); i = m.end(); continue
        if ch in "!(),/":
            toks.append(("p", ch)); i += 1; continue
        raise DecodeError(f"bad char {ch!r}")
    k = [0]

    def take(kind, val=None):
        if k[0] >= len(toks) or toks[k[0]][0] != kind or (val is not None and toks[k[0]][1] != val):
            raise DecodeError("unexpected token")
        k[0] += 1
        return toks[k[0] - 1][1]

    inv_rel = {v: r for r, v in _OPT_REL.items()}

    def node():
        neg = False
        if k[0] < len(toks) and toks[k[0]] == ("p", "!"):
            take("p", "!"); neg = True
        head = take("w")
        take("p", "(")
        if head == "i" and not neg:
            c = node(); take("p", ","); q = node(); take("p", ")")
            return {"kind": "if", "condition": c, "consequence": q}
        if head not in inv_rel:
            raise DecodeError("relation")
        s = take("s"); take("p", ","); o = take("s"); take("p", ")")
        feats = dict(DEFAULTS, polarity="negative" if neg else "positive")
        if k[0] < len(toks) and toks[k[0]] == ("p", "/"):
            take("p", "/"); codes = take("w"); idx = 0
            for f, table in _OPT_FEAT:
                inv = {v: x for x, v in table.items()}
                if idx < len(codes) and codes[idx] in inv:
                    feats[f] = inv[codes[idx]]; idx += 1
            if idx != len(codes) or not codes:
                raise DecodeError("codes")
        return dict(kind="pred", subject=s, relation=inv_rel[head], object=o, **feats)

    n = node()
    if k[0] != len(toks):
        raise DecodeError("trailing")
    return _doc(n)


RENDER["m_opt"] = r_m_opt
DECODE["m_opt"] = d_m_opt
