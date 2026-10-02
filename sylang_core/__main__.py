"""Translate one bounded document between the five deterministic core formats."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import FORMATS, decode, encode
from .codec import MAX_TEXT_BYTES, SylangError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source_format", required=True, choices=FORMATS)
    parser.add_argument("--to", dest="target_format", required=True, choices=FORMATS)
    parser.add_argument("--input", type=Path, help="UTF-8 input file (default: standard input)")
    args = parser.parse_args()
    try:
        if args.input is None:
            data = sys.stdin.buffer.read(MAX_TEXT_BYTES + 1)
        else:
            with args.input.open("rb") as stream:
                data = stream.read(MAX_TEXT_BYTES + 1)
        if len(data) > MAX_TEXT_BYTES:
            raise ValueError("input exceeds the UTF-8 byte limit")
        output = encode(decode(data.decode("utf-8"), args.source_format), args.target_format)
    except (OSError, UnicodeError, ValueError, SylangError) as error:
        print(f"sylang: {error}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write((output + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
