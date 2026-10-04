"""Fetch the review's tokenizer-only assets from official vendor distributions.

No model weights, no remote code execution. Wheels are downloaded with
``pip download --no-deps`` and only the named data files are extracted; every
extracted file is verified against a pinned SHA-256 before use. Hugging Face was
unreachable under the review environment's egress policy, so the Qwen3
vocabulary comes from Alibaba's DashScope wheel and is validated count-for-count
against the repository's committed Qwen3 results by ``run.py``.

Usage (explicit opt-in, ~70 MB of wheels, ~50 MB extracted):
    python benchmarks/claude-review-2026-10-04/fetch_assets.py --download
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "review-tokenizers"

# (pip requirement, member path inside wheel, target relative path, sha256 of extracted file)
WHEEL_ASSETS = [
    ("dashscope==1.27.7", "dashscope/resources/qwen.tiktoken", "qwen.tiktoken",
     "b2b1b8dfb5cc5f024bafc373121c6aba3f66f9a5a0269e243470a1de16a33186"),
    ("llama-models==0.3.0", "llama_models/llama3/tokenizer.model", "llama3/tokenizer.model",
     "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55"),
    ("llama-models==0.3.0", "llama_models/llama4/tokenizer.model", "llama4/tokenizer.model",
     "d0bdbaf59b0762c8c807617e2d8ea51420eb1b1de266df2495be755c8e0ed6ed"),
    ("mistral-common==1.12.0", "mistral_common/data/tekken_240911.json", "tekken_240911.json",
     "1948e2d48b0e7377f1bb5f1210f1ae5f984934e75713fc07e2452729b8365316"),
    # tiktoken cache files (named by sha1 of the official URL); SHA-256 equals tiktoken's expected_hash.
    ("litellm==1.104.0", "litellm/litellm_core_utils/tokenizers/fb374d419588a4632f3f557e76b4b70aebbca790",
     "fb374d419588a4632f3f557e76b4b70aebbca790",
     "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"),
    ("litellm==1.104.0", "litellm/litellm_core_utils/tokenizers/9b5ad71b2ce5302211f9c61530b329a4922fc6a4",
     "9b5ad71b2ce5302211f9c61530b329a4922fc6a4",
     "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
]
URL_ASSETS = [
    ("https://storage.googleapis.com/gemma-data/tokenizers/tokenizer_gemma3.model", "gemma3.model",
     "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c"),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def put(target: Path, data: bytes, expected: str) -> None:
    if sha256(data) != expected:
        raise SystemExit(f"SHA-256 mismatch for {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    print(f"ok {target.relative_to(CACHE)} {expected[:16]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="explicitly allow the downloads")
    args = parser.parse_args()
    if not args.download:
        for req, member, target, digest in WHEEL_ASSETS:
            print(f"{req:24s} {member} -> {target} sha256={digest}")
        for url, target, digest in URL_ASSETS:
            print(f"{url} -> {target} sha256={digest}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for req in sorted({a[0] for a in WHEEL_ASSETS}):
            subprocess.run([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "--only-binary=:all:",
                            req, "-d", tmp], check=True)
        wheels = list(Path(tmp).glob("*.whl"))
        for req, member, target, digest in WHEEL_ASSETS:
            name = req.split("==")[0].replace("-", "_")
            wheel = next(w for w in wheels if w.name.lower().startswith(name.lower()))
            with zipfile.ZipFile(wheel) as archive:
                put(CACHE / target, archive.read(member), digest)
    for url, target, digest in URL_ASSETS:
        with urllib.request.urlopen(url, timeout=60) as response:
            put(CACHE / target, response.read(), digest)


if __name__ == "__main__":
    main()
