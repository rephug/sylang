"""Explicit, bounded tokenizer-data download; never downloads or executes model code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (
    {
        "id": "Qwen/Qwen3-0.6B-Base",
        "revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "filename": "qwen3-base.json",
        "size_bytes": 7031645,
        "sha256": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "git_blob_sha1": "443909a61d429dff23010e5bddd28ff530edda00",
        "license": "Apache-2.0",
    },
    {
        "id": "Qwen/Qwen3.5-0.8B-Base",
        "revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        "filename": "qwen3.5-base.json",
        "size_bytes": 12807196,
        "sha256": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
        "license": "Apache-2.0",
    },
)


def verify(data: bytes, asset: dict) -> str:
    if len(data) != asset["size_bytes"]:
        raise ValueError("Unexpected size for " + asset["id"])
    sha = hashlib.sha256(data).hexdigest()
    if "sha256" in asset and sha != asset["sha256"]:
        raise ValueError("SHA-256 mismatch for " + asset["id"])
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if "git_blob_sha1" in asset and blob != asset["git_blob_sha1"]:
        raise ValueError("Pinned Git blob mismatch for " + asset["id"])
    parsed = json.loads(data)
    if parsed.get("model", {}).get("type") != "BPE":
        raise ValueError("Unexpected tokenizer format")
    return sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Explicitly allow ~19.9 MB of HTTPS downloads")
    args = parser.parse_args()
    if not args.download:
        print(json.dumps({"download_required": True, "assets": ASSETS}, indent=2))
        return
    cache = ROOT / ".cache" / "tokenizers"
    cache.mkdir(parents=True, exist_ok=True)
    manifest = []
    for asset in ASSETS:
        source = f"https://huggingface.co/{asset['id']}/resolve/{asset['revision']}/tokenizer.json"
        target = cache / asset["filename"]
        if target.exists():
            data = target.read_bytes()
        else:
            with urllib.request.urlopen(source, timeout=60) as response:
                data = response.read(asset["size_bytes"] + 1)
        sha = verify(data, asset)
        if not target.exists():
            target.write_bytes(data)
        manifest.append({
            "id": asset["id"], "revision": asset["revision"],
            "tokenizer_path": "../.cache/tokenizers/" + asset["filename"],
            "sha256": sha, "size_bytes": len(data), "license": asset["license"],
            "license_url": f"https://huggingface.co/{asset['id']}/blob/{asset['revision']}/LICENSE",
            "source": source,
        })
        print(asset["id"], len(data), sha)
    lock = ROOT / "benchmarks" / "tokenizers.lock.json"
    lock.parent.mkdir(exist_ok=True)
    lock.write_text(json.dumps({"tokenizers": manifest}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
