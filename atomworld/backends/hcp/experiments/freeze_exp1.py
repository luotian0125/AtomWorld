"""Create a verified, write-protected copy; never modify the source experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze(destination):
    destination = Path(destination).resolve()
    manifest = json.loads((destination / "freeze_manifest.json").read_text())
    for item in manifest["files"]:
        path = destination / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen file missing or changed: {path}")
    actual = {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()}
    expected = {item["path"] for item in manifest["files"]} | {"freeze_manifest.json"}
    if actual != expected:
        raise ValueError("Unexpected files in frozen snapshot")
    return manifest


def freeze(source, destination, root):
    source, destination, root = map(lambda p: Path(p).resolve(), (source, destination, root))
    if destination == source or source in destination.parents:
        raise ValueError("Snapshot must be outside the source experiment")
    if destination.exists():
        manifest = verify_freeze(destination)
        if manifest["source"] != str(source):
            raise ValueError("Existing freeze has a different source")
        return manifest
    if not (source / "evaluation/exp1_results.json").is_file():
        raise ValueError("Source is not a completed Exp1 experiment")
    destination.mkdir(parents=True, exist_ok=False)
    files = []
    candidates = [(p, Path("experiment") / p.relative_to(source))
                  for p in source.rglob("*") if p.is_file()]
    for folder in ("RL4KMC", "configs", "experiments", "scripts"):
        candidates += [(p, Path("code_at_freeze") / p.relative_to(root))
                       for p in (root / folder).rglob("*")
                       if p.is_file() and p.suffix in {".py", ".json", ".sh"}]
    for origin, relative in sorted(candidates):
        if origin.is_symlink():
            raise ValueError(f"Refusing symlink in snapshot: {origin}")
        before = sha256(origin)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        if before != sha256(target) or before != sha256(origin):
            raise RuntimeError(f"Source changed during freeze: {origin}")
        files.append({"path": relative.as_posix(), "sha256": before, "bytes": target.stat().st_size})
    manifest = {
        "schema_version": 1, "source": str(source),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_provenance": "code present at freeze time; NOT proof of original run code revision",
        "files": files,
    }
    path = destination / "freeze_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_freeze(destination)
    for p in destination.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = freeze(args.source, args.destination, args.root)
    print(json.dumps({"status": "verified", "files": len(result["files"]), "source": result["source"]}))
