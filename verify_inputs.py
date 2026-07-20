from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "metadata" / "input_data_manifest.csv"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


missing = []
mismatch = []
with MANIFEST.open(encoding="utf-8", newline="") as fh:
    for row in csv.DictReader(fh):
        path = ROOT / row["path"]
        if not path.exists():
            missing.append(row["path"])
        elif digest(path) != row["sha256"]:
            mismatch.append(row["path"])

if missing or mismatch:
    if missing:
        print("Missing inputs:", *missing, sep="\n- ")
    if mismatch:
        print("Hash mismatches:", *mismatch, sep="\n- ")
    raise SystemExit(1)
print("PASS: all model inputs exist and match the reported analysis hashes.")
