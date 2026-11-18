"""Check every file in this package against CHECKSUMS.sha256.   python verify_checksums.py"""
import hashlib, sys
from pathlib import Path
root = Path(__file__).resolve().parent
bad = missing = ok = 0
for line in (root / "CHECKSUMS.sha256").read_text().splitlines():
    h, name = line.split("  ", 1)
    p = root / name
    if not p.exists(): missing += 1; print("MISSING", name); continue
    if hashlib.sha256(p.read_bytes()).hexdigest() != h: bad += 1; print("CHANGED", name)
    else: ok += 1
print(f"{ok} files match, {bad} changed, {missing} missing")
sys.exit(1 if bad or missing else 0)
