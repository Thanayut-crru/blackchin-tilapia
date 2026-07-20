from pathlib import Path


ROOT = Path(__file__).resolve().parent
targets = [ROOT / ".zenodo.json", ROOT / "CITATION.cff", ROOT / "PRE_UPLOAD_CHECKLIST.txt"]
problems = []
for path in targets:
    text = path.read_text(encoding="utf-8")
    if "REPLACE BEFORE UPLOAD" in text:
        problems.append(f"placeholder remains in {path.name}")
if "[ ]" in (ROOT / "PRE_UPLOAD_CHECKLIST.txt").read_text(encoding="utf-8"):
    problems.append("PRE_UPLOAD_CHECKLIST.txt still contains unchecked items")
if problems:
    print("NOT READY TO PUBLISH:")
    for problem in problems:
        print("-", problem)
    raise SystemExit(2)
print("PASS: no metadata placeholders or unchecked pre-upload items remain.")
