#!/usr/bin/env python3
"""Compare generated projection/eva files against the checked-in target files."""

from __future__ import annotations

import filecmp
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
METHODS = ["IBM2", "LTP", "Levenshtein", "step1", "step2"]


def compare(label: str, generated: Path, target: Path) -> bool:
    if not generated.exists():
        print(f"{label}: missing generated file {generated}")
        return False
    ok = filecmp.cmp(generated, target, shallow=False)
    print(f"{label}: {'OK' if ok else 'DIFF'}")
    return ok


def main() -> None:
    all_ok = True
    for method in METHODS:
        all_ok &= compare(
            f"{method} projection",
            BASE / "generated_csv" / f"{method}_projection.csv",
            BASE / "csv" / f"{method}_projection.csv",
        )
        all_ok &= compare(
            f"{method} eva",
            BASE / "generated_eva" / f"eva_{method}.csv",
            BASE / "eva" / f"eva_{method}.csv",
        )
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
