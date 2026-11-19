"""Count class occurrences in one or more directories of YOLO labels."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def count_labels(root: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for path in root.rglob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                counts[int(fields[0])] += 1
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label_directories", type=Path, nargs="+")
    args = parser.parse_args()
    total: Counter[int] = Counter()
    for directory in args.label_directories:
        current = count_labels(directory)
        total.update(current)
        print(f"{directory}: {sum(current.values())} annotations")
    for class_id, count in sorted(total.items()):
        print(f"class {class_id}: {count}")

