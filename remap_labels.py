"""Remap class IDs in YOLO label files without changing the source directory."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_mapping(items: list[str]) -> dict[int, int]:
    mapping = {}
    for item in items:
        old, new = item.split(":", maxsplit=1)
        mapping[int(old)] = int(new)
    return mapping


def remap_file(source: Path, destination: Path, mapping: dict[int, int]) -> tuple[int, int]:
    kept, skipped, output = 0, 0, []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if len(fields) < 5 or int(fields[0]) not in mapping:
            skipped += 1
            continue
        fields[0] = str(mapping[int(fields[0])])
        output.append(" ".join(fields))
        kept += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output) + ("\n" if output else ""), encoding="utf-8")
    return kept, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--mapping", nargs="+", required=True, metavar="OLD:NEW")
    args = parser.parse_args()
    mapping = parse_mapping(args.mapping)
    totals = [0, 0]
    for source_file in sorted(args.source.rglob("*.txt")):
        kept, skipped = remap_file(
            source_file, args.destination / source_file.relative_to(args.source), mapping
        )
        totals[0] += kept
        totals[1] += skipped
    print(f"kept {totals[0]} annotations; skipped {totals[1]}")


if __name__ == "__main__":
    main()

