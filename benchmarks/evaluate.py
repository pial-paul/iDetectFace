"""Evaluate the live matching path on labeled single-person photographs."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

from idetectface.identity import identity_key
from idetectface.images import read_image
from idetectface.vision import Gallery

UNKNOWN = "@unknown"


def evaluate(directory, manifest, threshold):
    if not directory.is_dir():
        raise ValueError("Gallery directory does not exist.")
    gallery = Gallery.load(directory)
    enrolled = {identity_key(name) for name in gallery.names}
    if not enrolled:
        raise ValueError("The gallery needs a usable enrollment image.")
    seen = set(gallery.fingerprints)
    root = manifest.resolve().parent
    counts = Counter()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream, strict=True)
        if next(rows, None) != ["image", "expected"]:
            raise ValueError("The manifest header must be image,expected.")
        for row in rows:
            if len(row) != 2 or not all(value.strip() for value in row):
                raise ValueError(f"Incomplete manifest row at line {rows.line_num}.")
            filename, expected = row
            expected = identity_key(expected)
            if expected != UNKNOWN and expected not in enrolled:
                raise ValueError(
                    f"Expected identity at line {rows.line_num} is absent from the loaded gallery."
                )
            relative = Path(filename)
            if relative.anchor or ".." in relative.parts:
                raise ValueError(
                    f"Line {rows.line_num} must name a file inside the manifest folder."
                )
            path = root / relative
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Line {rows.line_num} resolves outside the manifest folder.")
            rgb, digest = read_image(path)
            if digest in seen:
                raise ValueError(f"Line {rows.line_num} reuses an enrollment or probe file.")
            seen.add(digest)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            matches = gallery.recognize(frame, threshold)
            cohort = "unknown" if expected == UNKNOWN else "known"
            counts[cohort] += 1
            if len(matches) != 1:
                counts[f"{cohort}_detection_failure"] += 1
                continue
            name = matches[0].name
            if expected == UNKNOWN:
                outcome = "correct_reject" if name is None else "false_accept"
            elif name is None:
                outcome = "false_reject"
            else:
                outcome = "correct_id" if identity_key(name) == expected else "wrong_id"
            counts[outcome] += 1
    if not counts:
        raise ValueError("The manifest contains no probe images.")
    return counts


def summary(counts):
    groups = {
        "known": ("correct_id", "wrong_id", "false_reject", "known_detection_failure"),
        "unknown": ("correct_reject", "false_accept", "unknown_detection_failure"),
    }
    for group, outcomes in groups.items():
        total = counts[group]
        print(f"{group}: {total} probes")
        for outcome in outcomes:
            rate = f"{counts[outcome] / total:.1%}" if total else "n/a"
            print(f"  {outcome:<26} {counts[outcome]:>4} / {total}  ({rate})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("Require 0 < --threshold <= 1.")
    try:
        summary(evaluate(args.gallery, args.probes, args.threshold))
    except (ValueError, OSError, csv.Error, cv2.error) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
