"""Checks #106's second capture export against the first. REAL DATA.

    python experiments/m5_01/verify_reexport.py <first export dir> <second export dir>

Exits non zero unless every window of the second equals the first's with the direction fields
removed, each direction list is as long as its references, and the fits and the summary are
equal. Of the directions it reads only the counts. It changes no file.
"""

import json
import pathlib
import sys

NEW = "reference_directions"


def strip(window):
    """The window without its directions, and whether their counts match the references."""
    window = dict(window)
    history = dict(window["history"])
    directions = window.pop(NEW), history.pop(NEW)
    window["history"] = history
    return window, (len(directions[0]) == len(window["references"])
                    and len(directions[1]) == len(history["references"]))


def differences(first, second):
    """The relative paths on which the exports differ, or are missing."""
    first, second = pathlib.Path(first), pathlib.Path(second)
    bad = []
    names = sorted(p.relative_to(first) for p in first.rglob("window-*.json"))
    theirs = sorted(p.relative_to(second) for p in second.rglob("window-*.json"))
    if names != theirs or not names:
        bad.append("the window files differ")
    for name in names:
        if not (second / name).exists():
            continue
        old = json.loads((first / name).read_text())
        new, counts = strip(json.loads((second / name).read_text()))
        if old != new or not counts:
            bad.append(str(name))
    for name in ("fits.json", "summary.json"):
        if json.loads((first / name).read_text()) != json.loads((second / name).read_text()):
            bad.append(name)
    return bad


def main():
    bad = differences(sys.argv[1], sys.argv[2])
    if bad:
        raise SystemExit(f"the exports differ: {bad}")
    print("the exports agree apart from the directions")


if __name__ == "__main__":
    main()
