"""#106's export check, on made-up windows only (SYNTHETIC)."""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "experiments" / "m5_01"))
import verify_reexport  # noqa: E402


def window(extra=False, value=1, directions=1):
    out = {"samples": [value], "references": [{"moved": True}],
           "history": {"references": []}}
    if extra:
        out["reference_directions"] = [1] * directions
        out["history"]["reference_directions"] = []
    return out


def export(root, content):
    (root / "holdout").mkdir(parents=True)
    (root / "holdout" / "window-0.json").write_text(json.dumps(content))
    (root / "fits.json").write_text("{}")
    (root / "summary.json").write_text("{}")


def test_equal_apart_from_directions(tmp_path):
    export(tmp_path / "a", window())
    export(tmp_path / "b", window(True))
    assert verify_reexport.differences(tmp_path / "a", tmp_path / "b") == []


def test_another_field_differing_is_caught(tmp_path):
    export(tmp_path / "a", window())
    export(tmp_path / "b", window(True, value=2))
    assert verify_reexport.differences(tmp_path / "a", tmp_path / "b") != []


def test_a_short_direction_list_is_caught(tmp_path):
    export(tmp_path / "a", window())
    export(tmp_path / "b", window(True, directions=0))
    assert verify_reexport.differences(tmp_path / "a", tmp_path / "b") != []
