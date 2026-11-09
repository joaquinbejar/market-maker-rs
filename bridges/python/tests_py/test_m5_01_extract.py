"""#106's extraction on made-up archives (no capture file): the design's hours, the member
names, the manifest check, and the refusals."""

import gzip
import hashlib
import io
import json
import pathlib
import sys
import tarfile
from datetime import datetime, timezone

import pytest

EXPERIMENT = pathlib.Path(__file__).resolve().parent.parent / "experiments" / "m5_01"
sys.path.insert(0, str(EXPERIMENT))
import extract  # noqa: E402

HOURS = [("20251221", 23), ("20251222", 0)]


def content(path):
    return f"made-up {path}\n".encode()


def build(tmp_path, corrupt=None, drop=None, prefix=""):
    """Three made-up archives holding the two hours, plus distractors, and their manifest."""
    archives = tmp_path / "archives"
    archives.mkdir()
    manifest = {}
    for kind, archive in extract.ARCHIVES.items():
        mode = "w:xz" if archive.endswith(".xz") else "w"
        with tarfile.open(archives / archive, mode) as tar:
            members = extract.wanted(HOURS)[kind]
            members[f"20251220/{'ex1.gz' if kind == 'd' else '1.gz'}"] = "distractor"
            for name, path in members.items():
                if name == drop:
                    continue
                data = content(path)
                if path != "distractor":
                    manifest[path] = (len(data), hashlib.sha256(data).hexdigest())
                payload = gzip.compress(b"tampered\n" if path == corrupt else data)
                info = tarfile.TarInfo(prefix + name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return archives, manifest


def test_the_design_hours_are_220_from_12_21_h15_to_12_30_h18():
    hours = extract.hours()
    assert len(hours) == 220
    assert hours[0] == ("20251221", 15) and hours[-1] == ("20251230", 18)
    design = json.loads((EXPERIMENT / "design.json").read_text())
    assert "220 hours" in design["main_study"]["data"]["members"]


def test_the_member_names_map_to_the_manifest_paths():
    w = extract.wanted([("20251222", 7)])
    assert w["o"] == {"20251222/btc_07.data.gz": "o/20251222/btc_07.data"}
    assert w["d"] == {"20251222/ex7.gz": "d/20251222/ex7"}
    assert w["t"] == {"20251222/7.gz": "t/20251222/7"}
    assert extract.member_key("./book/20251222/ex7.gz", w["d"]) == "20251222/ex7.gz"
    assert extract.member_key("20251222/ex17.gz", w["d"]) is None


@pytest.mark.parametrize("prefix", ["", "./", "top/"])
def test_the_wanted_members_are_gunzipped_and_checked(tmp_path, prefix):
    archives, manifest = build(tmp_path, prefix=prefix)
    record = extract.extract(archives, manifest, tmp_path / "data", HOURS)
    assert len(record["files"]) == 6
    for path, sha in record["files"].items():
        data = (tmp_path / "data" / path).read_bytes()
        assert data == content(path) and hashlib.sha256(data).hexdigest() == sha
    assert not (tmp_path / "data" / "d" / "20251220").exists()
    assert json.loads((tmp_path / "data" / "checked.json").read_text())["files"] == record["files"]


def test_a_file_that_differs_from_the_manifest_is_deleted_and_refused(tmp_path):
    archives, manifest = build(tmp_path, corrupt="d/20251222/ex0")
    with pytest.raises(ValueError, match="differs from #76's manifest"):
        extract.extract(archives, manifest, tmp_path / "data", HOURS)
    assert not (tmp_path / "data" / "d" / "20251222" / "ex0").exists()
    assert not (tmp_path / "data" / "checked.json").exists()


def test_a_missing_member_is_refused(tmp_path):
    archives, manifest = build(tmp_path, drop="20251221/23.gz")
    with pytest.raises(ValueError, match="lacks 1 wanted members"):
        extract.extract(archives, manifest, tmp_path / "data", HOURS)


def test_a_manifest_other_than_76s_is_refused(tmp_path):
    other = tmp_path / "manifest.tsv"
    other.write_text("d/20251222/ex0\t1\tab\n")
    with pytest.raises(ValueError, match="not #76's manifest"):
        extract.read_manifest(other)
    assert extract.read_manifest(other, extract.sha256_file(other)) == {
        "d/20251222/ex0": (1, "ab")}


def test_the_archives_are_only_read(tmp_path):
    archives, manifest = build(tmp_path)
    before = {p.name: (p.stat().st_mtime_ns, extract.sha256_file(p)) for p in archives.iterdir()}
    extract.extract(archives, manifest, tmp_path / "data", HOURS)
    after = {p.name: (p.stat().st_mtime_ns, extract.sha256_file(p)) for p in archives.iterdir()}
    assert before == after
    assert datetime(2025, 12, 21, 15, tzinfo=timezone.utc) == extract.FIRST_HOUR
