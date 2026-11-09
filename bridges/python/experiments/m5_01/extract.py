"""Extracts #106's capture hours from the raw archives, checking each file against #76's
manifest. REAL DATA when run on the capture; the tests run it on made-up archives only.

    python experiments/m5_01/extract.py <archive dir> <manifest.tsv> <data dir>

The hours are the design's (main_study.data.members): every hour of the epoch's replay,
each as three files, `o/<day>/btc_<hh>.data`, `d/<day>/ex<h>` and `t/<day>/<h>` under the data
dir, gunzipped from the archive members `<day>/btc_<hh>.data.gz`, `<day>/ex<h>.gz` and
`<day>/<h>.gz`. Each archive is read once as a stream and never written; each file's sha256
is computed as it is written and checked against the manifest, and a file that differs is
deleted and the run refused. It writes `checked.json` in the data dir, with the manifest's
own sha256 and every file's hash, which the replay step requires.
"""

import gzip
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
from datetime import datetime, timedelta, timezone

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST_SHA256 = "94752eb07b5a4ad101806654704764489b60e9916e3ff2af234597ba2554a337"
ARCHIVES = {"o": "btc_orders_202512.tar.xz", "d": "book_diffs_202512.tar",
            "t": "trades_2025_12.tar"}
FIRST_HOUR = datetime(2025, 12, 21, 15, tzinfo=timezone.utc)
LAST_HOUR = datetime(2025, 12, 30, 18, tzinfo=timezone.utc)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hours(first=FIRST_HOUR, last=LAST_HOUR):
    """Every hour from `first` to `last`, both included, as (day, hour)."""
    out = []
    at = first
    while at <= last:
        out.append((at.strftime("%Y%m%d"), at.hour))
        at += timedelta(hours=1)
    return out


def wanted(hour_list):
    """Archive member name (without any leading directory) -> manifest path, per archive."""
    out = {kind: {} for kind in ARCHIVES}
    for day, h in hour_list:
        out["o"][f"{day}/btc_{h:02d}.data.gz"] = f"o/{day}/btc_{h:02d}.data"
        out["d"][f"{day}/ex{h}.gz"] = f"d/{day}/ex{h}"
        out["t"][f"{day}/{h}.gz"] = f"t/{day}/{h}"
    return out


def read_manifest(path, expected_sha256=MANIFEST_SHA256):
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{path} is not #76's manifest")
    entries = {}
    for line in pathlib.Path(path).read_text().splitlines():
        if line.strip():
            name, size, sha = line.split("\t")
            entries[name] = (int(size), sha)
    return entries


def member_key(name, members):
    """The wanted key a member name ends with, after its last date directory."""
    parts = name.lstrip("./").split("/")
    for i in range(len(parts) - 1):
        key = "/".join(parts[i:])
        if key in members:
            return key
    return None


def extract(archive_dir, manifest, data_dir, hour_list=None, archives=ARCHIVES):
    """Streams each archive once, gunzips the wanted members into `data_dir` with their
    sha256 checked against `manifest`, and writes `checked.json`."""
    hour_list = hours() if hour_list is None else hour_list
    data_dir = pathlib.Path(data_dir)
    targets = wanted(hour_list)
    checked = {}
    for kind, archive in archives.items():
        members = targets[kind]
        found = set()
        with tarfile.open(pathlib.Path(archive_dir) / archive, mode="r|*") as tar:
            for info in tar:
                key = member_key(info.name, members) if info.isfile() else None
                if key is None:
                    continue
                path = members[key]
                if path not in manifest:
                    raise ValueError(f"{path} is not in the manifest")
                out = data_dir / path
                out.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with gzip.open(tar.extractfile(info)) as src, open(out, "wb") as dst:
                    for chunk in iter(lambda: src.read(1 << 20), b""):
                        digest.update(chunk)
                        size += len(chunk)
                        dst.write(chunk)
                expected_size, expected_sha = manifest[path]
                if digest.hexdigest() != expected_sha or size != expected_size:
                    out.unlink()
                    raise ValueError(f"{path} differs from #76's manifest")
                checked[path] = digest.hexdigest()
                found.add(key)
        missing = sorted(set(members) - found)
        if missing:
            raise ValueError(f"{archive} lacks {len(missing)} wanted members, e.g. {missing[0]}")
    record = {"label": "REAL DATA: Hyperliquid BTC, December 2025 (Zenodo 18184441)",
              "manifest_sha256": MANIFEST_SHA256, "hours": len(hour_list),
              "files": dict(sorted(checked.items()))}
    (data_dir / "checked.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def remove(data_dir):
    """Deletes the extracted data after the holdout, as the design says."""
    shutil.rmtree(data_dir)


def main():
    archive_dir, manifest_path, data_dir = sys.argv[1:4]
    design = json.loads((HERE / "design.json").read_text())
    if design["design_version"] < 2:
        raise SystemExit("the main study is fixed in design version 2")
    record = extract(archive_dir, read_manifest(manifest_path), data_dir)
    print(json.dumps({"hours": record["hours"], "files": len(record["files"])}))


if __name__ == "__main__":
    main()
