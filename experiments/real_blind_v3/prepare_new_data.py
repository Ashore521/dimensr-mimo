"""Prepare two newly selected real-data holdouts without fitting or scoring models."""

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.request import urlopen
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RAW = ROOT / "outputs" / "real_blind_v3_raw"
SEALED = ROOT / "outputs" / "real_blind_v3_sealed"
SEED = 20261024
SOURCES = {
    "gas_turbine": {
        "uci_id": 551, "slug": "gas+turbine+co+and+nox+emission+data+set",
        "doi": "10.24432/C5WC95", "attribution": "UCI Gas Turbine CO and NOx Emission Data Set (2019)",
        "license": "CC BY 4.0", "zip_sha256": "55fdb1acc25f05bd9c77aac285c424e9154ffb0dbf1bbadb56b69a1142231295",
        "members_sha256": {
            "gt_2011.csv": "d87ceef9aa59533cc7d924d10de241b1b06ecd11f9b26bab59191ea0f8a76b9a",
            "gt_2012.csv": "be54b9d0e1a7de40c55d32fa489e75de892b000c066b5a09f09a19124ee29100",
            "gt_2013.csv": "13c437bb440ec2045bd12057e6654c41dd4107a661eac16ba2e878e897a08f9e",
            "gt_2014.csv": "c2a03c92c9c3207aad0c6be7de8d9b5b4bfa4720ad0efb2c1f21b6cec4d3f3fa",
            "gt_2015.csv": "9b08f35fde0d4b138232a605db4093c2b8bf9d6757e6f1fbd9534ad616c13591",
        },
        "feature_names": ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"],
        "target_names": ["CO", "NOX"],
        "feature_unit_labels": ["degree Celsius", "mbar (source label)", "percent", "mbar (source label)",
                                "mbar (source label)", "degree Celsius", "degree Celsius", "MWH (source label)", "mbar (source label)"],
        "target_unit_labels": ["mg/m^3", "mg/m^3"],
        "split_rule": "2011-2013 train; 2014 validation; 2015 sealed test; preserve published row order.",
        "group_rule": "Year-prefixed nonoverlapping blocks of 24 original records; not asserted calendar days.",
        "limitation": "Single plant, concurrent covariates, no individual timestamps; 2014 validation differs from original paper's 2014-2015 test protocol.",
    },
    "sml2010": {
        "uci_id": 274, "slug": "sml2010", "doi": "10.24432/C5RS3S",
        "attribution": "Pablo Romeu-Guallart and Francisco Zamora-Martinez (2014)", "license": "CC BY 4.0",
        "zip_sha256": "c270da7bca968dba4c117c3580b56d49a5031afea5cbf483951aa21f174545fa",
        "members_sha256": {
            "NEW-DATA-1.T15.txt": "bd5a4c4f1f6a0652258179c9963fb5c90885eedee4a981617898c18d6b5c4541",
            "NEW-DATA-2.T15.txt": "1c008643381903a51557a0599cbd7706ae71b07f2d449d165b17f13b6e12a97e",
        },
        "feature_names": ["Weather_Temperature", "CO2_Comedor_Sensor", "CO2_Habitacion_Sensor",
                          "Humedad_Comedor_Sensor", "Humedad_Habitacion_Sensor", "Lighting_Comedor_Sensor",
                          "Lighting_Habitacion_Sensor", "Precipitacion", "Meteo_Exterior_Crepusculo",
                          "Meteo_Exterior_Viento", "Meteo_Exterior_Sol_Oest", "Meteo_Exterior_Sol_Est",
                          "Meteo_Exterior_Sol_Sud", "Meteo_Exterior_Piranometro", "Exterior_Entalpic_1",
                          "Exterior_Entalpic_2", "Exterior_Entalpic_turbo", "Temperature_Exterior_Sensor",
                          "Humedad_Exterior_Sensor"],
        "target_names": ["Temperature_Comedor_Sensor", "Temperature_Habitacion_Sensor"],
        "feature_unit_labels": ["degree Celsius", "ppm", "ppm", "percent", "percent", "lux", "lux",
                                "rain fraction", "dusk; unit unspecified", "m/s", "lux", "lux", "lux",
                                "W/m^2", "binary on/off", "binary on/off", "binary on/off", "degree Celsius", "percent"],
        "target_unit_labels": ["degree Celsius", "degree Celsius"],
        "split_rule": "Sort eligible UTC dates; whole-day cuts floor(0.6*N_dates), floor(0.8*N_dates); preserve chronological records.",
        "group_rule": "UTC calendar day; no date crosses splits.",
        "limitation": "Synchronous indoor thermal-state reconstruction, not forecasting; one house, 15-minute averaging, serial dependence and unknown forecast-covariate issuance history.",
    },
}


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to overwrite an incompatible artifact: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)


def archive_members(key, source):
    path = RAW / f"{key}.zip"
    if not path.exists():
        url = f"https://archive.ics.uci.edu/static/public/{source['uci_id']}/{source['slug']}.zip"
        with urlopen(url, timeout=45) as response:
            content = response.read(20_000_001)
        if len(content) > 20_000_000 or sha256(content) != source["zip_sha256"]:
            raise ValueError(f"Unexpected downloaded source: {key}")
        write_once(path, content)
    content = path.read_bytes()
    if sha256(content) != source["zip_sha256"]:
        raise ValueError(f"Raw archive hash mismatch: {key}")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = {name: archive.read(name) for name in source["members_sha256"]}
    if any(sha256(value) != source["members_sha256"][name] for name, value in members.items()):
        raise ValueError(f"Raw member hash mismatch: {key}")
    return members


def finite_values(fields):
    try:
        values = np.array([float(value) for value in fields], dtype=float)
    except (ValueError, TypeError):
        return None
    return values if np.isfinite(values).all() else None


def read_gas(source, members):
    records = {"train": [], "val": [], "test": []}
    counts = dict(source_rows=0, incomplete_selected_columns=0)
    for name, content in sorted(members.items()):
        year = int(name[3:7])
        part = "train" if year <= 2013 else "val" if year == 2014 else "test"
        reader = csv.reader(io.StringIO(content.decode("utf-8-sig")))
        if next(reader) != source["feature_names"] + source["target_names"]:
            raise ValueError(f"Unexpected source schema: {name}")
        for row_number, row in enumerate(reader, start=2):
            counts["source_rows"] += 1
            values = finite_values(row) if len(row) == 11 else None
            if values is None:
                counts["incomplete_selected_columns"] += 1
                continue
            record_id = f"{name}:line:{row_number}"
            group = f"{year}:record_block24:{(row_number - 2) // 24:04d}"
            records[part].append((record_id, group, values[:9], values[9:]))
    counts["eligible_rows"] = sum(len(rows) for rows in records.values())
    assert counts["source_rows"] == counts["incomplete_selected_columns"] + counts["eligible_rows"]
    return records, counts


def read_sml(source, members):
    records, seen = [], set()
    counts = dict(source_rows=0, invalid_timestamp_or_columns=0, duplicate_timestamps=0, incomplete_selected_columns=0)
    expected_header = ["Date", "Time"] + source["target_names"] + source["feature_names"] + ["Day_Of_Week"]
    for name, content in sorted(members.items()):
        lines = content.decode("utf-8-sig").splitlines()
        header = [token.split(":", 1)[1] for token in lines[0].lstrip("#").split()]
        if header != expected_header:
            raise ValueError(f"Unexpected source schema: {name}")
        for row_number, line in enumerate(lines[1:], start=2):
            if not line.strip() or line.startswith("#"):
                continue
            counts["source_rows"] += 1
            row = line.split()
            if len(row) != 24:
                counts["invalid_timestamp_or_columns"] += 1
                continue
            try:
                timestamp = datetime.strptime(" ".join(row[:2]), "%d/%m/%Y %H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                counts["invalid_timestamp_or_columns"] += 1
                continue
            if timestamp in seen:
                counts["duplicate_timestamps"] += 1
                continue
            seen.add(timestamp)
            values = finite_values(row[2:23])
            if values is None:
                counts["incomplete_selected_columns"] += 1
                continue
            records.append((timestamp, f"{name}:line:{row_number}", timestamp.date().isoformat(), values[2:], values[:2]))
    records.sort(key=lambda record: record[0])
    dates = sorted({record[2] for record in records})
    if len(dates) < 5:
        raise ValueError("Too few eligible UTC dates for SML2010.")
    a, b = int(.6 * len(dates)), int(.8 * len(dates))
    labels = dict(zip(dates, ["train"] * a + ["val"] * (b - a) + ["test"] * (len(dates) - b)))
    splits = {name: [] for name in ("train", "val", "test")}
    for _, record_id, group, X, Y in records:
        splits[labels[group]].append((record_id, group, X, Y))
    counts["eligible_rows"] = len(records)
    counts["eligible_utc_dates"] = len(dates)
    assert sum(counts[name] for name in ("invalid_timestamp_or_columns", "duplicate_timestamps", "incomplete_selected_columns", "eligible_rows")) == counts["source_rows"]
    return splits, counts


def npz_bytes(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def prepare(key, source):
    members = archive_members(key, source)
    records, counts = read_gas(source, members) if key == "gas_turbine" else read_sml(source, members)
    arrays, sealed, id_hashes = {}, {}, {}
    seen_ids, seen_groups = set(), set()
    for part, rows in records.items():
        if not rows:
            raise ValueError(f"Empty split: {key}/{part}")
        ids = np.array([row[0] for row in rows])
        groups = np.array([row[1] for row in rows])
        assert len(set(ids)) == len(ids) and not set(ids) & seen_ids and not set(groups) & seen_groups
        seen_ids.update(ids)
        seen_groups.update(groups)
        X, Y = np.array([row[2] for row in rows]), np.array([row[3] for row in rows])
        assert X.shape == (len(rows), len(source["feature_names"]))
        assert Y.shape == (len(rows), len(source["target_names"]))
        assert np.isfinite(X).all() and np.isfinite(Y).all()
        destination = sealed if part == "test" else arrays
        destination.update({f"X_{part}": X, f"Y_{part}": Y, f"{part}_ids": ids, f"{part}_groups": groups})
        id_hashes[part] = sha256(("\n".join(ids.tolist()) + "\n").encode())
    trainval_path, sealed_path = HERE / "new_data" / f"{key}.npz", SEALED / f"{key}.npz"
    trainval_content, sealed_content = npz_bytes(arrays), npz_bytes(sealed)
    write_once(trainval_path, trainval_content)
    write_once(sealed_path, sealed_content)
    metadata = {name: value for name, value in source.items() if name != "slug"}
    metadata.update(
        source_url=f"https://archive.ics.uci.edu/dataset/{source['uci_id']}/{source['slug']}",
        download_url=f"https://archive.ics.uci.edu/static/public/{source['uci_id']}/{source['slug']}.zip",
        counts=counts, split_counts={part: len(rows) for part, rows in records.items()},
        split_group_counts={part: len({row[1] for row in rows}) for part, rows in records.items()},
        split_id_sha256=id_hashes,
        train_bounds=np.column_stack((arrays["X_train"].min(axis=0), arrays["X_train"].max(axis=0))).tolist(),
        training_statistics={name: {"mean": arrays[name].mean(axis=0).tolist(), "std": arrays[name].std(axis=0).tolist(),
                                    "min": arrays[name].min(axis=0).tolist(), "max": arrays[name].max(axis=0).tolist()}
                             for name in ("X_train", "Y_train")},
        trainval_file=trainval_path.relative_to(ROOT).as_posix(), sealed_file=sealed_path.relative_to(ROOT).as_posix(),
        trainval_sha256=sha256(trainval_content), sealed_sha256=sha256(sealed_content),
    )
    return metadata


def main():
    manifest = dict(protocol="real_blind_v3", baseline_commit="cfea3b1", seed=SEED,
                    split_randomness="No shuffling; chronological/year/day rules fixed before model fitting.",
                    units_policy="Published raw values and source labels; no inferred unit conversions or normalization.",
                    python_version=sys.version.split()[0], numpy_version=np.__version__,
                    prepare_script_sha256=sha256(Path(__file__).read_bytes()), datasets={})
    for key, source in SOURCES.items():
        manifest["datasets"][key] = prepare(key, source)
        print(f"{key}: split counts {manifest['datasets'][key]['split_counts']}; test targets sealed")
    write_once(HERE / "new_data" / "manifest.json",
               (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    print("Complete: no model fitted and no test values, target summaries or performance displayed.")


if __name__ == "__main__":
    main()
