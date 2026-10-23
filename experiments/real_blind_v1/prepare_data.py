"""Prepare predeclared real-data splits; never run or score a prediction model."""

import csv
from datetime import datetime
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
RAW = ROOT / "outputs" / "real_blind_v1_raw"
SEALED = ROOT / "outputs" / "real_blind_v1_sealed"
SEED = 20261023
SOURCES = {
    "concrete_slump": {
        "uci_id": 182,
        "slug": "concrete+slump+test",
        "doi": "10.24432/C5FG7D",
        "attribution": "I-Cheng Yeh (2007)",
        "license": "CC BY 4.0",
        "member": "slump_test.data",
        "zip_sha256": "68d6a6813261bf8ecf79d8d7cc82b665bd5e9e6800c7705a16e671688c78e491",
        "member_sha256": "78cea9fbb28ad8fc17f389e216e724dc04642c6c363143e67f6bcb97e74525ee",
        "feature_names": ["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr.", "Fine Aggr."],
        "target_names": ["SLUMP(cm)", "FLOW(cm)", "Compressive Strength (28-day)(Mpa)"],
        "feature_unit_labels": ["kg/m^3"] * 7,
        "target_unit_labels": ["cm", "cm", "MPa"],
        "split_kind": "random_rows",
        "limitation": "103 laboratory mixtures; production-batch independence is not established.",
    },
    "parkinsons": {
        "uci_id": 189,
        "slug": "parkinsons+telemonitoring",
        "doi": "10.24432/C5ZS3N",
        "attribution": "Athanasios Tsanas and Max Little (2009)",
        "license": "CC BY 4.0",
        "member": "parkinsons_updrs.data",
        "zip_sha256": "2f82bb0ef96fa7d8d7edf4d97b89173e07c2cca7723440a64bc38918a83345df",
        "member_sha256": "f2c7d5025dec4e92e7feae367a5f7ccf58789a10ac6b54bdf15976c599f9dd39",
        "feature_names": ["Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP",
                          "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11",
                          "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"],
        "target_names": ["motor_UPDRS", "total_UPDRS"],
        "feature_unit_labels": ["percent notation in source", "not specified", "not specified", "not specified",
                                "not specified", "not specified", "dB", "not specified", "not specified",
                                "not specified", "not specified", "not specified", "not specified",
                                "not specified", "not specified", "not specified"],
        "target_unit_labels": ["UPDRS score", "UPDRS score"],
        "split_kind": "subject_groups",
        "limitation": "Real voice recordings; both clinician UPDRS labels are linearly interpolated and correlated.",
    },
    "air_quality": {
        "uci_id": 360,
        "slug": "air+quality",
        "doi": "10.24432/C59K5F",
        "attribution": "Saverio De Vito (2008)",
        "license": "UCI CC BY 4.0 banner conflicts with legacy research-only/no-commercial statement; local research only.",
        "member": "AirQualityUCI.csv",
        "zip_sha256": "d4a64013fb385288a8a48d9d193ca7079b2e1bbddf6f8d458feb8c08ab2b8a2a",
        "member_sha256": "13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996",
        "feature_names": ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)",
                          "PT08.S5(O3)", "T", "RH", "AH"],
        "target_names": ["CO(GT)", "NOx(GT)", "NO2(GT)"],
        "feature_unit_labels": ["sensor response; unit unspecified"] * 5 + ["degree Celsius", "percent", "unit unspecified"],
        "target_unit_labels": ["mg/m^3", "ppb", "microgram/m^3"],
        "split_kind": "chronological",
        "limitation": "Single-site sensor drift, cross-sensitivity, serial dependence, complete-case selection and unresolved license conflict.",
    },
}


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing artifact differs; refusing to overwrite: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)


def source_bytes(key, source):
    path = RAW / f"{key}.zip"
    if not path.exists():
        url = f"https://archive.ics.uci.edu/static/public/{source['uci_id']}/{source['slug']}.zip"
        with urlopen(url, timeout=45) as response:
            content = response.read(20_000_001)
        if len(content) > 20_000_000 or sha256(content) != source["zip_sha256"]:
            raise ValueError(f"Unexpected source download: {key}")
        write_once(path, content)
    content = path.read_bytes()
    if sha256(content) != source["zip_sha256"]:
        raise ValueError(f"Cached source hash mismatch: {key}")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        member = archive.read(source["member"])
    if sha256(member) != source["member_sha256"]:
        raise ValueError(f"Source member hash mismatch: {key}")
    return member


def number(value):
    try:
        return float(value.replace(",", "."))
    except (AttributeError, ValueError):
        return float("nan")


def read_rows(key, source):
    reader = csv.DictReader(io.StringIO(source_bytes(key, source).decode("utf-8-sig")),
                            delimiter=";" if key == "air_quality" else ",")
    columns = source["feature_names"] + source["target_names"]
    if not set(columns).issubset(reader.fieldnames):
        raise ValueError(f"Source schema changed: {key}")
    counts = dict(source_rows=0, invalid_identifier_or_time=0, duplicate_timestamps=0, incomplete_selected_columns=0)
    records, seen_times = [], set()
    for row_number, row in enumerate(reader, start=2):
        counts["source_rows"] += 1
        row_id = f"{source['member']}:line:{row_number}"
        order = row_number
        if key == "air_quality":
            try:
                timestamp = datetime.strptime(f"{row['Date']} {row['Time']}", "%d/%m/%Y %H.%M.%S")
            except (KeyError, ValueError):
                counts["invalid_identifier_or_time"] += 1
                continue
            if timestamp in seen_times:
                counts["duplicate_timestamps"] += 1
                continue
            seen_times.add(timestamp)
            group = timestamp.date().isoformat()
            order = timestamp.isoformat()
        elif key == "parkinsons":
            subject = number(row.get("subject#"))
            if not np.isfinite(subject) or subject < 1 or int(subject) != subject:
                counts["invalid_identifier_or_time"] += 1
                continue
            group = int(subject)
        else:
            group = row_id
        values = np.array([number(row.get(name)) for name in columns], dtype=float)
        if not np.isfinite(values).all() or (key == "air_quality" and np.any(values == -200)):
            counts["incomplete_selected_columns"] += 1
            continue
        records.append((order, row_id, group, values))
    records.sort(key=lambda record: record[0])
    if len(records) < 5:
        raise ValueError(f"Too few eligible rows: {key}")
    values = np.array([record[3] for record in records])
    ids = np.array([record[1] for record in records])
    groups = np.array([record[2] for record in records])
    assert len(set(ids)) == len(ids)
    counts["eligible_rows"] = len(records)
    assert sum(counts[k] for k in ("invalid_identifier_or_time", "duplicate_timestamps", "incomplete_selected_columns", "eligible_rows")) == counts["source_rows"]
    n_features = len(source["feature_names"])
    return values[:, :n_features], values[:, n_features:], ids, groups, counts


def split_indices(key, ids, groups):
    rng = np.random.default_rng(SEED)
    if key == "parkinsons":
        subjects = rng.permutation(np.unique(groups))
        cuts = [int(.6 * len(subjects)), int(.8 * len(subjects))]
        splits = [np.flatnonzero(np.isin(groups, part)) for part in np.split(subjects, cuts)]
    else:
        indices = rng.permutation(len(ids)) if key == "concrete_slump" else np.arange(len(ids))
        splits = list(np.split(indices, [int(.6 * len(ids)), int(.8 * len(ids))]))
    assert all(len(indices) > 0 for indices in splits)
    assert sorted(np.concatenate(splits).tolist()) == list(range(len(ids)))
    if key != "air_quality":
        group_sets = [set(groups[indices]) for indices in splits]
        assert all(not group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3))
    else:
        assert splits[0][-1] < splits[1][0] <= splits[1][-1] < splits[2][0]
    return dict(zip(("train", "val", "test"), splits))


def array_bytes(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def ordered_id_hash(ids):
    return sha256(("\n".join(ids.tolist()) + "\n").encode("utf-8"))


def main():
    manifest = dict(protocol="real_blind_v1", seed=SEED, source_checked_date="2026-10-23",
                    split_rule="Cuts floor(0.6*N), floor(0.8*N); N=rows except Parkinsons subject groups.",
                    units_policy="Original published values and labels; no scaling or inferred SI units.",
                    python_version=sys.version.split()[0], numpy_version=np.__version__,
                    prepare_script_sha256=sha256(Path(__file__).read_bytes()), datasets={})
    for key, source in SOURCES.items():
        X, Y, ids, groups, counts = read_rows(key, source)
        split = split_indices(key, ids, groups)
        arrays, sealed = {}, {}
        for name, indices in split.items():
            destination = sealed if name == "test" else arrays
            destination.update({f"X_{name}": X[indices], f"Y_{name}": Y[indices],
                                f"{name}_ids": ids[indices], f"{name}_groups": groups[indices]})
        public_path = HERE / "data" / f"{key}.npz"
        sealed_path = SEALED / f"{key}.npz"
        public_bytes, sealed_bytes = array_bytes(arrays), array_bytes(sealed)
        write_once(public_path, public_bytes)
        write_once(sealed_path, sealed_bytes)
        metadata = {field: value for field, value in source.items() if field not in ("slug",)}
        metadata.update(
            source_url=f"https://archive.ics.uci.edu/dataset/{source['uci_id']}/{source['slug']}",
            download_url=f"https://archive.ics.uci.edu/static/public/{source['uci_id']}/{source['slug']}.zip",
            counts=counts,
            split_counts={name: len(indices) for name, indices in split.items()},
            split_group_counts={name: len(np.unique(groups[indices])) for name, indices in split.items()},
            split_id_sha256={name: ordered_id_hash(ids[indices]) for name, indices in split.items()},
            train_bounds=np.column_stack((arrays["X_train"].min(axis=0), arrays["X_train"].max(axis=0))).tolist(),
            trainval_file=str(public_path.relative_to(ROOT).as_posix()),
            sealed_file=str(sealed_path.relative_to(ROOT).as_posix()),
            trainval_sha256=sha256(public_bytes), sealed_sha256=sha256(sealed_bytes),
        )
        manifest["datasets"][key] = metadata
        print(f"{key}: prepared train/validation/test counts {metadata['split_counts']}; test targets sealed")
    content = (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    write_once(HERE / "data" / "manifest.json", content)
    print("Preparation complete. No model fitted, no test values or test performance inspected.")


if __name__ == "__main__":
    main()
