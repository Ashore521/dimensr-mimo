"""Retrospective scoring of fixed archives; never fits or opens new holdouts."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
from itertools import product
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr
from threadpoolctl import threadpool_limits
from dimensr_mimo import MIMOSymbolicRegressor
from dimensr_mimo.search import predict_candidate

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "experiments/contextual_transfer_v1"
sys.path.insert(0, str(SOURCE))
try:
    import context_study as source
finally:
    sys.path.pop(0)
from curves import decode, node_keys

METHODS, DATASETS = source.METHODS, source.DATASETS
sha, save = source.sha, source.save


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_freeze():
    frozen = read(HERE / "FREEZE.json")
    for path, digest in frozen["files"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    return frozen


def prepare():
    if (HERE / "FREEZE.json").exists():
        raise FileExistsError("Diagnosis already frozen; preserve it.")
    original = source.check_freeze()
    summary = read(SOURCE / "analysis/summary.json")
    assert summary["freeze_sha256"] == sha(SOURCE / "FREEZE.json")
    paths = [ROOT / p for p in original["files"]]
    paths += [SOURCE / "FREEZE.json", SOURCE / "analysis/summary.json", HERE / "PROTOCOL.md"]
    paths += list(HERE.glob("*.py")) + list((ROOT / "tests").glob("test_candidate_diagnosis*.py"))
    for job in source.jobs():
        out = SOURCE / "runs" / source.name(job)
        result = read(out / "result.json")
        assert sha(out / "result.json") == summary["result_sha256"][f"runs/{out.name}/result.json"]
        assert result["status"] == "ok" and result["freeze_sha256"] == sha(SOURCE / "FREEZE.json")
        for file, digest in [("archives.json", result["archives_sha256"]),
                             ("CHOICES.json", result["choices_sha256"])] + [
                (v["model"], v["model_sha256"]) for v in result["variants"].values()]:
            assert sha(out / file) == digest
            paths.append(out / file)
        paths.append(out / "result.json")
    save(HERE / "FREEZE.json", dict(
        files={p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(set(paths))},
        jobs=[source.name(j) for j in source.jobs()], methods=METHODS,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        python=platform.python_version(), packages={p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "scikit-learn", "threadpoolctl", "pytest")},
        evidence="Retrospective fixed-candidate diagnosis; outer minima are hindsight references."))


def identity(saved):
    return json.dumps({k: saved[k] for k in ("features", "coefficients", "intercepts")},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def deduplicate(pool):
    seen, unique = {}, []
    for index, saved in enumerate(pool):
        key = identity(saved)
        if key not in seen:
            seen[key] = len(unique)
            unique.append(dict(saved=saved, aliases=[]))
        unique[seen[key]]["aliases"].append(index)
    return unique


def active_head(saved, output):
    weights = saved["coefficients"][output]
    used = [i for i, w in enumerate(weights) if w != 0]
    return dict(features=[saved["features"][i] for i in used],
                coefficients=[[weights[i] for i in used]], intercepts=[saved["intercepts"][output]])


def metrics(y, prediction, scales):
    """Independent fixed-prediction metrics with explicit failed output scores."""
    assert prediction.shape == y.shape
    with np.errstate(all="ignore"):
        mse = np.mean(((prediction - y) / scales) ** 2, axis=0)
        sse = np.sum((prediction - y) ** 2, axis=0)
        sst = np.sum((y - y.mean(axis=0)) ** 2, axis=0)
    good = np.isfinite(prediction).all(axis=0) & np.isfinite(mse) & np.isfinite(sse)
    r2 = [float(1 - e / t) if t > 0 else float(e == 0) for e, t in zip(sse, sst)]
    return dict(mse=[float(x) if ok else None for x, ok in zip(mse, good)],
        nrmse=[float(np.sqrt(x)) if ok else None for x, ok in zip(mse, good)],
        r2=[r if ok and len(y) > 1 else None for r, ok in zip(r2, good)], finite=good.tolist())


def evaluate(saved, split, output, cache):
    key = (output, identity(saved))
    if key not in cache:
        c = decode(saved)
        columns = slice(None) if output is None else slice(output, output + 1)
        scores = {}
        for part, suffix in (("inner", "_val"), ("outer", "_eval")):
            with np.errstate(all="ignore"):
                prediction = predict_candidate(c, split["X" + suffix])
            scores[part] = metrics(split["Y" + suffix][:, columns], prediction, split["target_scale"][columns])
        keys = node_keys(c["features"])
        edges = int(np.count_nonzero(c["coefficients"]) + np.count_nonzero(c["intercepts"]))
        cache[key] = dict(**scores, keys=keys, edges=edges, complexity=len(keys) + edges)
    return cache[key]


def mean_score(record, part):
    values = record[part]["nrmse"]
    return sum(values) / len(values) if all(v is not None for v in values) else math.inf


def rank_diagnostics(inner, outer, selected):
    finite = [i for i, value in enumerate(outer) if value is not None]
    x, y = np.asarray(inner)[finite], np.asarray([outer[i] for i in finite])
    rho = float(spearmanr(x, y).statistic) if len(set(x)) > 1 and len(set(y)) > 1 else None
    percentile = None
    if selected in finite and len(finite) > 1:
        percentile = float((rankdata(y, method="average")[finite.index(selected)] - 1) / (len(finite) - 1))
    return dict(spearman=rho, selected_outer_rank_percentile=percentile,
                rank_finite_count=len(finite), nonfinite_outer_count=len(outer) - len(finite))


def enumerate_systems(pools, caps, *, joint=False):
    """Exact K unions. Inner selection never uses outer scores or finiteness."""
    nodes, encoded = {}, []
    for pool in pools:
        choices = []
        for r in pool:
            mask = 0
            for key in r["keys"]:
                mask |= 1 << nodes.setdefault(key, len(nodes))
            choices.append((mask, r["edges"], mean_score(r, "inner"), mean_score(r, "outer"),
                            r["eligible"], r["aliases"][0]))
        encoded.append(choices)
    if joint and len(encoded) != 1:
        raise ValueError("Joint systems must stay indivisible.")
    by_size = {}
    for combination in product(*encoded):
        mask, edges, inner, outer, eligible = 0, 0, 0., 0., True
        indices = tuple(c[5] for c in combination)
        for bits, n, v, t, ok, _ in combination:
            mask |= bits
            edges += n
            inner += v / len(encoded)
            outer += t / len(encoded)
            eligible &= ok
        k = mask.bit_count() + edges
        if k not in by_size:
            by_size[k] = dict(count=0, finite=0, eligible=0, eligible_finite=0,
                              inner=None, hindsight=None, eligible_hindsight=None)
        row = by_size[k]
        row["count"] += 1
        row["finite"] += math.isfinite(outer)
        row["eligible"] += eligible
        row["eligible_finite"] += eligible and math.isfinite(outer)
        for mode, score, valid in (("inner", inner, True), ("hindsight", outer, math.isfinite(outer)),
                ("eligible_hindsight", outer, not joint and eligible and math.isfinite(outer))):
            if valid and (row[mode] is None or (score, indices) < (row[mode][0], row[mode][1])):
                row[mode] = (score, indices, inner, outer)
    result = {}
    for label, cap in caps.items():
        feasible = [(k, r) for k, r in by_size.items() if cap is None or k <= cap]
        row = dict(cap=cap, **{key: sum(r[key] for _, r in feasible)
                               for key in ("count", "finite", "eligible", "eligible_finite")})
        for mode in ("inner", "hindsight", "eligible_hindsight"):
            options = [(r[mode][0], k, r[mode][1], r[mode][2], r[mode][3])
                       for k, r in feasible if r[mode] is not None]
            if options:
                _, k, indices, v, t = min(options)
                row[mode] = dict(indices=list(indices), complexity=k, inner_nrmse=v,
                                outer_nrmse=t if math.isfinite(t) else None,
                                outer_status="ok" if math.isfinite(t) else "nonfinite")
            else:
                row[mode] = None
        result[label] = row
    return result


def head_diagnosis(pool, selected, common_k, split, output, cache):
    projections = [active_head(c, output if len(c["intercepts"]) > 1 else 0) for c in pool]
    unique = deduplicate(projections)
    records = [evaluate(r["saved"], split, output, cache) for r in unique]
    at = next(i for i, r in enumerate(unique) if selected in r["aliases"])
    chosen = records[at]
    inner = [r["inner"]["nrmse"][0] for r in records]
    outer = [r["outer"]["nrmse"][0] for r in records]
    result = dict(selected_source_index=selected, pool_entries=len(pool), active_unique=len(unique),
        selected_k=chosen["complexity"], selected_inner_nrmse=inner[at], selected_outer_nrmse=outer[at],
        selected_outer_r2=chosen["outer"]["r2"][0], **rank_diagnostics(inner, outer, at))
    for label, cap in (("own", chosen["complexity"]), ("common", common_k)):
        options = [(r["outer"]["nrmse"][0], r["complexity"], unique[i]["aliases"][0], i)
                   for i, r in enumerate(records) if r["complexity"] <= cap and r["outer"]["finite"][0]]
        result[label + "_cap"] = cap
        result[label + "_finite_count"] = len(options)
        result[label + "_hindsight"] = None
        if options:
            error, k, index, i = min(options)
            result[label + "_hindsight"] = dict(outer_nrmse=error, complexity=k, source_index=index,
                outer_r2=records[i]["outer"]["r2"][0], inner_nrmse=inner[i])
    return result


def diagnose(job, split):
    folder = SOURCE / "runs" / source.name(job)
    archives, choices, original = (read(folder / f) for f in ("archives.json", "CHOICES.json", "result.json"))
    assert split["metadata"]["fold_sha256"] == original["fold_sha256"]
    cache, systems, heads, candidate_rows = {}, [], [], []
    validation_deltas, selected_deltas, raw_entries = [], [], 0
    q = len(split["target_names"])
    common_heads = [evaluate(active_head(archives["independent25"][j][i], 0), split, j, cache)["complexity"]
                    for j, i in enumerate(choices["independent25"]["indices"])]
    for method in METHODS:
        joint = method == "joint25"
        source_pools = [archives[method]] if joint else archives[method]
        pools = []
        for j, pool in enumerate(source_pools):
            minimum = min(c["validation_loss"] for c in pool)
            records = []
            raw_entries += len(pool)
            for unique in deduplicate(pool):
                saved, aliases = unique["saved"], unique["aliases"]
                score = evaluate(saved, split, None if joint else j, cache)
                assert all(score["inner"]["finite"])
                assert score["complexity"] == saved["complexity"]
                # Scalar combination accounting assumes no repeated exact basis within a head.
                assert len(set(decode(saved)["features"])) == len(saved["features"])
                loss = sum(score["inner"]["mse"]) / len(score["inner"]["mse"])
                for index in aliases:
                    expected = pool[index]["validation_loss"]
                    np.testing.assert_allclose(loss, expected, rtol=1e-10, atol=1e-12)
                    assert expected == saved["validation_loss"]
                    validation_deltas.append(abs(loss - expected))
                record = dict(score, aliases=aliases, eligible=saved["validation_loss"] <= 1.01 * minimum + 1e-12)
                records.append(record)
                candidate_rows.append(dict(method=method, output=None if joint else j, aliases=aliases,
                    origins=[pool[i]["origin"] for i in aliases], **{k: v for k, v in record.items()
                        if k not in ("keys", "aliases")}))
            pools.append(records)
        selected = choices[method]
        model = MIMOSymbolicRegressor.load(folder / selected["model"])
        assert model.model_["complexity"] == selected["complexity"]
        reconstructed = decode(archives[method][selected["indices"][0]]) if joint else source.shared.combine(
            archives[method], selected["indices"])
        assert identity(source.shared._serial_candidate(reconstructed)) == identity(read(folder / selected["model"])["model"])
        for part, suffix in (("inner", "_val"), ("outer", "_eval")):
            actual = metrics(split["Y" + suffix], model.predict(split["X" + suffix]), split["target_scale"])
            expected = original["variants"][method][part]
            for metric in ("nrmse", "r2"):
                np.testing.assert_allclose(actual[metric], expected[metric], rtol=1e-10, atol=1e-12)
                selected_deltas.extend(abs(a - b) for a, b in zip(actual[metric], expected[metric]))
        selected_records = [next(c for c in pool if index in c["aliases"])
                            for pool, index in zip(pools, selected["indices"])]
        k = len(set().union(*(r["keys"] for r in selected_records))) + sum(r["edges"] for r in selected_records)
        assert k == selected["complexity"]
        for part in ("inner", "outer"):
            np.testing.assert_allclose(np.mean([mean_score(r, part) for r in selected_records]),
                original["variants"][method][part]["mean_nrmse"], rtol=1e-10, atol=1e-12)
        caps = dict(own=k, common=choices["independent25"]["complexity"], unbounded=None)
        selections = enumerate_systems(pools, caps, joint=joint)
        old = original["variants"][method]
        assert selections["own"]["hindsight"]["outer_nrmse"] <= old["outer"]["mean_nrmse"] + 1e-10
        if not joint:
            assert all(r["eligible"] for r in selected_records)
            assert selections["own"]["eligible_hindsight"] is not None
        systems.append(dict(method=method, selected_k=k, selected_indices=selected["indices"],
            selected_inner_nrmse=old["inner"]["mean_nrmse"], selected_outer_nrmse=old["outer"]["mean_nrmse"],
            pool_entries=sum(map(len, source_pools)), unique_entries=sum(map(len, pools)), caps=selections))
        for j, target in enumerate(split["target_names"]):
            row = head_diagnosis(archives[method] if joint else archives[method][j],
                selected["indices"][0 if joint else j], common_heads[j], split, j, cache)
            np.testing.assert_allclose(row["selected_outer_nrmse"], old["outer"]["nrmse"][j], rtol=1e-10, atol=1e-12)
            heads.append(dict(method=method, output=j, target=target, **row))
    return dict(job=list(job), freeze_sha256=sha(HERE / "FREEZE.json"),
        source_result_sha256=sha(folder / "result.json"), fold_sha256=original["fold_sha256"],
        systems=systems, heads=heads, candidates=candidate_rows,
        verification=dict(raw_entries=raw_entries, unique_entries=len(candidate_rows),
            validation_scores_checked=len(validation_deltas), selected_models_checked=len(METHODS),
            selected_metric_components_checked=len(selected_deltas),
            max_validation_mse_delta=max(validation_deltas), max_selected_metric_delta=max(selected_deltas)))


def checkpoint(path, result):
    # The receipt commits a completed cell; partial files remain visible after interruption.
    save(path, result)
    save(path.with_suffix(".sha256.json"), dict(sha256=sha(path), freeze_sha256=sha(HERE / "FREEZE.json")))


def load_checkpoint(path):
    receipt = read(path.with_suffix(".sha256.json"))
    if receipt != dict(sha256=sha(path), freeze_sha256=sha(HERE / "FREEZE.json")):
        raise ValueError(f"Changed checkpoint or freeze: {path}")
    result = read(path)
    assert result["freeze_sha256"] == sha(HERE / "FREEZE.json")
    return result


def batch():
    check_freeze()
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    splits = {}
    with threadpool_limits(limits=1):
        for job in source.jobs():
            path = out / (source.name(job) + ".json")
            if path.exists():
                assert load_checkpoint(path)["job"] == list(job)
                print(path.stem, "verified checkpoint", flush=True)
                continue
            key = job[:2]
            if key not in splits:
                splits[key] = source.v3.prepare_split(*key)
            result = diagnose(job, splits[key])
            checkpoint(path, result)
            print(path.stem, result["verification"], flush=True)
    check_freeze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "batch"))
    args = parser.parse_args()
    prepare() if args.command == "prepare" else batch()
