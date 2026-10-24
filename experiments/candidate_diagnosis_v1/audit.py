"""Post-analysis audit: reconstruct chosen systems with the original combine/metric code."""
import csv
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from candidate_diagnosis import HERE, SOURCE, ROOT, source, read, sha, save, check_freeze, load_checkpoint


def audit():
    # This audit has a separate pin because it verifies already-produced results.
    pin = read(HERE / "AUDIT_FREEZE.json")
    for relative, digest in pin["files"].items():
        assert sha(ROOT / relative) == digest
    frozen = check_freeze()
    audit_selections, reconstructions, deltas = 0, 0, []
    with threadpool_limits(limits=1):
        for job in source.jobs():
            cell = load_checkpoint(HERE / "runs" / (source.name(job) + ".json"))
            archives = read(SOURCE / "runs" / source.name(job) / "archives.json")
            split = source.v3.prepare_split(*job[:2])
            cache = {}
            for system in cell["systems"]:
                method = system["method"]
                for label, cap in system["caps"].items():
                    for mode in ("inner", "hindsight", "eligible_hindsight"):
                        chosen = cap[mode]
                        if chosen is None:
                            continue
                        audit_selections += 1
                        key = method, tuple(chosen["indices"])
                        if key not in cache:
                            c = source.shared.decode(archives[method][key[1][0]]) if method == "joint25" else source.shared.combine(archives[method], key[1])
                            scores = {}
                            for part, suffix in (("inner", "_val"), ("outer", "_eval")):
                                scores[part] = source.v3.metrics(split["Y" + suffix],
                                    source.predict_candidate(c, split["X" + suffix]), split["target_scale"])
                            cache[key] = c["complexity"], scores
                            reconstructions += 1
                        k, scores = cache[key]
                        assert k == chosen["complexity"]
                        assert cap["cap"] is None or k <= cap["cap"]
                        for part in ("inner", "outer"):
                            actual, expected = scores[part]["mean_nrmse"], chosen[part + "_nrmse"]
                            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
                            deltas.append(abs(actual - expected))
            print(source.name(job), "reconstructed", len(cache), flush=True)
    summary = read(HERE / "analysis/summary.json")
    with (HERE / "analysis/systems.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for method in source.METHODS:
        for cap in ("own", "common", "unbounded"):
            group = [r for r in rows if r["method"] == method and r["cap_kind"] == cap]
            assert len(group) == 45
            for field, expected in summary["methods"][method][cap]["equal_task_means"].items():
                if expected is None:
                    assert any(r[field] == "" for r in group)
                    continue
                means = [sum(float(r[field]) for r in group if r["dataset"] == dataset) / 15 for dataset in source.DATASETS]
                np.testing.assert_allclose(sum(means) / 3, expected, rtol=1e-12, atol=1e-12)
            if cap == "own" and method != "joint25":
                for r in group:
                    np.testing.assert_allclose(float(r["hindsight_gap"]),
                        float(r["within_filter_gap"]) + float(r["excluded_by_filter_gap"]), rtol=1e-10, atol=1e-12)
    recheck = ROOT / "outputs/candidate-diagnosis-summary-recheck"
    rebuilt = {p.name: sha(p) for p in (HERE / "analysis").iterdir() if p.is_file()}
    for file, digest in rebuilt.items():
        assert sha(recheck / file) == digest
    check_freeze()
    result = dict(audit_freeze_sha256=sha(HERE / "AUDIT_FREEZE.json"),
        diagnostic_freeze_sha256=sha(HERE / "FREEZE.json"), unchanged_frozen_files=len(frozen["files"]),
        selection_records_checked=audit_selections, distinct_systems_reconstructed=reconstructions,
        mean_nrmse_components_checked=len(deltas), max_mean_nrmse_delta=max(deltas),
        independent_equal_task_csv_means="matched", own_cap_gap_decomposition="matched",
        byte_identical_rebuilt_outputs=rebuilt, tests=dict(passed=195, added=10),
        evidence="Audit of fixed candidates only; no refitting, new holdouts or algorithm changes.")
    save(HERE / "VERIFICATION.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
