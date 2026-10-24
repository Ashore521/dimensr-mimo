"""Deterministic tables for the frozen, retrospective candidate diagnosis."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from candidate_diagnosis import HERE, ROOT, METHODS, DATASETS, source, read, sha, save, check_freeze, load_checkpoint


def stats(values):
    good = [v for v in values if v is not None]
    return dict(expected=len(values), finite=len(good), missing=len(values) - len(good),
                mean=statistics.mean(good) if good else None,
                median=statistics.median(good) if good else None,
                min=min(good) if good else None, max=max(good) if good else None)


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def flatten_cells(cells):
    systems, heads, candidates = [], [], []
    for cell in cells:
        d, f, s = cell["job"]
        base = dict(dataset=d, fold=f, seed=s)
        for r in cell["systems"]:
            fixed = {k: v for k, v in r.items() if k != "caps"}
            for cap, entry in r["caps"].items():
                row = dict(base, **fixed, cap_kind=cap)
                row.update({k: v for k, v in entry.items() if k not in ("inner", "hindsight", "eligible_hindsight")})
                for mode in ("inner", "hindsight", "eligible_hindsight"):
                    selection = entry[mode]
                    row[mode + "_exists"] = selection is not None
                    for key in ("indices", "complexity", "inner_nrmse", "outer_nrmse", "outer_status"):
                        row[mode + "_" + key] = selection[key] if selection else None
                best, selected = row["hindsight_outer_nrmse"], r["selected_outer_nrmse"]
                row["hindsight_gap"] = None if best is None else selected - best
                row["hindsight_gap_pct"] = 100 * (selected - best) / selected if best is not None and selected else None
                eligible = row["eligible_hindsight_outer_nrmse"]
                row["within_filter_gap"] = None if eligible is None else selected - eligible
                row["excluded_by_filter_gap"] = None if eligible is None or best is None else eligible - best
                value = row["inner_outer_nrmse"]
                row["inner_counterfactual_delta"] = None if value is None else value - selected
                systems.append(row)
        for r in cell["heads"]:
            row = dict(base, **{k: v for k, v in r.items() if k not in ("own_hindsight", "common_hindsight")})
            for cap in ("own", "common"):
                selection = r[cap + "_hindsight"]
                for key in ("outer_nrmse", "outer_r2", "inner_nrmse", "complexity", "source_index"):
                    row[cap + "_hindsight_" + key] = selection[key] if selection else None
            heads.append(row)
        for r in cell["candidates"]:
            fixed = dict(base, **{k: v for k, v in r.items() if k not in ("inner", "outer")})
            for j in range(len(r["inner"]["nrmse"])):
                row = dict(fixed, score_output=j if r["output"] is None else r["output"])
                for part in ("inner", "outer"):
                    row.update({part + "_" + k: v[j] for k, v in r[part].items()})
                candidates.append(row)
    return systems, heads, candidates


SYSTEM_FIELDS = ("selected_k", "selected_outer_nrmse", "inner_outer_nrmse", "inner_complexity",
    "hindsight_outer_nrmse", "hindsight_complexity", "eligible_hindsight_outer_nrmse",
    "hindsight_gap", "hindsight_gap_pct", "within_filter_gap", "excluded_by_filter_gap",
    "inner_counterfactual_delta", "count", "finite", "eligible", "eligible_finite")
HEAD_FIELDS = ("selected_k", "selected_outer_nrmse", "selected_outer_r2", "active_unique", "spearman",
    "selected_outer_rank_percentile", "own_hindsight_outer_nrmse", "own_hindsight_outer_r2",
    "common_hindsight_outer_nrmse", "common_hindsight_outer_r2", "nonfinite_outer_count")


def aggregate_systems(rows):
    result = {k: stats([r[k] for r in rows]) for k in SYSTEM_FIELDS}
    result["headroom_ge_1pct"] = sum(r["hindsight_gap_pct"] is not None and r["hindsight_gap_pct"] >= 1 for r in rows)
    result["inner_better_equal_worse_missing"] = [
        sum(r["inner_counterfactual_delta"] is not None and r["inner_counterfactual_delta"] < -1e-12 for r in rows),
        sum(r["inner_counterfactual_delta"] is not None and abs(r["inner_counterfactual_delta"]) <= 1e-12 for r in rows),
        sum(r["inner_counterfactual_delta"] is not None and r["inner_counterfactual_delta"] > 1e-12 for r in rows),
        sum(r["inner_counterfactual_delta"] is None for r in rows)]
    return result


def paired_comparisons(rows):
    lookup = {(r["dataset"], r["fold"], r["seed"], r["method"]): r for r in rows if r["cap_kind"] == "common"}
    pairs = []
    for control in ("independent25", "whole25", "semantic25", "joint25", "independent20"):
        for job in source.jobs():
            a, b = (lookup[(*job, m)] for m in ("context25", control))
            row = dict(dataset=job[0], fold=job[1], seed=job[2], left="context25", right=control, cap=a["cap"])
            assert a["cap"] == b["cap"]
            for field in ("hindsight_outer_nrmse", "inner_outer_nrmse", "hindsight_complexity", "inner_complexity"):
                row["delta_" + field] = a[field] - b[field] if a[field] is not None and b[field] is not None else None
            pairs.append(row)
    return pairs


def summarize(systems, heads, pairs):
    groups, head_groups, pair_groups = {}, {}, {}
    for method in METHODS:
        groups[method] = {}
        for cap in ("own", "common", "unbounded"):
            rows = [r for r in systems if r["method"] == method and r["cap_kind"] == cap]
            assert len(rows) == 45
            datasets = {d: aggregate_systems([r for r in rows if r["dataset"] == d]) for d in DATASETS}
            group = aggregate_systems(rows)
            # Require complete datasets for an equal-task headline; finite-only means stay labeled above.
            group["equal_task_means"] = {k: statistics.mean(datasets[d][k]["mean"] for d in DATASETS)
                if all(datasets[d][k]["finite"] == 15 for d in DATASETS) else None for k in SYSTEM_FIELDS}
            group["datasets"] = datasets
            groups[method][cap] = group
        head_groups[method] = {}
        for dataset, target in sorted({(r["dataset"], r["target"]) for r in heads}):
            rows = [r for r in heads if r["method"] == method and r["dataset"] == dataset and r["target"] == target]
            assert len(rows) == 15
            head_groups[method][dataset + "/" + target] = {k: stats([r[k] for r in rows]) for k in HEAD_FIELDS}
    for control in sorted({r["right"] for r in pairs}):
        pair_groups[control] = {}
        for dataset in ("all", *DATASETS):
            rows = [r for r in pairs if r["right"] == control and (dataset == "all" or r["dataset"] == dataset)]
            fields = [k for k in rows[0] if k.startswith("delta_")]
            group = {k: stats([r[k] for r in rows]) for k in fields}
            values = [r["delta_hindsight_outer_nrmse"] for r in rows]
            group["hindsight_wins_ties_losses_missing"] = [sum(v is not None and v < -1e-12 for v in values),
                sum(v is not None and abs(v) <= 1e-12 for v in values), sum(v is not None and v > 1e-12 for v in values),
                sum(v is None for v in values)]
            pair_groups[control][dataset] = group
    return dict(methods=groups, outputs=head_groups, common_cap_paired=pair_groups)


def render_tables(summary):
    lines = ["# Fixed-candidate retrospective diagnosis", "",
        "Outer-optimal scores are hindsight references, not achieved predictive performance or blind evidence.", "",
        "## Own selected-system K cap", "",
        "| Method | Frozen outer NRMSE | Inner-only choice, revealed outer | Hindsight outer minimum | Within-filter gap | Excluded-by-filter gap | Cells with >=1% headroom |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(x):
        return "NA" if x is None else f"{x:.6f}"
    for m in METHODS:
        g = summary["methods"][m]["own"]
        v = g["equal_task_means"]
        fields = ("selected_outer_nrmse", "inner_outer_nrmse", "hindsight_outer_nrmse", "within_filter_gap", "excluded_by_filter_gap")
        lines.append(f"| {m} | " + " | ".join(fmt(v[k]) for k in fields) + f" | {g['headroom_ge_1pct']}/45 |")
    lines += ["", "## Common independent25 K cap", "", "| Method | Hindsight outer minimum | Inner-only choice, revealed outer | Hindsight mean K |", "|---|---:|---:|---:|"]
    for m in METHODS:
        v = summary["methods"][m]["common"]["equal_task_means"]
        lines.append(f"| {m} | " + " | ".join(fmt(v[k]) for k in ("hindsight_outer_nrmse", "inner_outer_nrmse", "hindsight_complexity")) + " |")
    lines += ["", "## Every output (active-head own K cap)", "",
        "Rank rho uses finite candidate pairs; percentile 0 is best. Joint projections are diagnostic views only.", "",
        "| Method | Dataset/output | Selected NRMSE | Hindsight NRMSE | Selected R2 | Hindsight R2 | Mean rank rho | Mean selected percentile |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for m in METHODS:
        for target, g in summary["outputs"][m].items():
            fields = ("selected_outer_nrmse", "own_hindsight_outer_nrmse", "selected_outer_r2", "own_hindsight_outer_r2", "spearman", "selected_outer_rank_percentile")
            lines.append(f"| {m} | {target} | " + " | ".join(fmt(g[k]["mean"]) for k in fields) + " |")
    return "\n".join(lines) + "\n"


def analyze(out):
    frozen = check_freeze()
    paths = [HERE / "runs" / (source.name(j) + ".json") for j in source.jobs()]
    cells = [load_checkpoint(p) for p in paths]
    assert [c["job"] for c in cells] == [list(j) for j in source.jobs()]
    systems, heads, candidates = flatten_cells(cells)
    pairs = paired_comparisons(systems)
    assert len(systems) == 810 and len(heads) == 720
    summary = dict(freeze_sha256=sha(HERE / "FREEZE.json"), cells=len(cells),
        cell_sha256={p.name: sha(p) for p in paths}, **summarize(systems, heads, pairs))
    out.mkdir(parents=True, exist_ok=False)
    for file, rows in (("systems.csv", systems), ("heads.csv", heads), ("candidate_scores.csv", candidates), ("paired.csv", pairs)):
        write_csv(out / file, rows)
    with (out / "TABLES.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render_tables(summary))
    summary["table_sha256"] = {p.name: sha(p) for p in sorted(out.iterdir())}
    save(out / "summary.json", summary)
    counters = [c["verification"] for c in cells]
    verification = dict(freeze_sha256=summary["freeze_sha256"], source_files_checked=len(frozen["files"]),
        cells=len(cells), source_hashes_after_analysis="unchanged", no_fit_or_new_holdout=True,
        **{k: sum(c[k] for c in counters) for k in ("raw_entries", "unique_entries", "validation_scores_checked", "selected_models_checked", "selected_metric_components_checked")},
        max_validation_mse_delta=max(c["max_validation_mse_delta"] for c in counters),
        max_selected_metric_delta=max(c["max_selected_metric_delta"] for c in counters),
        nonfinite_candidate_output_rows=sum(not r["outer_finite"] for r in candidates),
        empty_cap_sets=sum(r["count"] == 0 for r in systems),
        inner_selected_nonfinite_outer=sum(r["inner_outer_status"] == "nonfinite" for r in systems),
        total_unique_system_combinations=sum(r["count"] for r in systems if r["cap_kind"] == "unbounded"),
        output_sha256={p.name: sha(p) for p in sorted(out.iterdir())})
    check_freeze()
    save(out / "VERIFICATION.json", verification)
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "analysis")
    args = parser.parse_args()
    print(json.dumps(analyze(args.out), indent=2))
