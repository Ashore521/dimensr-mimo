"""Training-only transfer of legal subexpressions between independent archives."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import time

import numpy as np

from dimensr_mimo.features import Feature, evaluate_features
from dimensr_mimo.search import search, make_candidate, predict_candidate, pareto
from dimensr_mimo._vendor.intervals import audit_box
from dimensr_mimo._vendor.units import sub

_spec = importlib.util.spec_from_file_location("transfer_shared_helpers",
    Path(__file__).resolve().parents[1]/"shared_private_v1/hybrid.py")
shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shared)


def independent(X, Y, bounds, units, targets, coefficients, cfg):
    """The unchanged scalar search, with equal CPU allowances per output."""
    pools, diagnostics = [], []
    for j in range(Y.shape[1]):
        front, _, _, counts = search(X, Y[:, j:j+1], bounds, units, targets[j:j+1],
            coefficients, replace(cfg, search_seconds=cfg.search_seconds/Y.shape[1]))
        pools.append(front)
        diagnostics.append(counts)
    return pools, dict(search_cpu_seconds=sum(d["search_cpu_seconds"] for d in diagnostics),
        budget_overshoot_seconds=sum(d["budget_overshoot_seconds"] for d in diagnostics),
        stages=diagnostics)


def subexpressions(feature):
    """Retain the constants belonging to each subtree, in traversal order."""
    offset = 0
    found = []
    def visit(node):
        nonlocal offset
        start = offset
        for child in node.children:
            visit(child)
        if not node.children and node.token.startswith("c"):
            offset += 1
        if node.children:
            tree, values = node.canonical(feature.constants[start:offset])
            found.append(Feature(tree, tuple(values)))
    visit(feature.tree)
    return tuple(dict.fromkeys(found))


def semantic_scores(values, residual, centered=True):
    """Squared cosine: signs can be absorbed by the refitted output coefficient."""
    values = values - values.mean(axis=0) if centered else values
    magnitude = np.max(np.abs(values), axis=0)
    safe = np.where(magnitude > 0, magnitude, 1.)
    values = values/safe
    norms = np.linalg.norm(values, axis=0)
    scale = np.max(np.abs(residual))
    response = residual/scale if scale else residual
    norm_y = np.linalg.norm(response)
    denominator = norms * norm_y
    return np.divide((values.T @ response)**2, denominator**2,
        out=np.zeros(len(norms)), where=denominator > 0)


def transfer(pools, X, Y, bounds, targets, coefficients, cfg, *, seconds, rule):
    """Append candidates; never remove or mutate the original independent heads.

    shortcut: training Pareto beams and add/weakest-replace moves bound cost;
    revisit broader recombination only after this local transfer is evaluated.
    No validation or outer data is accepted. Donor banks are frozen at entry.
    """
    if rule not in ("random", "semantic") or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Expected random/semantic transfer and a positive CPU allowance.")
    started = time.process_time()
    deadline = started + seconds
    rng = np.random.default_rng(cfg.random_state + 104729)
    scales = shared.scales_of(Y)
    original = [list(p) for p in pools]
    additions = [[] for _ in pools]
    seen = [{frozenset(c["features"]) for c in pool} for pool in pools]
    counts = dict(rule=rule,head_fits=0,accepted=0,domain_rejections=0,rounds=0,
        attempted_per_output=[0]*len(pools),accepted_per_output=[0]*len(pools),
        original_counts=list(map(len,pools)))
    cache, sources, checked = {}, {}, set()
    # Interleave donor outputs so bank construction cannot consume the budget on output 0.
    for index in range(max(map(len,pools))):
        for j,pool in enumerate(pools):
            if index >= len(pool):
                continue
            for root in pool[index]["features"]:
                if time.process_time() >= deadline:
                    break
                if root not in checked:
                    checked.add(root)
                    if audit_box(root.tree,root.constants,bounds,max_boxes=cfg.interval_max_boxes)["status"] != "enclosed":
                        counts["domain_rejections"] += 1
                        continue
                elif root not in cache:
                    continue
                cache.setdefault(root,evaluate_features((root,),X)[:,0])
                for feature in subexpressions(root):
                    if time.process_time() >= deadline:
                        break
                    if feature.tree.size > cfg.max_nodes or feature.tree.depth > cfg.max_depth:
                        continue
                    if feature not in cache:
                        if audit_box(feature.tree,feature.constants,bounds,max_boxes=cfg.interval_max_boxes)["status"] != "enclosed":
                            counts["domain_rejections"] += 1
                            continue
                        cache[feature] = evaluate_features((feature,),X)[:,0]
                    if np.isfinite(cache[feature]).all():
                        sources.setdefault(feature,set()).add(j)
    banks = [[f for f,owners in sources.items() if owners-{j}
        and sub(targets[j],f.tree.units) in coefficients] for j in range(len(pools))]
    counts["donor_counts"] = list(map(len,banks))

    def proposals(j, parents):
        bank = banks[j]
        if not bank:
            return
        matrix = np.column_stack([cache[f] for f in bank])
        for parent in parents:
            if time.process_time() >= deadline:
                return
            remaining = [i for i,f in enumerate(bank) if f not in parent["features"]]
            if rule == "semantic":
                residual = Y[:,j]-predict_candidate(parent,X)[:,0]
                scores = semantic_scores(matrix,residual,targets[j] in coefficients)
                remaining.sort(key=lambda i:(-scores[i],bank[i].tree.size,i))
            else:
                rng.shuffle(remaining)
            old = parent["features"]
            weakest = int(np.argmin(np.abs(parent["coefficients"][0])*
                evaluate_features(old,X).std(axis=0))) if old else None
            for index in remaining[:8]:
                donor = bank[index]
                moves = [old+(donor,)] if len(old)<cfg.max_terms else []
                if old:
                    moves.append(tuple(f for k,f in enumerate(old) if k!=weakest)+(donor,))
                for features in moves:
                    key = frozenset(features)
                    if key not in seen[j]:
                        seen[j].add(key)
                        yield parent, features, donor

    while time.process_time() < deadline:
        counts["rounds"] += 1
        streams = [iter(proposals(j,pareto(original[j]+additions[j]))) for j in range(len(pools))]
        active = list(range(len(pools)))
        new = 0
        while active and time.process_time() < deadline:
            for j in list(active):
                if time.process_time() >= deadline:
                    break
                item = next(streams[j],None)
                if item is None:
                    active.remove(j)
                    continue
                if time.process_time() >= deadline:
                    break
                parent,features,donor = item
                candidate = make_candidate(features,evaluate_features(features,X),Y[:,j:j+1],
                    scales[j:j+1],targets[j:j+1],coefficients,cfg.head_ridge)
                counts["head_fits"] += 1
                counts["attempted_per_output"][j] += 1
                if (not np.isfinite(candidate["train_loss"]) or
                    (candidate["train_loss"] >= parent["train_loss"]-1e-14
                     and candidate["complexity"] >= parent["complexity"])):
                    continue
                if not shared.enclosed(candidate,bounds,targets[j:j+1],cfg.interval_max_boxes):
                    counts["domain_rejections"] += 1
                    continue
                candidate.update(origin=rule,donor_outputs=sorted(sources[donor]-{j}))
                additions[j].append(candidate)
                counts["accepted"] += 1
                counts["accepted_per_output"][j] += 1
                new += 1
        if not new:
            break
    result = [old+pareto(new) for old,new in zip(original,additions)]
    elapsed = time.process_time()-started
    counts.update(search_cpu_seconds=elapsed,budget_overshoot_seconds=max(0.,elapsed-seconds),
        unused_seconds=max(0.,seconds-elapsed),archive_counts=list(map(len,result)))
    return result,counts
