"""Optional, typed subtree grafts evaluated in the recipient's full context."""
import importlib.util
from pathlib import Path
import time

import numpy as np

from dimensr_mimo.features import Feature, evaluate_features
from dimensr_mimo.search import make_candidate, pareto
from dimensr_mimo._vendor.intervals import audit_box
from dimensr_mimo._vendor.units import sub

_spec = importlib.util.spec_from_file_location("contextual_legacy_transfer",
    Path(__file__).resolve().parents[1]/"semantic_transfer_v1/transfer.py")
legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(legacy)
shared = legacy.shared


def graft(feature, path, donor):
    """Replace exactly one occurrence, preserving all unaffected constants."""
    offset = 0
    for position, node in feature.tree.walk():
        if position == path:
            tree = feature.tree.replace(path, donor.tree)
            values = feature.constants[:offset] + donor.constants + feature.constants[offset+node.n_constants:]
            tree, values = tree.canonical(values)
            return Feature(tree, tuple(values))
        if not node.children and node.token.startswith("c"):
            offset += 1
    raise ValueError("Subtree path does not exist.")


def transfer(pools, X, Y, bounds, targets, coefficients, cfg, *, seconds, internal=True):
    """Extend independent archives; training data alone generate and score moves.

    Both ablation arms use the same random donor policy, head fitting and final
    selector. The internal arm additionally substitutes proper subtrees.
    shortcut: eight donors and four matching positions per donor/parent;
    broaden only in a separate experiment if this bounded search warrants it.
    """
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Expected a positive finite CPU allowance.")
    cfg.validate()
    started = time.process_time()
    deadline = started + seconds
    rng = np.random.default_rng(cfg.random_state + 104729)
    position_rng = np.random.default_rng(cfg.random_state + 13007)
    original = [list(pool) for pool in pools]
    additions = [[] for _ in pools]
    seen = [{frozenset(c["features"]) for c in pool} for pool in pools]
    scales = shared.scales_of(Y)
    counts = dict(internal=internal, head_fits=0, accepted=0, internal_attempts=0,
        internal_accepted=0, domain_rejections=0, size_rejections=0, rounds=0,
        attempted_per_output=[0]*len(pools), accepted_per_output=[0]*len(pools),
        original_counts=list(map(len,pools)))
    values, legal, owners = {}, {}, {}

    def admit(feature):
        if feature in legal:
            return legal[feature]
        if feature.tree.size > cfg.max_nodes or feature.tree.depth > cfg.max_depth:
            counts["size_rejections"] += 1
            legal[feature] = False
            return False
        ok = audit_box(feature.tree, feature.constants, bounds,
            max_boxes=cfg.interval_max_boxes)["status"] == "enclosed"
        if ok:
            column = evaluate_features((feature,), X)[:,0]
            ok = bool(np.isfinite(column).all())
            if ok:
                values[feature] = column
        if not ok:
            counts["domain_rejections"] += 1
        legal[feature] = ok
        return ok

    # Unsafe roots cannot donate otherwise safe-looking children.
    for index in range(max(map(len,pools), default=0)):
        for j, pool in enumerate(pools):
            if index >= len(pool):
                continue
            for root in pool[index]["features"]:
                if time.process_time() >= deadline:
                    break
                if not admit(root):
                    continue
                for feature in legacy.subexpressions(root):
                    if time.process_time() >= deadline:
                        break
                    if admit(feature):
                        owners.setdefault(feature,set()).add(j)
    banks = [[f for f, sources in owners.items() if sources-{j}] for j in range(len(pools))]
    counts["donor_counts"] = list(map(len,banks))

    def proposals(j, parents):
        for parent in parents:
            if time.process_time() >= deadline:
                return
            old = parent["features"]
            positions = [(k,path,node.units) for k,f in enumerate(old)
                for path,node in f.tree.walk() if path]
            # An internal donor need only match its insertion position's units.
            bank = [f for f in banks[j] if sub(targets[j],f.tree.units) in coefficients
                or (internal and any(u == f.tree.units for _,_,u in positions))]
            order = rng.permutation(len(bank))[:8]
            weakest = int(np.argmin(np.abs(parent["coefficients"][0]) *
                evaluate_features(old,X).std(axis=0))) if old else None
            for index in order:
                if time.process_time() >= deadline:
                    return
                donor = bank[index]
                moves = []
                if sub(targets[j],donor.tree.units) in coefficients:
                    if len(old) < cfg.max_terms:
                        moves.append((old+(donor,), "whole", None))
                    if old:
                        moves.append((tuple(f for k,f in enumerate(old) if k!=weakest)+(donor,), "whole", None))
                if internal:
                    compatible = [(k,path) for k,path,u in positions if u == donor.tree.units]
                    for pos in position_rng.permutation(len(compatible))[:4]:
                        k,path = compatible[pos]
                        replacement = graft(old[k],path,donor)
                        moves.append((old[:k]+(replacement,)+old[k+1:], "internal", [k,list(path)]))
                for features,kind,position in moves:
                    if time.process_time() >= deadline:
                        return
                    # Duplicate bases have identical columns; do not refit them twice.
                    features = tuple(dict.fromkeys(features))
                    key = frozenset(features)
                    if key not in seen[j]:
                        seen[j].add(key)
                        yield parent,features,donor,kind,position

    while time.process_time() < deadline:
        counts["rounds"] += 1
        streams = [iter(proposals(j,pareto(original[j]+additions[j]))) for j in range(len(pools))]
        active = list(range(len(pools)))
        accepted = 0
        while active and time.process_time() < deadline:
            for j in list(active):
                if time.process_time() >= deadline:
                    break
                item = next(streams[j],None)
                if item is None:
                    active.remove(j)
                    continue
                parent,features,donor,kind,position = item
                counts["attempted_per_output"][j] += 1
                counts["internal_attempts"] += int(kind == "internal")
                # Audit the full modified basis, not only the donor subtree.
                if not all(admit(f) for f in features) or time.process_time() >= deadline:
                    continue
                matrix = np.column_stack([values[f] for f in features])
                candidate = make_candidate(features,matrix,Y[:,j:j+1],scales[j:j+1],
                    targets[j:j+1],coefficients,cfg.head_ridge)
                counts["head_fits"] += 1
                if (not np.isfinite(candidate["train_loss"]) or
                    (candidate["train_loss"] >= parent["train_loss"]-1e-14
                     and candidate["complexity"] >= parent["complexity"])):
                    continue
                if not shared.enclosed(candidate,bounds,targets[j:j+1],cfg.interval_max_boxes):
                    counts["domain_rejections"] += 1
                    continue
                candidate.update(origin=kind, donor_outputs=sorted(owners[donor]-{j}),
                    graft_position=position)
                additions[j].append(candidate)
                counts["accepted"] += 1
                counts["internal_accepted"] += int(kind == "internal")
                counts["accepted_per_output"][j] += 1
                accepted += 1
        if not accepted:
            break
    result = [old+pareto(new) for old,new in zip(original,additions)]
    elapsed = time.process_time()-started
    counts.update(search_cpu_seconds=elapsed, budget_overshoot_seconds=max(0.,elapsed-seconds),
        unused_seconds=max(0.,seconds-elapsed), archive_counts=list(map(len,result)))
    return result,counts
