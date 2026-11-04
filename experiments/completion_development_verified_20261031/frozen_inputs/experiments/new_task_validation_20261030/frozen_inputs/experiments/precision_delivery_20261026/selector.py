"""Choose a small system only inside frozen absolute and relative constraints."""
from copy import deepcopy
import math
import numpy as np
from dimensr_mimo import SearchConfig
from dimensr_mimo.graph import System
from dimensr_mimo.pareto import violations
from experiments.guarded_selection_20261026 import selector as guarded
from experiments.typed_coverage_20261026 import audit

STRUCTURE=('shared_graph_size','independent_parameters','expanded_total_nodes','maximum_output_depth')


def select(source,limits):
    if set(limits)!=set(STRUCTURE)|{'nrmse'}:raise ValueError('Explicit accuracy and structure limits are required.')
    q=source['n_outputs']
    if len(limits['nrmse'])!=q or any(not math.isfinite(v) or v<0 for v in limits['nrmse']):
        raise ValueError('Accuracy limits must be finite, nonnegative and match outputs.')
    if any(type(limits[k]) is not int or limits[k]<0 for k in STRUCTURE):raise ValueError('Structure limits must be nonnegative integers.')
    fallback=guarded.derive(source,'guarded');guard=fallback['post_selection'];cfg=SearchConfig(**source['config'])
    feasible=[];counts=dict(absolute_accuracy=0,structure=0,relative_constraints=0)
    for i,c in enumerate(source['candidates']):
        errors=c['errors']+c['validation_errors']
        if len(c['errors'])!=q or any(not math.isfinite(v) or v<0 for v in errors):raise ValueError('Invalid candidate errors.')
        system=System.from_dict(c['system']);audit.units(system,source['feature_units'],source['target_units'],source['constant_units'])
        shape=audit.complexity(system)
        absolute=all(a<=b for a,b in zip(c['errors'],limits['nrmse'])) and all(a<=b for a,b in zip(c['validation_errors'],limits['nrmse']))
        simple=all(shape[k]<=limits[k] for k in STRUCTURE)
        protected=(c['violation']==0 and violations(np.asarray(c['errors']),np.asarray(source['reference_train']),cfg)==0 and
            all(a<=b for a,b in zip(c['validation_errors'],guard['validation_bounds'])))
        counts['absolute_accuracy']+=absolute;counts['structure']+=simple;counts['relative_constraints']+=protected
        if absolute and simple and protected:feasible.append((tuple(shape[k] for k in STRUCTURE)+(float(np.mean(c['validation_errors'])),i),i))
    model=deepcopy(fallback)
    index=min(feasible)[1] if feasible else guard['candidate_index']
    if feasible:
        candidate=source['candidates'][index]
        model['best']={k:deepcopy(candidate[k]) for k in guarded.FIELDS}
        model['validation_nrmse']=list(candidate['validation_errors'])
        model['diagnostics'].update(validation_feasible=True,validation_violation=0.,selected_score=candidate['score'],sharing=System.from_dict(candidate['system']).sharing)
    model['post_selection']=dict(method='absolute_precision_then_structure',status='accepted' if feasible else 'not_met',
        limits=deepcopy(limits),candidate_index=index,feasible_count=len(feasible),candidate_count=len(source['candidates']),
        individual_gate_counts=counts,complexity_priority=list(STRUCTURE),
        validation_bounds=guard['validation_bounds'],relative_training_and_validation_constraints=True,
        source_model_sha256=guarded.digest(source),test_data_used=False,search_reused=True,
        fallback_method=None if feasible else 'guarded',reason=None if feasible else 'No candidate satisfies every frozen gate.')
    # diagnostics survive the existing estimator's save/load cycle.
    model['diagnostics']['application_acceptance']=deepcopy(model['post_selection'])
    return model
