"""Choose a small system only inside frozen absolute and relative constraints."""
from copy import deepcopy
import math
import numpy as np
from .config import SearchConfig
from .graph import System
from .pareto import violations
from . import selection as guarded
from . import audit

STRUCTURE=('shared_graph_size','independent_parameters','expanded_total_nodes','maximum_output_depth')


def validate_limits(limits,q=None):
    if not isinstance(limits,dict):raise ValueError('Acceptance limits must be a mapping.')
    if set(limits)!=set(STRUCTURE)|{'nrmse'}:raise ValueError('Explicit accuracy and structure limits are required.')
    if not isinstance(limits['nrmse'],(list,tuple)) or not limits['nrmse'] or (q is not None and len(limits['nrmse'])!=q) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in limits['nrmse']):
        raise ValueError('Accuracy limits must be finite, nonnegative and match outputs.')
    if any(type(limits[k]) is not int or limits[k]<0 for k in STRUCTURE):raise ValueError('Structure limits must be nonnegative integers.')


def select(source,limits):
    q=source['n_outputs'];validate_limits(limits,q)
    fallback=guarded.derive(source,'guarded');guard=fallback['post_selection'];cfg=SearchConfig(**source['config'])
    baseline=None
    if cfg.sharing_strategy=='closure':
        record=source['diagnostics'].get('sharing_closure',{})
        count=record.get('base_candidate_count')
        if type(count) is not int or not 0<count<=len(source['candidates']):
            raise ValueError('Invalid closure base candidate count.')
        base=deepcopy(source);base['candidates']=base['candidates'][:count]
        payload=[{k:c[k] for k in guarded.FIELDS} for c in base['candidates']]
        if guarded.digest(payload)!=record.get('base_candidate_sha256'):
            raise ValueError('Closure base pool differs from its training freeze.')
        base['config']['sharing_strategy']='random'
        if 'precision_repair' in base['config']:
            base['config']['precision_repair']=False
        base['diagnostics'].pop('sharing_closure')
        base['diagnostics']['frozen_candidate_sha256']=guarded.digest(payload)
        anchor=base['candidates'][guarded.choose(base['candidates'],'original')['candidate_index']]
        base['best']={k:deepcopy(anchor[k]) for k in guarded.FIELDS}
        base['validation_nrmse']=list(anchor['validation_errors'])
        base['diagnostics'].update(validation_feasible=anchor['violation']==0,validation_violation=anchor['violation'],
            selected_score=anchor['score'],sharing=System.from_dict(anchor['system']).sharing)
        baseline=select(base,limits)
        guard=baseline['post_selection']
        training_bounds=(np.asarray(baseline['best']['errors'])*1.1+1e-8).tolist()
        validation_bounds=(np.asarray(baseline['validation_nrmse'])*1.1+1e-8).tolist()
        fallback=deepcopy(source)
        fallback['best']=deepcopy(baseline['best']);fallback['validation_nrmse']=list(baseline['validation_nrmse'])
        for key in ('validation_feasible','validation_violation','selected_score','sharing'):
            fallback['diagnostics'][key]=deepcopy(baseline['diagnostics'][key])
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
        if baseline is not None:
            protected=protected and all(a<=b for a,b in zip(c['errors'],training_bounds)) and all(
                a<=b for a,b in zip(c['validation_errors'],validation_bounds))
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
    if baseline is not None:
        model['post_selection'].update(base_candidate_count=count,base_candidate_sha256=record['base_candidate_sha256'],
            base_candidate_index=guard['candidate_index'],base_training_bounds=training_bounds,
            base_validation_bounds=validation_bounds,base_choice_sha256=guarded.digest(baseline['best']),
            fallback_method=None if feasible else 'base_precision',test_protection_guaranteed=False)
    # diagnostics survive the existing estimator's save/load cycle.
    model['diagnostics']['application_acceptance']=deepcopy(model['post_selection'])
    return model
