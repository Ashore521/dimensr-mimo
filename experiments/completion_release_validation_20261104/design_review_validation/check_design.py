"""Static/five-point design review only. No fit, preflight or experimental scoring."""
from collections import Counter
from fractions import Fraction
import ast,hashlib,importlib.util,json,re,sys,time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.completion_release_validation_20261104 import cases
from experiments.compositional_completion_20261031 import harness
from dimensr_mimo import SearchConfig
from types import SimpleNamespace

read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
issues=[]
def check(ok,label):
    if not ok:issues.append(label)
def canon(value):return json.dumps(value,sort_keys=True)

def main():
    out=HERE/'DESIGN_CHECKS.json'
    if out.exists():raise RuntimeError('Keep design-review checkpoint.')
    names=['PLAN.md','CANDIDATE_LOCK.json','cases.py','test_cases.py','NOVELTY_AUDIT.json','ORACLE_AUDIT.json','case_validation/pytest_01.log']
    initial={n:sha(HERE/n) for n in names}
    lock=read(HERE/'CANDIDATE_LOCK.json');parent=ROOT/'experiments/compositional_completion_20261031/CANDIDATE_LOCK.json'
    actual={p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')}
    check(len(actual)==len(lock['source_files'])==41 and actual==set(lock['source_files']),'complete_41_source_set')
    source_bad=[n for n,h in lock['source_files'].items() if sha(ROOT/n)!=h]
    check(not source_bad,'source_bytes_match')
    check(sha(parent)==lock['parent_candidate_lock_sha256'] and read(parent)['source_files']==lock['source_files'],'parent_lock_identity')
    birth={n:(HERE/n).stat().st_birthtime for n in ['CANDIDATE_LOCK.json','PLAN.md','cases.py','test_cases.py']}
    check(lock['locked_unix']<=birth['CANDIDATE_LOCK.json']<=min(birth[n] for n in ['PLAN.md','cases.py','test_cases.py']),'timestamps_support_lock_before_task_files')
    oracle=read(HERE/'ORACLE_AUDIT.json')
    check(oracle['cases_sha256']==initial['cases.py'] and oracle['test_sha256']==initial['test_cases.py'] and oracle['test_log_sha256']==initial['case_validation/pytest_01.log'],'oracle_input_bindings')
    helper=ROOT/'experiments/new_task_validation_20261030/post_review_audit.py'
    module_spec=importlib.util.spec_from_file_location('_design_static_raw_helpers',helper)
    helpers=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(helpers);helpers.check=check
    config=harness.configuration(SimpleNamespace(ROOT=ROOT))
    check(config['wall_seconds']==52 and config['max_evaluations']==1200 and config['population_size']==24 and config['epochs']==4,'original_config_budget')
    check(config['initialization']=='compound' and config['sharing_strategy']=='closure' and config['precision_repair'] is False and config['compositional_completion'] is False,'base_configuration')
    check(SearchConfig().initialization=='legacy' and SearchConfig().sharing_strategy=='random' and SearchConfig().compositional_completion is False,'opt_in_defaults')
    X=np.array([[.2,.2],[.2,1.4],[1.4,.2],[1.4,1.4],[.7,.8]])
    a,b=X.T
    direct={
      'log_affine_ratio':np.column_stack((np.log((1+.4*a)/(1+.7*b)),a*np.log((1+.4*a)/(1+.7*b)))),
      'reciprocal_quadratic_norm':np.column_stack((1/np.sqrt(1+a*a),b/np.sqrt(1+a*a))),
      'oscillatory_affine_saturation':np.column_stack((np.sin(.6*a)/(1+.5*b),np.cos(.6*a)/(1+.5*b))),
      'exponential_affine_saturation':np.column_stack((np.exp(-.4*a)/(1+.6*a),b*np.exp(-.4*a)/(1+.6*a))),
    }
    oracle_ix={r['family']:r for r in oracle['rows']};rows=[]
    for family in cases.FAMILIES:
        spec=cases.design(family+'_n0');graph,constants=cases.truth_graph(family+'_n0');payload=graph.to_dict()
        shape,sharing=helpers.graph_audit(payload,spec,family)
        error=float(np.max(np.abs(helpers.raw(payload,constants,X)-direct[family])))
        check(error<=1e-14 and np.isfinite(direct[family]).all(),family+':independent_five_point_raw')
        check(np.array_equal(cases.latent(family+'_n0',X),cases.latent(family+'_n1',X)),family+':same_latent_across_noise')
        check(shape==oracle_ix[family]['structure'],family+':oracle_structure')
        check(all(shape[k]<=spec['acceptance'][k] for k in ['shared_graph_size','independent_parameters','expanded_total_nodes','maximum_output_depth']),family+':structure_thresholds')
        for noise in [0,1]:
            name=family+'_n'+str(noise);sp=cases.design(name)
            expected=dict(nrmse=[.03 if noise else 1e-4]*2,shared_graph_size=24,independent_parameters=6,expanded_total_nodes=30,maximum_output_depth=6)
            check(sp['acceptance']==expected and sp['clean_test_nrmse']==[.02 if noise else 1e-4]*2 and sp['noise_fraction']==.01*noise,name+':fixed_thresholds')
            check(sp['bounds']==[[.2,1.4],[.2,1.4]] and len(sp['features'])==len(sp['targets'])==2,name+':bounds_dimensions')
            cfg=harness.effective_config(SimpleNamespace(METHODS=('base','completion'),cases=cases),dict(case=name,seed=10241,method='completion'),config)
            cfg.validate()
        rows.append(dict(family=family,independent_shape=shape,independent_sharing=sharing,max_five_point_raw_difference=error,declared_units=spec['targets']))
    matrix=harness.matrix(SimpleNamespace(cases=cases,METHODS=('base','completion')))
    check(len(cases.CASES)==8 and cases.SEEDS==(10241,10242) and len(matrix)==32 and len({(j['case'],j['seed'],j['method']) for j in matrix})==32,'16_pairs_32_systems')
    check(all(j['seconds']==60 for j in matrix) and Counter(j['method'] for j in matrix)=={'base':16,'completion':16},'balanced_matrix')
    # Static generator checks; no experimental train/test arrays generated here.
    tree=ast.parse((HERE/'cases.py').read_text(encoding='utf-8'));functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    generate=functions['generate'];first=generate.body[0]
    check(isinstance(first,ast.If) and any(isinstance(x,ast.Raise) for x in ast.walk(first)) and 'ALL_MODELS_FROZEN.json' in ast.unparse(first),'test_guard_precedes_generation')
    check('96' in ast.unparse(functions['inputs']) and '48' in ast.unparse(functions['inputs']) and '192' in ast.unparse(functions['inputs']),'96_48_192_source_counts')
    check('[20261104, seed, code, CASES.index(name)]' in ast.unparse(functions['inputs']) and '[20261104, seed, code, CASES.index(name)]' in ast.unparse(generate),'distinct_noise_seed_split_stream_key')
    novelty=read(HERE/'NOVELTY_AUDIT.json')
    history_bad=[n for n,h in novelty['source_files'].items() if not (HERE/'history_sources'/n).is_file() or sha(HERE/'history_sources'/n)!=h]
    live_history_bad=[n for n,h in novelty['source_files'].items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    check(len(novelty['source_files'])==54 and not history_bad,'54_history_original_byte_copies')
    check(len([p for p in (HERE/'history_sources').rglob('*') if p.is_file()])==54,'history_archive_members')
    historical=[];freeze_hashes={}
    for path in sorted((ROOT/'experiments').glob('*/FREEZE.json')):
        frozen=read(path)
        for name,sp in frozen.get('specs',{}).items():
            historical.append(dict(source=path.relative_to(ROOT).as_posix(),case=name,formulas=sp.get('formulas',[]),features=sp['features'],targets=sp['targets']))
            freeze_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
    check(Counter(map(canon,historical))==Counter(map(canon,novelty['historical_spec_records'])) and len(historical)==120,'120_registered_records_reconstructed')
    pair_hits={}
    for family in cases.FAMILIES:
        sp=cases.design(family+'_n0')
        pair_hits[family]=[r for r in historical if all(r[k]==sp[k] for k in ['formulas','features','targets'])]
        check(not pair_hits[family],family+':no_exact_formula_pair_input_output_unit_hits')
    check(all(not r['exact_registered_pair_hits'] for r in novelty['comparisons']),'audit_zero_hits_agrees')
    unique=len({canon({k:r[k] for k in ['formulas','features','targets']}) for r in historical})
    omitted=[r for r in historical if not r['formulas']]
    omitted_unit_matches={family:[r for r in omitted if all(r[k]==cases.design(family+'_n0')[k] for k in ['features','targets'])] for family in cases.FAMILIES}
    plan=(HERE/'PLAN.md').read_text(encoding='utf-8')
    check(all(x in plan for x in ['2880','600','1200','52','60','16/16','32/32','1.1+1e-8']),'planned_budget_and_scientific_gates')
    check(not any((HERE/n).exists() for n in ['PREFLIGHT.json','FREEZE.json','BUDGET.json','TRAINING_STARTED.json','ALL_MODELS_FROZEN.json','SCORES.json','VERIFICATION.json']),'no_formal_execution')
    metadata_path=Path('C:/Users/608/.codex/sessions/2026/10/30/rollout-2026-10-30T17-07-49-01a18c10-baea-7b10-bd6d-96f104545fe5.jsonl')
    meta=None
    for number,line in enumerate(metadata_path.open(encoding='utf-8'),1):
        value=json.loads(line)
        if value.get('type')=='turn_context':
            q=value['payload'];meta=dict(path=str(metadata_path),line=number,timestamp=value.get('timestamp'),model=q.get('model'),effort=q.get('effort'))
    check(meta['model']=='gpt-6.1-sol' and meta['effort']=='high' and meta['timestamp'].startswith('2026-11-04'),'current_runtime_sol_high')
    check(all(sha(HERE/n)==h for n,h in initial.items()),'assigned_inputs_unchanged')
    result=dict(status='design_supported_pending_harness_review' if not issues else 'design_findings',issues=issues,checked_unix=time.time(),runtime_metadata=meta,
      input_sha256=initial,parent_candidate_lock_sha256=sha(parent),source_count=41,source_hash_mismatches=source_bad,source_manifest=lock['source_files'],file_birth_times=birth,
      constructed_matrix=matrix,config=config,fixed_points=X.tolist(),oracle_independent_rows=rows,
      analytic_domain_bounds=dict(log_numerator=[1.08,1.56],log_denominator=[1.14,1.98],log_ratio=[1.08/1.98,1.56/1.14],reciprocal_radicand=[1.04,2.96],oscillatory_denominator=[1.1,1.7],oscillatory_argument=[.12,.84],exponential_argument=[-.56,-.08],exponential_denominator=[1.12,1.84]),
      novelty=dict(archived_source_files=54,archive_mismatches=history_bad,live_history_mismatches=live_history_bad,registered_records=120,records_with_formula_pairs=120-len(omitted),records_without_formula_pairs=len(omitted),formula_omitted_records_matching_new_units=omitted_unit_matches,distinct_formula_pair_input_output_unit_records=unique,source_freeze_count=len(freeze_hashes),source_freeze_sha256=freeze_hashes,exact_hits=pair_hits,scope=novelty['scope']),
      tests_log_sha256=sha(HERE/'design_review_validation/pytest_review.log'),tests_log=(HERE/'design_review_validation/pytest_review.log').read_text(encoding='utf-8').strip(),
      helper_sha256=sha(helper),script_sha256=sha(Path(__file__)),no_experimental_scoring_fit_search_or_preflight=True,harness_implementation_not_reviewed=True,execution_clearance=False)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],issues=issues,source_count=41,history_records=120,unique_formula_pairs=unique,history_live_mismatches=live_history_bad,runtime=meta,sha256=sha(out))))

if __name__=='__main__':main()
