"""Read-only terminal receipt comparison; never calls formal run/verify or fits."""
from collections import Counter
import hashlib,json,math,time
from pathlib import Path

HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
OLD=ROOT/'experiments/completion_historical_20261031'
PRIOR=ROOT/'experiments/precision_repair_20261028'
OUT=HERE/'post_review_checks.json'
issues=[]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
key=lambda j:(j['case'],j['seed'],j['method'])

def check(ok,name):
    if not ok:issues.append(name)

def hashes(files,root):
    bad=[n for n,h in files.items() if not (root/n).is_file() or sha(root/n)!=h]
    check(not bad,'hashes:'+str(root));return bad

def close(a,b,name):
    if isinstance(a,list):
        check(len(a)==len(b),name+':length')
        for i,(x,y) in enumerate(zip(a,b)):close(x,y,name+':'+str(i))
    else:check(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14),name)

def complete(row):
    try:
        errors=row['clean_nrmse']+row['observed_nrmse']
        return (row['status']=='complete' and len(row['clean_nrmse'])==len(row['observed_nrmse'])==2
                and all(math.isfinite(x) and x>=0 for x in errors+[row['mean_clean_nrmse']])
                and type(row['both_accepted']) is bool
                and all(type(row['structure'][k]) is int and row['structure'][k]>=0 for k in
                        ['shared_graph_size','independent_parameters','expanded_total_nodes','maximum_output_depth']))
    except (KeyError,TypeError,ValueError):return False

def main():
    if OUT.exists():raise RuntimeError('Keep terminal review checkpoint.')
    f,v,s,e,a=(read(HERE/n) for n in ['FREEZE.json','VERIFICATION.json','SUMMARY.json','EXECUTION.json','FROZEN_INPUT_ARCHIVE.json'])
    of,ov,os,score,om,oldreview=(read(OLD/n) for n in ['FREEZE.json','VERIFICATION.json','SUMMARY.json','SCORES.json','ALL_MODELS_FROZEN.json','post_review_checks.json'])
    prereview=read(HERE/'PRE_RUN_CHECKS.json');jobs=of['jobs']
    before={n:sha(HERE/n) for n in ['FREEZE.json','VERIFICATION.json','SUMMARY.json','EXECUTION.json','FROZEN_INPUT_ARCHIVE.json','verification.log']}
    check(len(f['files'])==a['file_count']==245 and f['files']==a['files'],'new_input_manifest')
    check(set(f['files'])=={p.relative_to(HERE/'frozen_inputs').as_posix() for p in (HERE/'frozen_inputs').rglob('*') if p.is_file()},'exact_new_archive_members')
    integrity=dict(new_live=hashes(f['files'],ROOT),new_archive=hashes(f['files'],HERE/'frozen_inputs'),
                   original_live=hashes(of['files'],ROOT),original_archive=hashes(of['files'],OLD/'frozen_inputs'),original_runs=hashes(om['files'],OLD))
    check(len(of['files'])==966 and len(om['files'])==374,'original_counts')
    check(a['freeze_sha256']==v['freeze_sha256']==sha(HERE/'FREEZE.json'),'new_freeze_chain')
    check(a['status']=='archived_original_bytes' and a['original_historical_inputs_and_runs_unchanged'] is True,'archive_status')
    check(v['original_sha256']==f['original_sha256'],'original_chain_copied')
    check(all(sha(OLD/n)==h for n,h in f['original_sha256'].items()),'original_chain_unchanged')
    check(all(sha(OLD/n)==h for n,h in oldreview['key_sha256'].items()),'original_review_receipt_chain')
    check(of['frozen_unix']<om['frozen_unix']<=score['scored_unix']<=ov['verified_unix']<f['frozen_unix']<v['verified_unix'],'old_new_chronology')
    budget=read(OLD/'BUDGET.json');deadline=budget['start_unix']+budget['limit_seconds']
    end=f['frozen_unix']+e['wall_seconds']
    check(f['original_deadline_unix']==deadline and f['deadline_unix']==min(f['frozen_unix']+600,deadline) and f['verification_cap_seconds']==600,'original_deadline_formula')
    check(f['frozen_unix']<v['verified_unix']<=end<f['deadline_unix'] and e['wall_seconds']<=600,'completion_within_bound')
    check(e['returncode']==0 and e['termination_reason'] is None and e['verification_passed'] is True and e['no_training_performed'] is True,'execution_receipt')
    check(v['status']=='passed' and v['expected_systems']==v['verified_systems']==34 and v['budget_met'] is True and v['inputs_unchanged'] is True,'new_passed_receipt')
    check(ov['status']=='failed' and ov['verified_systems']==17 and os['technical_pass'] is False and os['adoption_pass'] is False,'original_failed_preserved')
    check(v['original_verdict']==s['original_verdict']==f['original_verdict']=='failed' and s['verification_of_sealed_results'] is True and s['development_authorized'] is False,'verdict_contract_separation')
    check(len(v['rows'])==34 and [r['job'] for r in v['rows']]==jobs and len({key(r['job']) for r in v['rows']})==34,'34_ordered_unique_full_rows')
    newix={key(r['job']):r for r in v['rows']};obsix={key(r['job']):r for r in oldreview['systems']};oldix={key(r['job']):r for r in ov['rows']};scoreix={key(r['job']):r for r in score['rows']}
    prior=read(PRIOR/'VERIFICATION.json');priorix={key(r['job']):r for r in prior['rows']}
    prior_scores=read(PRIOR/'SCORES.json')['rows']
    check(len(prior_scores)==len({key(r['job']) for r in prior_scores})==34 and all(complete(r) for r in prior_scores),'old_score_full_denominator')
    check([r['job'] for r in score['rows']]==jobs and all(complete(r) for r in score['rows']),'unchanged_score_full_denominator')
    required={'job','verified','clean_nrmse','observed_nrmse','both_accepted','structure','mechanism_pass','domain_pass','candidate_count','absolute_structure_feasible','application_acceptance','physical_acceptance','closure'}
    rows=[]
    for job in jobs:
        k=key(job);row=newix[k];ob=obsix[k];sc=scoreix[k];folder=OLD/'runs'/('_'.join([job['case'],'s'+str(job['seed']),job['method']]))
        model=read(folder/'model.json');pool=read(folder/'pool.json');physical=read(folder/'physical.json')
        check(required<=set(row) and row['verified'] is True and 'error' not in row and row['mechanism_pass'] is True and row['domain_pass'] is True,'full_row:'+str(k))
        check(row['structure']==ob['structure']==sc['structure'] and row['both_accepted']==ob['both_accepted']==sc['both_accepted'],'selected_identity:'+str(k))
        close(row['clean_nrmse'],ob['clean_nrmse'],'independent_clean:'+str(k));close(row['observed_nrmse'],ob['observed_nrmse'],'independent_observed:'+str(k))
        check(row['closure']==ob['actual_closure'],'independent_matching:'+str(k))
        check(row['candidate_count']==pool['candidates']==ob['saved_candidates'] and row['absolute_structure_feasible']==pool['feasible_count'],'pool_fields:'+str(k))
        check(row['application_acceptance']==model['post_selection']['status'] and row['physical_acceptance']==physical['physical_acceptance']['status'],'acceptance_fields:'+str(k))
        if job['method']=='completion':check(row==oldix[k],'candidate_formal_row_exactly_preserved:'+str(k))
        else:
            check(row['archived_control_identity'] is True,'repaired_control_identity:'+str(k))
            priorrow=priorix[job['case'],job['seed'],'repair']
            check(all(row[k2]==val for k2,val in priorrow.items() if k2 not in ('job','closure')),'prior_repair_fields:'+str(k))
            check(all(row['repair'][q]==val for q,val in ob['actual_supplement'].items()),'independent_repair_counts:'+str(k))
        rows.append(dict(job=job,verified=True,selected_predictions_and_structure_agree=True,matching_counters_agree=True,candidate_count=row['candidate_count']))
    changed={k for k in set(s)|set(os) if s.get(k)!=os.get(k)}
    expected_changes={'technical_pass','adoption_pass','archived_control_identity_pass','closure_mechanism_pass','closure_by_method','original_verdict','verification_of_sealed_results','development_authorized'}
    check(changed==expected_changes,'only_expected_summary_fields_changed')
    check(s['methods']==oldreview['methods_recomputed'] and s['pairs']==oldreview['pairs_recomputed'] and s['historical_targets']==oldreview['historical_targets_recomputed'] and s['completion_discovery_by_family']==oldreview['family_discovery_recomputed'],'independent_summary_values')
    check(s['closure_by_method']==oldreview['supplemental_observations_not_official_reclassification']['actual_matching'],'independent_both_arm_matching')
    check(s['full_denominator'] is True and s['expected_systems']==s['expected_outputs']==34 and s['expected_pairs']==17,'summary_denominators')
    for n in ['technical_pass','adoption_pass','paired_base_identity_pass','archived_control_identity_pass','closure_mechanism_pass','control_acceptance_preserved','macro_clean_error_nonincrease','mean_graph_nonincrease']:
        check(s[n] is True,'scientific_gate:'+n)
    check(s['protected_outputs']==34 and all(s['historical_targets'].values()) and s['candidate_all_accepted'] is False and len(s['completion_failures'])==4,'remaining_failures_and_targets')
    check(sum(r['candidate_count'] for r in v['rows'])==oldreview['saved_candidate_count']==5508,'candidate_count_total')
    check(all(prereview['checks'].values()) and len(prereview['control_pairs'])==17 and all(p['phase_preconditions_and_normalized_identity'] for p in prereview['control_pairs']),'reuse_bound_timing_preconditions')
    check(all(sha(HERE/n)==h for n,h in prereview['sha256'].items()),'reviewed_code_and_tests_unchanged')
    lines=(HERE/'verification.log').read_text(encoding='utf-8').splitlines()
    check(lines==['_'.join([j['case'],'s'+str(j['seed']),j['method']])+' verified' for j in jobs],'34_success_log_lines')
    check(not any((HERE/n).exists() for n in ['SCORES.json','TRAINING_STARTED.json','ALL_MODELS_FROZEN.json','runs']),'no_new_training_or_score_outputs')
    check(not (OLD/'POST_REVIEW_ACCEPTANCE.json').exists(),'no_original_acceptance')
    development=ROOT/'experiments/completion_development_20261031'
    check(not any((development/n).exists() for n in ['PREFLIGHT.json','BUDGET.json','FREEZE.json','TRAINING_STARTED.json','ALL_MODELS_FROZEN.json']),'development_not_started')
    integrity['final_new_live']=hashes(f['files'],ROOT);integrity['final_new_archive']=hashes(f['files'],HERE/'frozen_inputs')
    integrity['final_original_live']=hashes(of['files'],ROOT);integrity['final_original_archive']=hashes(of['files'],OLD/'frozen_inputs');integrity['final_original_runs']=hashes(om['files'],OLD)
    check(all(sha(HERE/n)==h for n,h in before.items()),'new_result_receipts_unchanged_during_review')
    result=dict(status='supported_isolated_verification_pass' if not issues else 'review_findings',issues=issues,
        independent_scope='New/old hashes, receipt chronology/deadline, all 34 row fields, old full score denominator, exact candidate rows and allowed summary differences, comparison with previous independent numerical/event reconstruction. No formal entry point, fit/search or data generation.',
        reused_evidence=dict(original_post_review_sha256=sha(OLD/'POST_RUN_REVIEW.md'),original_checks_sha256=sha(OLD/'post_review_checks.json'),pre_run_review_sha256=sha(HERE/'PRE_RUN_REVIEW.md'),pre_run_checks_sha256=sha(HERE/'PRE_RUN_CHECKS.json'),selected_graphs=34,supplement_events=161,matching_events=784,all_candidate_metrics_independently_recomputed=False),
        source_hashes=prereview['sha256'],new_receipt_sha256=before,original_receipt_sha256=f['original_sha256'],integrity=integrity,
        chronology=dict(original_model_seal=om['frozen_unix'],original_scores=score['scored_unix'],original_failed_verification=ov['verified_unix'],new_freeze=f['frozen_unix'],new_verified=v['verified_unix'],recorded_outer_end=f['frozen_unix']+e['wall_seconds'],original_deadline=deadline,outer_deadline_margin_seconds=deadline-end),
        new_verified_systems=34,rows=rows,summary_changed_fields=sorted(changed),methods=s['methods'],actual_matching=s['closure_by_method'],family_discovery=s['completion_discovery_by_family'],
        original_verdict='failed',isolated_verification_verdict='passed',development_authorized=False,requires_separate_continuation_review=True,checked_unix=time.time(),script_sha256=sha(Path(__file__)))
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],issues=issues,sha256=sha(OUT),deadline_margin_seconds=deadline-end)))

if __name__=='__main__':main()
