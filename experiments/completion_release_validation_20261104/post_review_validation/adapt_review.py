"""Adapt the previously independent review to this new-task contract only."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE/'audit.py'
s=p.read_text(encoding='utf-8')
s=s.replace('Independent sealed development reconstruction.','Independent sealed new-output reconstruction.')
s=s.replace("PRIOR=ROOT/'experiments/new_task_validation_20261030'","PRIOR=ROOT/'experiments/completion_development_verified_20261031'")
s=s.replace("helpers.check=check","helpers.check=check;helpers.data=data")
s=s.replace("    normalizer_path=ROOT/'experiments/completion_historical_20261031/post_review_validation/audit.py'\n    oldaudit=load_helper(normalizer_path,'_private_history_normalization_helpers')\n",'')
s=s.replace("archive['file_count']==4084","archive['file_count']==8691").replace("len(f['files'])==archive['file_count']==4084","len(f['files'])==archive['file_count']==8691").replace("'4084_input_manifest'","'8691_input_manifest'")
s=s.replace("f['candidate_lock_sha256']==acceptance['candidate_lock_sha256']","f['candidate_lock_sha256']==acceptance['source_lock_sha256']")
s=s.replace("acceptance['status']=='accepted_for_single_development_execution' and acceptance['accepted_unix']<f['budget']['start_unix'] and acceptance['independent_review_sha256']=={n:sha(HERE/n) for n in ['HARNESS_REVIEW.md','HARNESS_REVIEW_CHECKS.json']}","acceptance['status']=='accepted_for_single_locked_candidate_validation' and acceptance['accepted_unix']<f['budget']['start_unix'] and acceptance['harness_review_sha256']==sha(HERE/'HARNESS_REVIEW.md') and acceptance['harness_review_checks_sha256']==sha(HERE/'HARNESS_REVIEW_CHECKS.json')")
s=s.replace("    check(acceptance['test_validation_sha256']==sha(HERE/'TEST_VALIDATION.json'),'pre_run_tests_binding')", "    check(acceptance['test_validation_sha256']==sha(HERE/'TEST_VALIDATION.json'),'pre_run_tests_binding')\n    check(acceptance['prior_acceptance_sha256']==sha(PRIOR/'POST_REVIEW_ACCEPTANCE.json') and acceptance['local_candidate_lock_sha256']==sha(HERE/'CANDIDATE_LOCK.json'),'eight_key_prerequisite_and_local_lock')")
s=s.replace(";identity=[];margins={k:[] for k in ['current_neural','current_gp','old_neural','old_gp']};data_hashes={}",";data_hashes={}")
s=s.replace("    old_score_ix={key(r['job']):r for r in read(PRIOR/'SCORES.json')['rows']}\n",'')
start=s.index("        if job['method']=='base':\n            oldfolder=PRIOR/")
end=s.index("        info=dict(triggered=",start)
s=s[:start]+s[end:]
s=s.replace("    check(all(min(v)>0 for v in margins.values()),'all_control_phase_deadlines_nonbinding')\n",'')
s=s.replace("index={key(r['job']):r for r in results};pairs=[]","index={key(r['job']):r for r in results};pairs=[];selected_pairs=[]")
needle="        check(p==row,'paired_summary:'+case+str(seed));pairs.append(p)"
s=s.replace(needle,needle+"\n        same=dict(case=case,seed=seed,graph_equal=ma['best']['system']==mb['best']['system'],constants_equal=ma['best']['constants']==mb['best']['constants'],best_payload_equal=ma['best']==mb['best'],base_selected_is_foundation=any(c['system']==mb['best']['system'] and c['constants']==mb['best']['constants'] for c in pb),completion_selected_is_foundation=any(c['system']==ma['best']['system'] and c['constants']==ma['best']['constants'] for c in pa))\n        selected_pairs.append(same)")
s=s.replace(" and len(identity)==16",'')
start=s.index("    check(technical==summary['technical_pass']")
end=s.index("    integrity['final_live']",start)
s=s[:start]+'''    check(technical==summary['technical_pass'] is True and adoption==summary['adoption_pass'] is False and summary['protected_outputs']==protected==32,'scientific_gates')
    for name in ['full_denominator','paired_base_identity_pass','closure_mechanism_pass','control_acceptance_preserved','macro_clean_error_nonincrease','mean_graph_nonincrease']:check(summary[name] is True,'summary_'+name)
    check(summary['candidate_all_accepted'] is False,'candidate_all_accepted_false')
    check('archived_control_identity' not in json.dumps(summary)+json.dumps(verified) and summary['historical_control_comparison']=='not_applicable_no_same_task_archive','no_fake_archived_control')
    failures=[]
    for r in results:
        if r['job']['method']!='completion' or r['accepted']:continue
        spec=f['specs'][r['job']['case']]
        reasons=[k+' exceeds '+str(limit) for k,limit in spec['acceptance'].items() if k!='nrmse' and r['shape'][k]>limit]
        for label,errors,limits in [('train',r['train_nrmse'],spec['acceptance']['nrmse']),('validation',r['validation_nrmse'],spec['acceptance']['nrmse']),('observed_test',r['observed_nrmse'],spec['acceptance']['nrmse']),('clean_test',r['clean_nrmse'],spec['clean_test_nrmse'])]:
            reasons.extend(label+' output'+str(i)+' exceeds '+str(limit) for i,(e,limit) in enumerate(zip(errors,limits)) if not math.isfinite(e) or e<0 or e>limit)
        saved=next(x for x in summary['completion_failures'] if x['job']==r['job'])
        check(reasons==saved['failure_reasons'],str(r['job'])+':failure_reasons')
        failures.append(dict(job=r['job'],failure_reasons=reasons,shape=r['shape'],train_nrmse=r['train_nrmse'],validation_nrmse=r['validation_nrmse'],observed_nrmse=r['observed_nrmse'],clean_nrmse=r['clean_nrmse']))
    check(len(failures)==len(summary['completion_failures'])==11,'all_11_failures')
    family_acceptance={family:{method:sum(r['accepted'] for r in results if r['job']['method']==method and f['specs'][r['job']['case']]['family']==family) for method in ['base','completion']} for family in families}
''' +s[end:]
s=s.replace("    check(not (ROOT/'experiments/completion_release_validation_20261104/FREEZE.json').exists(),'release_validation_not_started')","    check(not (HERE/'POST_REVIEW_ACCEPTANCE.json').exists(),'review_did_not_write_acceptance')")
s=s.replace(",normalizer_helper_sha256=sha(normalizer_path)",'')
s=s.replace("control_identity=identity,control_phase_margins={k:dict(count=len(v),minimum=min(v),maximum=max(v)) for k,v in margins.items()},", "historical_control_comparison='not_applicable_no_same_task_archive',selected_pair_identity=selected_pairs,completion_failures=failures,family_acceptance=family_acceptance,")
s=s.replace("no_acceptance_written=True)","no_acceptance_written=True,adapted_from_independent_audit_sha256=sha(ROOT/'experiments/completion_development_verified_20261031/post_review_validation/audit.py'))")
data='''
def data(name,seed):
    families=('log_affine_ratio','reciprocal_quadratic_norm','oscillatory_affine_saturation','exponential_affine_saturation')
    cases=tuple(f'{f}_n{n}' for f in families for n in [0,1]);family=name.rsplit('_n',1)[0]
    def inputs(split):
        code,count={'train':(111,96),'validation':(222,48),'test':(333,192)}[split]
        rng=np.random.default_rng(np.random.SeedSequence([20261104,seed,code,cases.index(name)]))
        return rng.uniform(np.array([.2,.2]),np.array([1.4,1.4]),(count,2))
    def latent(X):
        a,b=X.T
        if family=='log_affine_ratio':
            z=np.log((1+.4*a)/(1+.7*b));return np.column_stack((z,a*z))
        if family=='reciprocal_quadratic_norm':
            z=np.sqrt(1+a*a);return np.column_stack((1/z,b/z))
        if family=='oscillatory_affine_saturation':return np.column_stack((np.sin(.6*a)/(1+.5*b),np.cos(.6*a)/(1+.5*b)))
        z=np.exp(-.4*a)/(1+.6*a);return np.column_stack((z,b*z))
    scale=latent(inputs('train')).std(axis=0);out={}
    for split in ['train','validation','test']:
        X=inputs(split);clean=latent(X)
        rng=np.random.default_rng(np.random.SeedSequence([20261104,seed,{'train':411,'validation':422,'test':433}[split],cases.index(name)]))
        observed=clean+(.01 if name.endswith('_n1') else 0.)*scale*rng.normal(size=clean.shape)
        out[split]=(X,observed,clean)
    return out

'''
s=s.replace('def main():\n',data+'def main():\n',1)
p.write_text(s,encoding='utf-8')
