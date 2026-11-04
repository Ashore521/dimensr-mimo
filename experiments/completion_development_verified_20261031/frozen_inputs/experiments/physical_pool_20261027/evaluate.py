"""Score only sealed deliveries; independently recompute every physical candidate."""
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from dimensr_mimo import DimenSRMIMORegressor,PhysicalModel,AffineCoordinates,audit
from dimensr_mimo.physical import restore_graph
from dimensr_mimo.graph import System
from dimensr_mimo.selection import digest,FIELDS
from dimensr_mimo.acceptance import STRUCTURE
from experiments.native_api_20261026.verify import raw,require
from experiments.physical_pool_20261027 import data,study
read,write,sha=data.read,data.write,data.sha


def error(pred,true,scale):return (np.sqrt(np.mean((pred-true)**2,axis=0))/scale).tolist()


def guards():
    frozen=read(HERE/'FREEZE.json');sealed=read(HERE/'ALL_MODELS_FROZEN.json')
    study.verify_files(frozen['files']);study.verify_files(frozen['sources']);study.verify_files(sealed['files'],HERE)
    require(sealed['freeze_sha256']==sha(HERE/'FREEZE.json'),'Broken freeze chain.')
    require(not frozen['test_data_used'] and not sealed['test_data_used'],'Test isolation record differs.')
    require(frozen['frozen_unix']<=read(HERE/'TRAINING_STARTED.json')['start_unix']<=sealed['frozen_unix'],'Freeze chronology differs.')
    return frozen,sealed


def score_job(job):
    folder=HERE/'runs'/data.name(job);resource=read(folder/'resources.json')
    row=dict(job=job,status=resource['status'],resource=resource)
    if resource['status']!='complete':return row
    spec=data.spec(job);X,Y=data.arrays(job,'test');clean=data.clean(job,X)
    for name in ('control','selected'):
        model=PhysicalModel.load(folder/(name+'.json'));record=read(folder/(name+'.json'))
        prediction=raw(record['system'],record['constants'],X)
        np.testing.assert_array_equal(prediction,model.predict(X))
        scale=np.asarray(model.source['nrmse_scale_physical'])
        observed=error(prediction,Y,scale);latent=error(prediction,clean,scale) if clean is not None else None
        test_ok=all(a<=b for a,b in zip(observed,spec['limits']['nrmse'])) and all(model.structure[k]<=spec['limits'][k] for k in STRUCTURE)
        if latent is not None:test_ok=test_ok and all(a<=b for a,b in zip(latent,spec['clean_limits']))
        row[name]=dict(observed_nrmse=observed,clean_nrmse=latent,comparison_nrmse=latent if latent is not None else observed,
            structure=model.structure,development_accepted=model.acceptance['status']=='accepted',test_accepted=test_ok,
            both_accepted=model.acceptance['status']=='accepted' and test_ok)
    a,b=np.asarray(row['selected']['comparison_nrmse']),np.asarray(row['control']['comparison_nrmse'])
    row['protected']=(a<=b*1.1+1e-8).tolist()
    row['structure_nonincreasing']=tuple(row['selected']['structure'][k] for k in STRUCTURE)<=tuple(row['control']['structure'][k] for k in STRUCTURE)
    return row


def verify_job(job,frozen,score):
    folder=HERE/'runs'/data.name(job);study.verify_files(read(folder/'CELL_FROZEN.json')['files'],folder)
    result=read(folder/'result.json');report=read(folder/'selection.json');model=DimenSRMIMORegressor.load(ROOT/job['source'])
    require(result['source_sha256']==frozen['sources'][job['source']],'Source model drift.')
    require(result['searches']==0 and not result['coefficients_refitted'] and result['source_model_unchanged'],'Search/refit occurred.')
    require(report['search_fingerprint']==model.search_fingerprint_==result['search_fingerprint'],'Search fingerprint differs.')
    require(report['pool_sha256']==model.diagnostics_['frozen_candidate_sha256'],'Pool hash differs.')
    spec=data.spec(job);coordinates=AffineCoordinates(**spec['coordinates']);train=data.arrays(job,'train');validation=data.arrays(job,'validation')
    metadata=frozen['metadata'][data.name(job)]
    require(spec==metadata['spec'],'Frozen physical specification differs.')
    require({key:audit.data_digest(*pair) for key,pair in (('training',train),('validation',validation))}==metadata['data_sha256']==result['data_sha256'],'Physical data drift.')
    scale=model.scales_*np.asarray(coordinates.y_scale);limits=spec['limits'];records=report['rows'];feasible=[];valid=[]
    require(len(records)==len(model.candidates_)==report['candidate_count'],'Incomplete pool audit.')
    for i,(candidate,row) in enumerate(zip(model.candidates_,records)):
        require(row['candidate_index']==i and row['candidate_sha256']==digest({k:candidate[k] for k in FIELDS}),'Candidate identity differs.')
        graph,constants,catalog,roles=restore_graph(System.from_dict(candidate['system']),candidate['constants'],
            model.feature_units_,model.target_units_,model.constant_units_,coordinates)
        audits=graph.audit(constants,spec['bounds'],max_boxes=128)
        predictions=[raw(graph.to_dict(),constants,X) for X,Y in (train,validation)]
        numeric=all(a['status']=='enclosed' for a in audits) and all(np.isfinite(p).all() for p in predictions)
        output_bounds=model.config_.output_bounds
        if output_bounds is not None:
            lo,hi=output_bounds;stretch=np.asarray(coordinates.y_scale);offset=np.asarray(coordinates.y_offset)
            numeric=numeric and all((lo is None or np.all(p>=lo*stretch+offset)) and (hi is None or np.all(p<=hi*stretch+offset)) for p in predictions)
        require(numeric==row['numerically_valid'],'Physical numerical gate differs.')
        if not numeric:continue
        audit.units(graph,model.feature_units_,model.target_units_,catalog)
        shape=audit.complexity(graph);require(shape==row['physical_structure'],'Physical complexity differs.')
        errors=[]
        for label,(X,Y),prediction in zip(('train','validation'),(train,validation),predictions):
            values=error(prediction,Y,scale);errors.append(np.asarray(values))
            np.testing.assert_allclose(values,row['physical_accuracy'][label+'_nrmse'],rtol=1e-12,atol=1e-14)
            source_prediction=raw(candidate['system'],candidate['constants'],coordinates.inputs(X))
            np.testing.assert_array_equal(prediction,coordinates.restore_outputs(source_prediction))
        amount=sum(float(np.maximum(e-(r*(1+model.config_.degradation_relative)+model.config_.degradation_absolute),0).sum())
            for e,r in zip(errors,(model.reference_train_,model.reference_validation_)))
        amount+=float(np.maximum(errors[1]-report['validation_bounds'],0).sum())
        absolute=all(np.all(e<=limits['nrmse']) for e in errors);simple=all(shape[k]<=limits[k] for k in STRUCTURE)
        accepted=absolute and simple and amount==0
        require(bool(accepted)==row['accepted'] and float(amount)==row['relative_violation'],'Physical acceptance differs.')
        key=tuple(shape[k] for k in STRUCTURE)+(float(errors[1].mean()),i)
        valid.append(((amount,float(errors[1].mean()),tuple(shape[k] for k in STRUCTURE),i),i))
        if accepted:feasible.append((key,i))
    index=min(feasible)[1] if feasible else min(valid)[1] if valid else None
    require(index==report['candidate_index'] and len(feasible)==report['feasible_count'] and len(valid)==report['valid_count'],'Physical selection replay differs.')
    require(report['status']==('accepted' if feasible else 'not_met'),'Selection status differs.')
    require(index is not None,'No valid physical model.')
    selected=PhysicalModel.load(folder/'selected.json');control=PhysicalModel.load(folder/'control.json')
    require(selected.source['physical_pool_selection']['report_sha256']==digest(report),'Report digest differs.')
    require(selected.source['system']==model.candidates_[index]['system'] and selected.source['constants']==model.candidates_[index]['constants'],'Selected source was changed.')
    require(control.source['system']==model.best_.system.to_dict() and control.source['constants']==model.best_.constants.tolist(),'Control source was changed.')
    for physical in (control,selected):
        record=physical.to_dict()
        for X,Y in (train,validation,data.arrays(job,'test')):
            np.testing.assert_array_equal(physical.predict(X),raw(record['system'],record['constants'],X))
    require(score_job(job)==score,'Score replay differs.')
    return dict(job=job,verified=True,candidate_count=len(records),physical_valid=len(valid),physical_feasible=len(feasible),
        selected_index=index,legacy_data_check=not report['source_data_check']['byte_identity_verified'],
        source_model_unchanged=True,searches=0,coefficients_refitted=False)


def evaluate():
    if (HERE/'SCORES.json').exists():raise RuntimeError('Keep existing scores.')
    frozen,sealed=guards();scores=[];verified=[]
    for job in frozen['jobs']:
        require(time.time()<study.budget(),'Scoring budget expired.')
        try:row=score_job(job)
        except Exception as error:row=dict(job=job,status='scoring_failed',error=repr(error))
        scores.append(row)
    write(HERE/'SCORES.json',dict(scored_unix=time.time(),all_models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json'),rows=scores))
    for job,score in zip(frozen['jobs'],scores):
        require(time.time()<study.budget(),'Verification budget expired.')
        try:
            require(score['status']=='complete','Incomplete cell: '+score['status'])
            row=verify_job(job,frozen,score)
        except Exception as error:row=dict(job=job,verified=False,error=repr(error))
        verified.append(row);print(data.name(job),'verified' if row['verified'] else row['error'],flush=True)
    complete=[r for r in scores if r['status']=='complete'];technical=len(complete)==24 and all(r['verified'] for r in verified)
    protected=sum(sum(r['protected']) for r in complete)
    both=[r for r in complete if r['control']['both_accepted'] and r['selected']['both_accepted']]
    counts={n:sum(r[n]['both_accepted'] for r in complete) for n in ('control','selected')}
    summary=dict(expected_systems=24,complete=len(complete),verified=sum(r['verified'] for r in verified),
        candidate_count=sum(r.get('candidate_count',0) for r in verified),both_accepted=counts,
        protected_outputs=protected,expected_outputs=54,both_accepted_structure_nonincreasing=sum(r['structure_nonincreasing'] for r in both),
        both_accepted_pairs=len(both),technical_pass=technical,
        adoption_pass=technical and protected==54 and counts['selected']>=counts['control'] and all(r['structure_nonincreasing'] for r in both),
        searches=0,coefficients_refitted=False,production_default_changed=False,
        mean_comparison_nrmse={n:float(np.mean([np.mean(r[n]['comparison_nrmse']) for r in complete])) if len(complete)==24 else None for n in ('control','selected')},
        claim_scope='Historical frozen-pool physical reselection, not new discovery or blind validation.')
    write(HERE/'VERIFICATION.json',dict(status='passed' if technical else 'failed',rows=verified,
        expected=24,verified=summary['verified'],elapsed_seconds=time.time()-frozen['budget']['start_unix'],
        freeze_sha256=sha(HERE/'FREEZE.json'),models_sha256=sha(HERE/'ALL_MODELS_FROZEN.json'),scores_sha256=sha(HERE/'SCORES.json')))
    write(HERE/'SUMMARY.json',summary);print(summary,flush=True)


if __name__=='__main__':evaluate()
