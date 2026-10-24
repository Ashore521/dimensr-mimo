"""One sealed external evaluation after the complete model grid is frozen."""
from datetime import datetime,timezone
import json
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from dimensr_mimo import MIMOSymbolicRegressor
from common import HERE,SEALED,DATASETS,METHODS,SYMBOLIC,sha,save,jobs,name,check_freeze,v3,baseline_predict

def verify_models():
    check_freeze()
    frozen=json.loads((HERE/'MODELS_FROZEN.json').read_text())
    if frozen['freeze_sha256']!=sha(HERE/'FREEZE.json'):raise ValueError('Wrong training freeze')
    if set(frozen['records'])!={name(j) for j in jobs()}:raise ValueError('Incomplete model grid')
    records={}
    for job in jobs():
        key=name(job);out=HERE/'runs'/key;receipt=frozen['records'][key]
        if sha(out/'result.json')!=receipt['result_sha256']:raise ValueError('Training result changed')
        r=json.loads((out/'result.json').read_text())
        if (r['dataset'],r['seed'])!=job or r['status']!='ok':raise ValueError('Wrong training identity/status')
        if sha(out/'CHOICES.json')!=receipt['choices_sha256']:raise ValueError('Choices changed')
        if r['choices']!=json.loads((out/'CHOICES.json').read_text()):raise ValueError('Inconsistent choices')
        if set(r['choices'])!=set(METHODS):raise ValueError('Incomplete methods')
        for label,choice in r['choices'].items():
            if sha(out/choice['model'])!=receipt['models'][label] or receipt['models'][label]!=choice['model_sha256']:
                raise ValueError('Model changed')
        records[key]=r
    return records

def group_errors(errors,indices,count):
    sums=np.zeros((count,errors.shape[1]));np.add.at(sums,indices,errors)
    return sums

def main():
    # This gate runs completely before the first np.load of a sealed array.
    records=verify_models();metadata=json.loads((HERE/'data/manifest.json').read_text())
    out=HERE/'evaluation';out.mkdir(exist_ok=False)
    save(out/'OPENED.json',dict(models_freeze_sha256=sha(HERE/'MODELS_FROZEN.json'),time_utc=datetime.now(timezone.utc).isoformat()))
    rows=[];data_info={};error_files={}
    with threadpool_limits(limits=1):
        for dataset in DATASETS:
            meta=metadata[dataset];path=SEALED/f'{dataset}.npz'
            if sha(path)!=meta['sealed_sha256']:raise ValueError('Sealed data changed')
            with np.load(path,allow_pickle=False) as arrays:X,Y,groups=arrays['X'],arrays['Y'],arrays['groups']
            block=groups if dataset=='video' else (groups-groups.min())//7
            unique,index=np.unique(block,return_inverse=True);counts=np.bincount(index)
            scales=np.array(meta['target_scale']);bounds=np.array(meta['bounds'])
            expanded=np.column_stack((np.minimum(bounds[:,0],X.min(axis=0)),np.maximum(bounds[:,1],X.max(axis=0))))
            data_info[dataset]=dict(test_rows=len(X),resampling_groups=len(unique),target_names=meta['target_names'],
                outside_training_box_fraction=float(np.mean(np.any((X<bounds[:,0])|(X>bounds[:,1]),axis=1))))
            errors=dict(counts=counts)
            for seed in range(5):
                key=name((dataset,seed));record=records[key]
                for label in METHODS:
                    choice=record['choices'][label];model_path=HERE/'runs'/key/choice['model']
                    row=dict(dataset=dataset,seed=seed,method=label,status='ok',complexity=choice['complexity'],
                        training_cpu_seconds=choice['training_cpu_seconds'],validation=choice['validation'],
                        selected_transfer_heads=choice['selected_transfer_heads'],model_sha256=choice['model_sha256'])
                    try:
                        if label in SYMBOLIC:
                            model=MIMOSymbolicRegressor.load(model_path);prediction=model.predict(X)
                            audit=model.audit_domain(expanded)
                            row['expanded_domain']=dict(status=audit['status'],features=[a['status'] for a in audit['features']],outputs=[a['status'] for a in audit['outputs']])
                        else:prediction=baseline_predict(joblib.load(model_path),X)
                        row['test']=v3.metrics(Y,prediction,scales)
                        errors[f'{label}_s{seed}']=group_errors(((prediction-Y)/scales)**2,index,len(unique))
                    except (ValueError,FloatingPointError,OverflowError) as exc:row.update(status='failed',error=str(exc))
                    rows.append(row)
            error_path=out/f'{dataset}_group_errors.npz'
            with error_path.open('xb') as stream:np.savez_compressed(stream,**errors)
            error_files[error_path.name]=sha(error_path)
    save(out/'results.json',dict(models_freeze_sha256=sha(HERE/'MODELS_FROZEN.json'),datasets=data_info,
        group_error_sha256=error_files,rows=rows))
    print(f'Evaluated {len(rows)} models; failures={sum(r["status"]!="ok" for r in rows)}')

if __name__=='__main__':main()
