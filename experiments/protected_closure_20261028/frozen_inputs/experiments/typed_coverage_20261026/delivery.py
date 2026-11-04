"""Physical-unit delivery and predeclared feasibility; no test-set selection."""
import numpy as np
from dimensr_mimo.graph import Builder,System
from dimensr_mimo._single.units import unit,encode
from experiments.typed_coverage_20261026 import audit


def accepted(errors,structure,limits):
    return bool(np.isfinite(errors).all() and len(errors)==len(limits['nrmse']) and
        all(e<=v for e,v in zip(errors,limits['nrmse'])) and
        all(structure[k]<=limits[k] for k in limits if k!='nrmse'))


def graph(models,single):
    if not single:return models[0].best_.system,models[0].best_.constants
    b=Builder()
    for m in models:b.roots.append(b.tree(m.best_.tree,m.best_.constants))
    return b.finish()


def record(system,constants,spec,X,Y,V,Z):
    unit_report=audit.units(system,spec['features'],spec['targets'],spec['constants'])
    structure=audit.complexity(system);scales=Y.std(axis=0)
    train=np.sqrt(np.mean((system.evaluate(X,constants)-Y)**2,axis=0))/scales
    validation=np.sqrt(np.mean((system.evaluate(V,constants)-Z)**2,axis=0))/scales
    return dict(system=system.to_dict(),parameters=[dict(index=i,value=float(c),units=u) for i,(c,u) in enumerate(zip(constants,unit_report['parameter_units']))],
        units=unit_report,structure=structure,train_nrmse=train.tolist(),validation_nrmse=validation.tolist(),
        development_accepted=accepted(train,structure,spec['acceptance']) and accepted(validation,structure,spec['acceptance']),
        physical_variables=[dict(symbol=f'x{i}',name=n,units=encode(unit(u))) for i,(n,u) in enumerate(zip(spec['feature_names'],spec['features']))],
        physical_outputs=[dict(name=n,units=encode(unit(u))) for n,u in zip(spec['target_names'],spec['targets'])],
        input_bounds=spec['bounds'],input_transform=spec['input_transform'],output_transform=spec['output_transform'],
        prediction_semantics='Original unsimplified graph; SI values, no input/output rescaling.',acceptance=spec['acceptance'])


def pool(source,spec):
    rows=[]
    for i,c in enumerate(source['candidates']):
        s=System.from_dict(c['system']);structure=audit.complexity(s)
        audit.units(s,spec['features'],spec['targets'],spec['constants'])
        rows.append(dict(index=i,train_nrmse=c['errors'],validation_nrmse=c['validation_errors'],structure=structure,
            feasible=accepted(c['errors'],structure,spec['acceptance']) and accepted(c['validation_errors'],structure,spec['acceptance'])))
    return dict(candidates=len(rows),feasible_count=sum(r['feasible'] for r in rows),rows=rows,
        selection_changed=False,test_data_used=False)
