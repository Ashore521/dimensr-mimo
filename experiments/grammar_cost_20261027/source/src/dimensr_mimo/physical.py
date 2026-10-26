"""Explicit affine coordinates and a raw formula graph in physical variables.

Scale factors are dimensionless numeric stretches; coordinate quantities retain
their declared dimensions. Offsets carry the corresponding input/output units.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from .graph import System,Vertex
from ._single.units import ZERO,unit,encode
from . import audit
from .acceptance import validate_limits,STRUCTURE
from .pareto import violations


@dataclass(frozen=True)
class AffineCoordinates:
    x_offset: tuple
    x_scale: tuple
    y_offset: tuple
    y_scale: tuple

    def __post_init__(self):
        for key in ('x_offset','x_scale','y_offset','y_scale'):
            a=np.asarray(getattr(self,key),float)
            if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Coordinates require finite nonempty vectors.')
            if key.endswith('scale') and np.any(a<=0):raise ValueError('Coordinate scales must be positive.')
            object.__setattr__(self,key,tuple(a.tolist()))
        if len(self.x_offset)!=len(self.x_scale) or len(self.y_offset)!=len(self.y_scale):raise ValueError('Offset and scale dimensions differ.')

    @classmethod
    def identity(cls,p,q):return cls((0.,)*p,(1.,)*p,(0.,)*q,(1.,)*q)

    @classmethod
    def from_training(cls,X,Y):
        X,Y=np.asarray(X,float),np.asarray(Y,float)
        if X.ndim!=2 or Y.ndim!=2 or len(X)!=len(Y) or len(X)<2 or not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError('Training coordinates require finite paired 2D arrays.')
        sx,sy=X.std(axis=0),Y.std(axis=0)
        return cls(X.mean(axis=0),np.where(sx>0,sx,1.),Y.mean(axis=0),np.where(sy>0,sy,1.))

    def inputs(self,X):
        X=np.asarray(X,float)
        if X.ndim!=2 or X.shape[1]!=len(self.x_offset) or not np.isfinite(X).all():raise ValueError('Invalid physical input array.')
        return (X-np.asarray(self.x_offset))/np.asarray(self.x_scale)

    def outputs(self,Y):
        Y=np.asarray(Y,float)
        if Y.ndim!=2 or Y.shape[1]!=len(self.y_offset) or not np.isfinite(Y).all():raise ValueError('Invalid physical output array.')
        return (Y-np.asarray(self.y_offset))/np.asarray(self.y_scale)

    def restore_outputs(self,Y):return np.asarray(Y,float)*np.asarray(self.y_scale)+np.asarray(self.y_offset)

    def input_bounds(self,bounds):
        bounds=np.asarray(bounds,float)
        if bounds.shape!=(len(self.x_offset),2) or not np.isfinite(bounds).all() or np.any(bounds[:,0]>=bounds[:,1]):
            raise ValueError('Physical bounds require one finite increasing interval per input.')
        return ((bounds-np.asarray(self.x_offset)[:,None])/np.asarray(self.x_scale)[:,None]).tolist()

    def to_dict(self):return {k:list(getattr(self,k)) for k in ('x_offset','x_scale','y_offset','y_scale')}


def restore_graph(system,constants,features,targets,catalog,coordinates):
    """Substitute affine coordinates without algebraic simplification or refitting."""
    features,targets,catalog=[list(map(unit,seq)) for seq in (features,targets,catalog)]
    audit.units(system,features,targets,catalog)
    constants=np.asarray(constants,float)
    if constants.shape!=(system.n_parameters,) or not np.isfinite(constants).all():raise ValueError('Invalid source parameters.')
    if len(features)!=len(coordinates.x_offset) or len(targets)!=len(coordinates.y_offset):raise ValueError('Coordinate and model dimensions differ.')
    nodes=[];values=[];roles=[];mapped={};parameters={};inputs={}
    def terminal(value,units,role,**metadata):
        if units not in catalog:catalog.append(units)
        p=len(values);values.append(float(value));roles.append(dict(role=role,**metadata))
        nodes.append(Vertex('c'+str(catalog.index(units)),units=units,parameter=p));return len(nodes)-1
    def operation(token,children,units):
        nodes.append(Vertex(token,tuple(children),units));return len(nodes)-1
    for i,v in enumerate(system.nodes):
        if v.parameter is not None:
            if v.parameter not in parameters:
                mapped[i]=terminal(constants[v.parameter],v.units,'fitted',source_parameter=v.parameter)
                parameters[v.parameter]=nodes[mapped[i]].parameter
            else:
                nodes.append(Vertex('c'+str(catalog.index(v.units)),units=v.units,parameter=parameters[v.parameter]));mapped[i]=len(nodes)-1
        elif not v.children:
            j=int(v.token[1:]);shift,stretch=coordinates.x_offset[j],coordinates.x_scale[j]
            if (shift!=0 or stretch!=1) and j in inputs:mapped[i]=inputs[j];continue
            nodes.append(Vertex(v.token,units=v.units));value=len(nodes)-1
            if shift!=0:value=operation('sub',(value,terminal(shift,v.units,'fixed_input_offset',feature=j)),v.units)
            if stretch!=1:value=operation('div',(value,terminal(stretch,ZERO,'fixed_input_scale',feature=j)),v.units)
            mapped[i]=value;inputs[j]=value
        else:mapped[i]=operation(v.token,[mapped[k] for k in v.children],v.units)
    roots=[]
    for j,root in enumerate(system.roots):
        value=mapped[root];shift,stretch=coordinates.y_offset[j],coordinates.y_scale[j]
        if stretch!=1:value=operation('mul',(value,terminal(stretch,ZERO,'fixed_output_scale',output=j)),targets[j])
        if shift!=0:value=operation('add',(value,terminal(shift,targets[j],'fixed_output_offset',output=j)),targets[j])
        roots.append(value)
    graph=System(tuple(nodes),tuple(roots))
    return graph,np.asarray(values),[encode(u) for u in catalog],roles


class PhysicalModel:
    """Prediction and audited delivery in original physical coordinates."""
    def __init__(self,source,coordinates,input_bounds,physical_limits=None):
        self.source=deepcopy(source);self.coordinates=coordinates
        self.input_bounds=np.asarray(input_bounds,float).tolist()
        coordinates.input_bounds(self.input_bounds)
        graph=System.from_dict(source['system'])
        if len(source['feature_names'])!=len(source['feature_units']) or len(set(source['feature_names']))!=len(source['feature_names']):
            raise ValueError('Physical feature names do not match inputs.')
        self.system,self.constants,self.constant_units,self.parameter_roles=restore_graph(graph,source['constants'],source['feature_units'],source['target_units'],source['constant_units'],coordinates)
        self.units=audit.units(self.system,source['feature_units'],source['target_units'],self.constant_units)
        self.structure=audit.complexity(self.system)
        self.source_structure=audit.complexity(graph)
        self.domain_audits=self.system.audit(self.constants,self.input_bounds,max_boxes=128)
        if any(a['status']!='enclosed' for a in self.domain_audits):raise ValueError('Physical domain is invalid or unresolved.')
        self.physical_limits=deepcopy(physical_limits)
        self.acceptance=dict(status='not_assessed',reason='No frozen physical-coordinate limits supplied.')
        if physical_limits is not None:
            validate_limits(physical_limits,len(self.system.roots))
            identity=coordinates==AffineCoordinates.identity(len(source['feature_units']),len(source['target_units']))
            verified=source.get('physical_accuracy')
            train,validation=(verified['train_nrmse'],verified['validation_nrmse']) if verified else (source['training_nrmse'],source['validation_nrmse'])
            if len(train)!=len(self.system.roots) or len(validation)!=len(train) or not np.isfinite(train+validation).all():
                raise ValueError('Invalid source accuracy metrics.')
            accuracy=all(0<=a<=b for row in (train,validation) for a,b in zip(row,physical_limits['nrmse']))
            structure=all(self.structure[k]<=physical_limits[k] for k in STRUCTURE)
            protected=source['relative_constraints_ok'] is True
            known=bool(verified) or identity
            status='not_met' if not structure or not protected or (known and not accuracy) else 'accepted' if known else 'not_assessed'
            self.acceptance=dict(status=status,
                accuracy_pass=accuracy if known else None,structure_pass=structure,relative_constraints_pass=protected,
                limits=deepcopy(physical_limits),metric='NRMSE using the stored physical training normalization',
                metric_source='physical_arrays_recomputed' if verified else 'identical_coordinates' if identity else 'physical_arrays_required',
                scope='Training and validation acceptance; not a test or engineering-use guarantee.')

    @classmethod
    def from_estimator(cls,model,coordinates=None,input_bounds=None,physical_limits=None,training_data=None,validation_data=None):
        if not model.config_.use_units:raise ValueError('Physical export requires unit constraints.')
        coordinates=coordinates or AffineCoordinates.identity(model.n_features_in_,model.n_outputs_)
        if input_bounds is None:
            b=np.asarray(model.diagnostics_['domain']['input_bounds'])
            input_bounds=(b*np.asarray(coordinates.x_scale)[:,None]+np.asarray(coordinates.x_offset)[:,None]).tolist()
        protected=model.config_.use_constraints and violations(model.best_.errors,model.reference_train_,model.config_)==0 and violations(model.validation_nrmse_,model.reference_validation_,model.config_)==0
        post=getattr(model,'post_selection_',{})
        if 'validation_bounds' in post:protected=protected and all(a<=b for a,b in zip(model.validation_nrmse_,post['validation_bounds']))
        source=dict(system=model.best_.system.to_dict(),constants=model.best_.constants.tolist(),
            feature_units=[encode(u) for u in model.feature_units_],target_units=[encode(u) for u in model.target_units_],
            constant_units=[encode(u) for u in model.constant_units_],feature_names=list(model.feature_names_),
            training_nrmse=model.best_.errors.tolist(),validation_nrmse=model.validation_nrmse_.tolist(),
            nrmse_scale_physical=(model.scales_*np.asarray(coordinates.y_scale)).tolist(),relative_constraints_ok=bool(protected),
            output_bounds=model.config_.output_bounds,search_fingerprint=model.search_fingerprint_,
            search_coordinate_acceptance=deepcopy(model.diagnostics_.get('application_acceptance')))
        if (training_data is None)!=(validation_data is None):raise ValueError('Provide physical training and validation arrays together.')
        if training_data is not None:
            physical=cls(source,coordinates,input_bounds)
            accuracy={};scales=np.asarray(source['nrmse_scale_physical'])
            if not np.isfinite(scales).all() or np.any(scales<=0):raise ValueError('Invalid physical error normalization.')
            for name,data in (('train',training_data),('validation',validation_data)):
                X,Y=map(lambda a:np.asarray(a,float),data);prediction=physical.predict(X)
                if Y.shape!=prediction.shape or not len(X) or not np.isfinite(Y).all():raise ValueError('Invalid paired physical evaluation arrays.')
                accuracy[name+'_nrmse']=(np.sqrt(np.mean((prediction-Y)**2,axis=0))/scales).tolist()
                accuracy[name+'_rows']=len(X)
                accuracy[name+'_sha256']=hashlib.sha256(X.astype('<f8').tobytes()+Y.astype('<f8').tobytes()).hexdigest()
            source['physical_accuracy']=accuracy
        return cls(source,coordinates,input_bounds,physical_limits)

    def predict(self,X):
        self.coordinates.inputs(X)
        X=np.asarray(X,float);bounds=np.asarray(self.input_bounds)
        if np.any(X<bounds[:,0]) or np.any(X>bounds[:,1]):raise ValueError('Inputs lie outside the declared physical domain.')
        pred=self.system.evaluate(X,self.constants)
        if not np.isfinite(pred).all():raise ValueError('Raw physical formula is undefined.')
        limits=self.source.get('output_bounds')
        if limits is not None:
            lo,hi=limits;scale=np.asarray(self.coordinates.y_scale);offset=np.asarray(self.coordinates.y_offset)
            if (lo is not None and np.any(pred<lo*scale+offset)) or (hi is not None and np.any(pred>hi*scale+offset)):raise ValueError('Output bounds violated.')
        return pred

    @property
    def formulas(self):return self.system.formulas(self.constants,self.source['feature_names'])

    def to_dict(self):
        reach=[self.system.reachable(r) for r in self.system.roots]
        shared=[dict(vertex=i,token=v.token,children=list(v.children),outputs=[j for j,r in enumerate(reach) if i in r])
            for i,v in enumerate(self.system.nodes) if v.children and sum(i in r for r in reach)>1]
        parameters=[dict(index=i,value=float(value),units=self.units['parameter_units'][i],**role,
            outputs=[j for j,r in enumerate(reach) if any(self.system.nodes[k].parameter==i for k in r)])
            for i,(value,role) in enumerate(zip(self.constants,self.parameter_roles))]
        return dict(format='dimensr-mimo-physical',schema_version=1,source=deepcopy(self.source),coordinates=self.coordinates.to_dict(),
            input_bounds=deepcopy(self.input_bounds),physical_limits=deepcopy(self.physical_limits),system=self.system.to_dict(),
            constants=self.constants.tolist(),constant_units=self.constant_units,parameters=parameters,formulas=self.formulas,
            shared_expressions=shared,source_structure=self.source_structure,physical_structure=self.structure,
            fitted_parameters=sum(r['role']=='fitted' for r in self.parameter_roles),fixed_coordinate_parameters=sum(r['role']!='fitted' for r in self.parameter_roles),
            source_sharing=System.from_dict(self.source['system']).sharing,physical_sharing=self.system.sharing,
            physical_inputs=[dict(name=n,units=u) for n,u in zip(self.source['feature_names'],self.source['feature_units'])],
            physical_outputs=[dict(name=f'y{j}',units=u,root=self.system.roots[j]) for j,u in enumerate(self.source['target_units'])],
            physical_acceptance=self.acceptance,domain_audits=self.domain_audits,
            guarantee='Raw graph semantics; finite numerical domain screening, not a continuous-domain proof.')

    def save(self,path):Path(path).write_text(json.dumps(self.to_dict(),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text(encoding='utf-8'))
        if d.get('format')!='dimensr-mimo-physical' or d.get('schema_version')!=1:raise ValueError('Unsupported physical model schema.')
        model=cls(d['source'],AffineCoordinates(**d['coordinates']),d['input_bounds'],d['physical_limits']);expected=model.to_dict()
        for key in ('system','constants','constant_units','parameters','physical_structure','source_structure','physical_acceptance','shared_expressions'):
            if expected[key]!=d[key]:raise ValueError('Physical model metadata or graph mismatch: '+key)
        return model
