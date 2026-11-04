"""Declared dimensional development strata; no test access from fitting."""
import numpy as np
from dimensr_mimo._single.units import ZERO,unit,add,sub,scale,encode

L=unit([1,0,0,0,0,0,0]);M=unit([0,1,0,0,0,0,0]);T=unit([0,0,1,0,0,0,0])
V=sub(L,T);K=scale(L,-1);A=sub(L,scale(T,2))
CASES=('products','dimensional_frequency','partial','private_scales','high_dimensional','boundary')
SEEDS=(601,602)


def design(name):
    if name=='products':
        f=(M,V);y=(add(M,V),add(M,scale(V,2)));c=(ZERO,);bounds=[[.3,1.4],[.2,1.3]]
        names=['mass','velocity'];targets=['momentum','kinetic_energy']
    elif name=='dimensional_frequency':
        f=(L,L);y=(L,V);c=(ZERO,K,L,V);bounds=[[-.8,.8]]*2
        names=['distance0','distance1'];targets=['displacement','speed_response']
    elif name=='partial':
        f=(T,L);y=(L,L);c=(ZERO,A,L,K);bounds=[[.2,1.1],[-.8,.8]]
        names=['time','distance'];targets=['response0','response1']
    elif name=='private_scales':
        f=(L,T);y=(scale(L,2),scale(T,2));c=(ZERO,*y);bounds=[[.2,1.1],[.3,1.3]]
        names=['length','time'];targets=['small_area','large_time_squared']
    elif name=='high_dimensional':
        f=(L,)*11;y=(ZERO,ZERO);c=(ZERO,K);bounds=[[-.8,.8]]*11
        names=[f'distance{j}' for j in range(11)];targets=['wave0','wave1']
    elif name=='boundary':
        f=(L,L);y=(ZERO,L);c=(ZERO,K,L);bounds=[[.2,1.2],[.3,1.3]]
        names=['length0','length1'];targets=['log_response','length_response']
    else:raise ValueError('Unknown dimensional stratum.')
    return dict(features=[encode(u) for u in f],targets=[encode(u) for u in y],constants=[encode(u) for u in c],
                bounds=bounds,feature_names=names,target_names=targets,
                input_transform='identity in declared SI units',output_transform='identity; normalization only in error metric',
                acceptance=dict(nrmse=[1e-4,1e-4],shared_graph_size=24,independent_parameters=6,expanded_total_nodes=30,maximum_output_depth=6))


def generate(name,split,seed):
    if name not in CASES or seed not in SEEDS:raise ValueError('Outside the fixed dimensional matrix.')
    code={'train':111,'validation':222,'test':333}[split];n={'train':96,'validation':48,'test':192}[split]
    spec=design(name);bounds=np.asarray(spec['bounds']);rng=np.random.default_rng(np.random.SeedSequence([20261026,seed,code,CASES.index(name)]))
    X=rng.uniform(bounds[:,0],bounds[:,1],(n,len(bounds)));a,b=X[:,:2].T
    if name=='products':Y=np.column_stack((a*b,.5*a*b*b))
    elif name=='dimensional_frequency':Y=np.column_stack((.8*np.sin(1.7*a),.6*np.cos(1.7*b)))
    elif name=='partial':Y=np.column_stack((.7*a*a+b,.7*a*a+.5*np.sin(1.3*b)))
    elif name=='private_scales':Y=np.column_stack((.001*a*a,1000*b*b))
    elif name=='high_dimensional':Y=np.column_stack((np.sin(1.7*X[:,9]),np.cos(1.7*X[:,10])))
    else:Y=np.column_stack((np.log(a+b),np.sqrt(a+b)))
    return X,Y


def training_data(name,split,seed):
    if split not in ('train','validation'):raise ValueError('Training cannot request test data.')
    return generate(name,split,seed)
