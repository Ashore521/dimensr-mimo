"""Fixed new output pairs; truth graphs are never supplied to model fitting."""
from pathlib import Path
import numpy as np
from dimensr_mimo._single.units import ZERO, unit, scale, encode

HERE = Path(__file__).resolve().parent
L = unit([1,0,0,0,0,0,0])
T = unit([0,0,1,0,0,0,0])
FAMILIES = ('logarithmic_gain','logistic_gain','algebraic_norm','bilinear_saturation')
CASES = tuple(f'{family}_n{noise}' for family in FAMILIES for noise in (0,1))
SEEDS = (9131,9132)


def seeds(name):
    if name not in CASES:
        raise ValueError('Outside the frozen new-task matrix.')
    return SEEDS


def design(name):
    seeds(name)
    family,noise = name.rsplit('_n',1)
    if family in ('logarithmic_gain','logistic_gain'):
        features,constants,names = (T,L),(ZERO,scale(T,-1)),['time','amplitude']
        targets = (ZERO,L) if family=='logarithmic_gain' else (L,scale(L,2))
        formulas = (['log(1+(0.9*x0)**2)','x1*log(1+(0.9*x0)**2)'] if family=='logarithmic_gain' else
                    ['x1/(1+exp(-0.8*x0))','x1**2/(1+exp(-0.8*x0))'])
    elif family=='algebraic_norm':
        features,targets,constants,names = (L,L),(L,ZERO),(ZERO,),['length0','length1']
        formulas = ['sqrt(x0**2+x1**2)','x0/sqrt(x0**2+x1**2)']
    else:
        features,targets,constants,names = (L,scale(L,-1)),(L,scale(L,-1)),(ZERO,),['length','inverse_length']
        formulas = ['x0/(1+0.6*x0*x1)','x1/(1+0.6*x0*x1)']
    return dict(features=[encode(u) for u in features],targets=[encode(u) for u in targets],
        constants=[encode(u) for u in constants],feature_names=names,target_names=['response0','response1'],
        formulas=formulas,bounds=[[.2,1.4],[.2,1.4]],family=family,noise_fraction=.01*int(noise),
        input_transform='identity in declared SI units',output_transform='identity; normalization only in error metric',
        acceptance=dict(nrmse=[.03 if int(noise) else 1e-4]*2,shared_graph_size=24,
            independent_parameters=6,expanded_total_nodes=30,maximum_output_depth=6),
        clean_test_nrmse=[.02 if int(noise) else 1e-4]*2)


def latent(name,X):
    family=design(name)['family']; a,b=np.asarray(X).T
    if family=='logarithmic_gain':
        response=np.log(1+(.9*a)**2)
        return np.column_stack((response,b*response))
    if family=='logistic_gain':
        denominator=1+np.exp(-.8*a)
        return np.column_stack((b/denominator,b*b/denominator))
    if family=='algebraic_norm':
        response=np.sqrt(a*a+b*b)
        return np.column_stack((response,a/response))
    denominator=1+.6*a*b
    return np.column_stack((a/denominator,b/denominator))


def inputs(name,split,seed):
    if seed not in seeds(name):
        raise ValueError('Outside the frozen new-task seeds.')
    code={'train':111,'validation':222,'test':333}[split]
    n={'train':96,'validation':48,'test':192}[split]
    rng=np.random.default_rng(np.random.SeedSequence([20261030,seed,code,CASES.index(name)]))
    bounds=np.asarray(design(name)['bounds'])
    return rng.uniform(bounds[:,0],bounds[:,1],(n,2))


def generate(name,split,seed):
    if split=='test' and not (HERE/'ALL_MODELS_FROZEN.json').exists():
        raise RuntimeError('Test labels require the all-model seal.')
    X=inputs(name,split,seed); spec=design(name); Y=latent(name,X)
    scales=latent(name,inputs(name,'train',seed)).std(axis=0)
    rng=np.random.default_rng(np.random.SeedSequence([20261030,seed,
        {'train':411,'validation':422,'test':433}[split],CASES.index(name)]))
    return X,Y+spec['noise_fraction']*scales*rng.normal(size=Y.shape)


def training_data(name,split,seed):
    if split not in ('train','validation'):
        raise ValueError('Training cannot request test data.')
    return generate(name,split,seed)


def truth_graph(name):
    """Static dimensional/size/domain audit; no fit or search consumes this graph."""
    from dimensr_mimo.graph import System, Vertex
    from dimensr_mimo.structure import operator
    from dimensr_mimo._single.expression import Node
    spec=design(name);nodes=[];values=[]
    def x(j):
        nodes.append(Vertex('x'+str(j),units=unit(spec['features'][j])));return len(nodes)-1
    def c(value,u):
        index=next(i for i,s in enumerate(spec['constants']) if unit(s)==u)
        parameter=len(values);values.append(value)
        nodes.append(Vertex('c'+str(index),units=u,parameter=parameter));return len(nodes)-1
    def op(token,*children):
        units=operator(token,*[Node('x0',units=nodes[i].units) for i in children]).units
        nodes.append(Vertex(token,children,units));return len(nodes)-1
    a,b=x(0),x(1)
    if spec['family']=='logarithmic_gain':
        response=op('log',op('add',c(1.,ZERO),op('square',op('mul',c(.9,scale(T,-1)),a))))
        roots=(response,op('mul',b,response))
    elif spec['family']=='logistic_gain':
        denominator=op('add',c(1.,ZERO),op('exp',op('mul',c(-.8,scale(T,-1)),a)))
        roots=(op('div',b,denominator),op('div',op('square',b),denominator))
    elif spec['family']=='algebraic_norm':
        response=op('sqrt',op('add',op('square',a),op('square',b)))
        roots=(response,op('div',a,response))
    else:
        denominator=op('add',c(1.,ZERO),op('mul',c(.6,ZERO),op('mul',a,b)))
        roots=(op('div',a,denominator),op('div',b,denominator))
    return System(tuple(nodes),roots),np.asarray(values)
