"""New declared development families; held-out labels gated by the supervisor."""
import numpy as np
from dimensr_mimo._single.units import ZERO,unit,sub,add,scale,encode

L=unit([1,0,0,0,0,0,0]);M=unit([0,1,0,0,0,0,0]);T=unit([0,0,1,0,0,0,0])
V=sub(L,T);KT=scale(T,-1);KV2=scale(V,-2)
FAMILIES=('damped','rational','polynomial')
CASES=tuple(f'{family}_n{noise}' for family in FAMILIES for noise in (0,1))
SEEDS=(701,702)


def design(name):
    if name not in CASES:raise ValueError('Outside the frozen matrix.')
    family,level=name.split('_n');noise=.01*int(level)
    if family=='damped':
        features=(T,L);targets=(L,L);constants=(ZERO,KT);bounds=[[.1,1.6],[.5,1.5]]
        names=['time','amplitude'];outputs=['cosine_response','sine_response']
        formulas=['x1*exp(-0.4*x0)*cos(1.1*x0)','x1*exp(-0.4*x0)*sin(1.1*x0)']
    elif family=='rational':
        features=(T,L);targets=(L,scale(L,2));constants=(ZERO,KT);bounds=[[.1,1.6],[.5,1.5]]
        names=['time','length'];outputs=['length_response','area_response']
        formulas=['x1/(1+(0.7*x0)**2)','x1**2/(1+(0.7*x0)**2)']
    else:
        features=(M,V);targets=(add(M,V),add(M,scale(V,2)));constants=(ZERO,KV2);bounds=[[.3,1.4],[.2,1.3]]
        names=['mass','velocity'];outputs=['momentum_response','energy_response']
        formulas=['x0*x1*(1+0.3*x1**2)','0.5*x0*x1**2*(1+0.3*x1**2)']
    limits=dict(nrmse=[.03 if noise else 1e-4]*2,shared_graph_size=24,
        independent_parameters=6,expanded_total_nodes=30,maximum_output_depth=6)
    return dict(family=family,noise_fraction=noise,features=[encode(u) for u in features],targets=[encode(u) for u in targets],
        constants=[encode(u) for u in constants],bounds=bounds,feature_names=names,target_names=outputs,formulas=formulas,
        input_transform='identity in declared SI units',output_transform='identity; noise and error scale from training only',
        acceptance=limits,clean_test_nrmse=[.02 if noise else 1e-4]*2)


def inputs(name,split,seed):
    if seed not in SEEDS:raise ValueError('Outside the frozen seeds.')
    spec=design(name);code={'train':111,'validation':222,'test':333}[split]
    n={'train':96,'validation':48,'test':192}[split]
    rng=np.random.default_rng(np.random.SeedSequence([20261027,seed,code,FAMILIES.index(spec['family'])]))
    b=np.asarray(spec['bounds']);return rng.uniform(b[:,0],b[:,1],(n,2))


def latent(name,X):
    family=design(name)['family'];a,b=X.T
    if family=='damped':
        envelope=b*np.exp(-.4*a);return np.column_stack((envelope*np.cos(1.1*a),envelope*np.sin(1.1*a)))
    if family=='rational':return np.column_stack((b/(1+(.7*a)**2),b*b/(1+(.7*a)**2)))
    common=1+.3*b*b;return np.column_stack((a*b*common,.5*a*b*b*common))


def generate(name,split,seed):
    X=inputs(name,split,seed);Y=latent(name,X);spec=design(name)
    clean_train=latent(name,inputs(name,'train',seed));scales=clean_train.std(axis=0)
    rng=np.random.default_rng(np.random.SeedSequence([20261027,seed,{'train':411,'validation':422,'test':433}[split],FAMILIES.index(spec['family'])]))
    return X,Y+spec['noise_fraction']*scales*rng.normal(size=Y.shape)


def training_data(name,split,seed):
    if split not in ('train','validation'):raise ValueError('Fitting cannot request test data.')
    return generate(name,split,seed)


def truth_graph(name):
    """Grammar preflight only; never supplied to initialization or model fitting."""
    from dimensr_mimo.graph import System,Vertex
    from dimensr_mimo.structure import operator
    from dimensr_mimo._single.expression import Node
    spec=design(name);nodes=[];values=[]
    def x(j):
        nodes.append(Vertex('x'+str(j),units=unit(spec['features'][j])));return len(nodes)-1
    def c(value,u):
        p=len(values);values.append(value);i=next(i for i,s in enumerate(spec['constants']) if unit(s)==u)
        nodes.append(Vertex('c'+str(i),units=u,parameter=p));return len(nodes)-1
    def op(token,*children):
        u=operator(token,*[Node('x0',units=nodes[i].units) for i in children]).units
        nodes.append(Vertex(token,children,u));return len(nodes)-1
    a,b=x(0),x(1)
    if spec['family']=='damped':
        envelope=op('mul',b,op('exp',op('mul',c(-.4,KT),a)))
        phase=op('mul',c(1.1,KT),a)
        roots=(op('mul',envelope,op('cos',phase)),op('mul',envelope,op('sin',phase)))
    elif spec['family']=='rational':
        denominator=op('add',c(1.,ZERO),op('square',op('mul',c(.7,KT),a)))
        roots=(op('div',b,denominator),op('div',op('square',b),denominator))
    else:
        common=op('add',c(1.,ZERO),op('mul',c(.3,KV2),op('square',b)))
        momentum=op('mul',op('mul',a,b),common)
        roots=(momentum,op('mul',op('mul',c(.5,ZERO),op('mul',a,op('square',b))),common))
    return System(tuple(nodes),roots),np.asarray(values)
