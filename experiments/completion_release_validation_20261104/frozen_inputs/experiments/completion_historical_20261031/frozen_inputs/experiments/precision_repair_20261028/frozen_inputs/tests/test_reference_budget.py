from dataclasses import replace
from fractions import Fraction
import numpy as np
import pytest
from dimensr_mimo import SearchConfig
from dimensr_mimo.grammar import Grammar
from dimensr_mimo.generation import SystemGrammar
from dimensr_mimo.graph import System,Vertex,in_bounds
from dimensr_mimo.policy import Policy
from dimensr_mimo.syntax import Node
from dimensr_mimo._single.units import ZERO,unit


def test_expanded_reference_masks_account_for_size_depth_and_siblings():
    g=Grammar([ZERO],ZERO,[ZERO],('add','square'),max_nodes=5,max_depth=3,
        reference_units=(ZERO,),reference_shapes=((3,2),))
    ref=g.index['r0']
    assert ref not in g.options(ZERO,2,3) and ref not in g.options(ZERO,5,1)
    assert ref in g.options(ZERO,3,2)
    g.trace(Node('add',(Node('r0'),Node('x0'))))
    with pytest.raises(ValueError):g.trace(Node('add',(Node('r0'),Node('r0'))))
    with pytest.raises(ValueError):g.trace(Node('square',(Node('square',(Node('r0'),)),)))


def test_system_grammar_cache_distinguishes_equal_unit_references_by_shape():
    cfg=SearchConfig(reference_budget='expanded',max_nodes=5,max_depth=3)
    g=SystemGrammar((ZERO,),(ZERO,ZERO),(ZERO,),cfg)
    small=(Vertex('x0'),)
    large=(Vertex('x0'),Vertex('square',(0,)),Vertex('square',(1,)))
    a=g.grammar(1,small,(0,));b=g.grammar(1,large,(2,))
    assert a is not b and a.terminal_shapes['r0']==(1,1) and b.terminal_shapes['r0']==(3,3)
    assert a.index['r0'] in a.options(ZERO,2,2)
    assert b.index['r0'] not in b.options(ZERO,2,2)


@pytest.mark.parametrize('outputs,seed',[(2,31),(3,32),(4,33)])
def test_all_generated_shared_systems_obey_expanded_limits(outputs,seed):
    L=unit([1,0,0,0,0,0,0]);H=tuple(x*Fraction(1,2) for x in L)
    cfg=SearchConfig(reference_budget='expanded',max_nodes=9,max_depth=5,max_references=20)
    targets=(L,H,ZERO,L)[:outputs]
    g=SystemGrammar((L,H),targets,(ZERO,L,H),cfg)
    policy=Policy(g.n_tokens,seed=seed,context_size=g.context_size);rng=np.random.default_rng(seed)
    systems=[g.generate(rng,policy) for _ in range(32)]
    assert all(in_bounds(s,cfg) for s,_,_ in systems)
    assert any(s.sharing['shared_nodes']>0 for s,_,_ in systems)
    lp,entropy=policy.log_probs([tr for _,_,tr in systems])
    assert np.isfinite(lp.detach().numpy()).all() and np.isfinite(entropy.detach().numpy()).all()
    update=policy.update([tr for _,_,tr in systems],list(np.linspace(0,1,len(systems))))
    assert update['gradient_norm']>0


def test_expanded_mode_without_references_preserves_proposals():
    cfg=SearchConfig(max_nodes=7,max_depth=4,use_sharing=False)
    a=SystemGrammar((ZERO,),(ZERO,ZERO),(ZERO,),cfg)
    b=SystemGrammar((ZERO,),(ZERO,ZERO),(ZERO,),replace(cfg,reference_budget='expanded'))
    ra=np.random.default_rng(19);rb=np.random.default_rng(19)
    for _ in range(30):
        x,xc,xt=a.generate(ra);y,yc,yt=b.generate(rb)
        assert x==y and xt.tokens==yt.tokens
        np.testing.assert_array_equal(xc,yc)


def test_invalid_reference_budget_rejected():
    with pytest.raises(ValueError):SearchConfig(reference_budget='unknown').validate()
    with pytest.raises(ValueError):Grammar([ZERO],ZERO,[ZERO],('add',),reference_units=(ZERO,),reference_shapes=((0,1),))


def test_shared_parameter_graph_replays_at_exact_expanded_boundary():
    cfg=SearchConfig(reference_budget='expanded',max_nodes=4,max_depth=3)
    g=SystemGrammar((ZERO,),(ZERO,ZERO),(ZERO,),cfg)
    graph=System((Vertex('x0'),Vertex('c0',parameter=0),Vertex('mul',(0,1)),
        Vertex('sin',(2,)),Vertex('cos',(2,))),(3,4))
    assert in_bounds(graph,cfg) and graph.n_parameters==1
    trace=g.trace(graph);policy=Policy(g.n_tokens,seed=9,context_size=g.context_size)
    lp,entropy=policy.log_probs([trace])
    assert np.isfinite(lp.detach().numpy()).all() and np.isfinite(entropy.detach().numpy()).all()
    assert policy.update([trace],[1.],elite_traces=[trace])['gradient_norm']>0
