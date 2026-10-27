"""Unit legality and directly interpretable complexity of an unsimplified DAG."""
from .graph import System
from ._single.units import unit,encode


def units(system,feature_units,target_units,constant_units):
    # Reconstruct to check every operator, topology and bound-parameter unit.
    System.from_dict(system.to_dict())
    features,targets,constants=[tuple(map(unit,seq)) for seq in (feature_units,target_units,constant_units)]
    if len(system.roots)!=len(targets):raise ValueError('Output count differs from unit declarations.')
    for root,target in zip(system.roots,targets):
        if system.nodes[root].units!=target:raise ValueError('Output dimension mismatch.')
    parameter_units={}
    for node in system.nodes:
        if node.children:continue
        if node.token.startswith('x'):
            index=int(node.token[1:])
            if index>=len(features) or node.units!=features[index]:raise ValueError('Input dimension mismatch.')
        else:
            index=int(node.token[1:])
            if index>=len(constants) or node.units!=constants[index]:raise ValueError('Constant dimension mismatch.')
            parameter_units[node.parameter]=encode(node.units)
    return dict(status='passed',output_units=[encode(u) for u in targets],
                parameter_units=[parameter_units[i] for i in range(system.n_parameters)])


def complexity(system):
    sizes,operations,depths=[],[],[]
    for node in system.nodes:
        sizes.append(1+sum(sizes[i] for i in node.children))
        operations.append(int(bool(node.children))+sum(operations[i] for i in node.children))
        depths.append(1+max((depths[i] for i in node.children),default=0))
    rows=[]
    for root in system.roots:
        reachable=system.reachable(root)
        rows.append(dict(expanded_nodes=sizes[root],expanded_operations=operations[root],depth=depths[root],
            unique_dag_nodes=len(reachable),independent_parameters=len({system.nodes[i].parameter for i in reachable if system.nodes[i].parameter is not None})))
    return dict(shared_dag_vertices=len(system.nodes),shared_graph_size=system.size,
                shared_dag_operations=sum(bool(n.children) for n in system.nodes),
                maximum_output_depth=max(depths[r] for r in system.roots),independent_parameters=system.n_parameters,
                expanded_total_nodes=sum(r['expanded_nodes'] for r in rows),
                expanded_total_operations=sum(r['expanded_operations'] for r in rows),outputs=rows)
