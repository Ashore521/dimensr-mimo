"""Independent paired verification including actual generated-system legality."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.reference_budget_20261027 import study
from experiments.phase_budget_20261027 import evaluate as previous
from experiments.native_api_20261026.verify import require
previous.HERE=HERE;previous.PREVIOUS=HERE.parent/'grammar_cost_20261027'
previous.original.HERE=HERE;previous.original.study.HERE=HERE
original_verify=previous.original.verify_cell


def verify_cell(job,score,frozen):
    row=original_verify(job,score,frozen);folder=HERE/'runs'/study.supervisor.name(job)
    model=study.read(folder/'model.json');counts=study.read(folder/'GENERATION_CHECK.json')
    require(model['config']['reference_budget']=='expanded','Wrong reference budget.')
    require(counts['passed'] and counts['invalid_expanded_systems']==0,'Invalid generated systems.')
    actual=sum(e.get('proposals',0) for e in model['history'] if e['phase']=='neural')
    require(counts['neural_proposals']==actual and actual>0,'Generation audit count differs.')
    row['generation_check']=counts
    return row


previous.original.verify_cell=verify_cell

if __name__=='__main__':
    previous.original.evaluate();previous.verify()
    summary=study.read(HERE/'SUMMARY.json')
    summary.update(historical_controls=21,paired_token_reference_controls=6,earlier_controls=15,
        scope='Expanded reference budget on historical data; changes search distribution, not blind validation.')
    for pair in summary['pairs']:
        for key in list(pair):
            if key.startswith('global_'):pair['token_reference_'+key[7:]]=pair.pop(key)
            elif key.startswith('phased_'):pair['expanded_reference_'+key[7:]]=pair.pop(key)
    study.write(HERE/'SUMMARY.json',summary)
