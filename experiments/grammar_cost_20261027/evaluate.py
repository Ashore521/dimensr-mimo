"""Reuse six-cell independent checks; name the immediately previous controls."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.grammar_cost_20261027 import study
from experiments.phase_budget_20261027 import evaluate as previous
previous.HERE=HERE;previous.PREVIOUS=HERE.parent/'phase_budget_20261027'
previous.original.HERE=HERE;previous.original.study.HERE=HERE

if __name__=='__main__':
    previous.original.evaluate();previous.verify()
    summary=study.read(HERE/'SUMMARY.json')
    summary.update(historical_controls=15,paired_previous_phased=6,earlier_global_and_independent=9,
        scope='Exact grammar implementation repair; same historical data and time limits, not blind validation.')
    for pair in summary['pairs']:
        for key in list(pair):
            if key.startswith('global_'):pair['previous_phased_'+key[7:]]=pair.pop(key)
            elif key.startswith('phased_'):pair['optimized_phased_'+key[7:]]=pair.pop(key)
    study.write(HERE/'SUMMARY.json',summary)
