"""Two independently switchable seed interventions; unchanged joint search."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from dimensr_mimo import regressor
from experiments.typed_coverage_20261026 import cases,schedule,worker as previous
from experiments.unit_composition_20261026 import aligned,structure
read,write,sha=previous.read,previous.write,previous.sha
ARMS=('balanced','aligned','normalized','combined')


@contextmanager
def intervention(method,spec):
    if method not in ARMS:raise ValueError('Unknown intervention.')
    old_generator,old_structure=schedule.generator,regressor.structured_seeds
    if method in ('aligned','combined'):schedule.generator=lambda config,q:aligned.generator(config,spec)
    if method in ('normalized','combined'):regressor.structured_seeds=structure.structured_seeds
    try:yield
    finally:schedule.generator,regressor.structured_seeds=old_generator,old_structure


def run(folder,job,config):
    with intervention(job['method'],cases.design(job['case'])):
        previous.run(folder,dict(job,method='balanced'),config)
    result=read(folder/'result.json');result.update(job=job,intervention=dict(alignment=job['method'] in ('aligned','combined'),normalization=job['method'] in ('normalized','combined')))
    write(folder/'result.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('job');p.add_argument('config');a=p.parse_args()
    run(Path(a.folder),read(Path(a.job)),read(Path(a.config)))
