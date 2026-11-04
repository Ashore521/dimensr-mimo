"""Check hard-stop and failure propagation with disposable non-research children."""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('lineage_supervisor', HERE/'run_bounded.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
results = []
for name, source, code, reason in (
    ('deadline', "import pathlib,json,time\np=pathlib.Path(__file__).parent\n(p/'BUDGET.json').write_text(json.dumps(dict(start_unix=time.time(),limit_seconds=.2)))\ntime.sleep(30)\n", 1, 'global_deadline'),
    ('missing_verification', 'pass\n', 1, None),
    ('success', "import pathlib,json\n(pathlib.Path(__file__).parent/'VERIFICATION.json').write_text(json.dumps(dict(status='passed')))\n", 0, None)):
    folder = ROOT/'outputs'/('lineage-supervisor-selfcheck-'+name)
    folder.mkdir()
    (folder/'study.py').write_text(source)
    runner.HERE = folder
    actual = 0
    try:
        runner.main()
    except SystemExit as error:
        actual = error.code
    record = json.loads((folder/'EXECUTION.json').read_text())
    assert actual == code and record['termination_reason'] == reason
    assert record['wall_seconds'] < 10
    results.append(dict(case=name, exitcode=actual, execution=record))
(HERE/'SUPERVISOR_SELFCHECK.json').write_text(json.dumps(dict(status='passed', rows=results), indent=2)+'\n')
print('Three disposable-process supervisor checks passed; no symbolic-regression fits.')
