"""Read-only source-lock comparison; writes only this implementation receipt."""
import difflib
import hashlib
import json
from pathlib import Path
import platform


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STAGE = OUT.parent
CHANGED = {'src/dimensr_mimo/config.py', 'src/dimensr_mimo/acceptance.py',
           'src/dimensr_mimo/sharing.py', 'src/dimensr_mimo/regressor.py'}
NEW = ['src/dimensr_mimo/completion.py', 'tests/test_compositional_completion.py',
       'docs/COMPOSITIONAL_COMPLETION.zh-CN.md']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


lock = json.loads((STAGE/'BASE_SOURCE_LOCK.json').read_text(encoding='utf-8'))
rows, changes = {}, []
for name, original in lock['source_files'].items():
    live = ROOT/name
    frozen = ROOT/lock['source_archive']/name
    assert sha(frozen) == original, 'Frozen source archive changed: '+name
    current = sha(live)
    rows[name] = dict(base_sha256=original, current_sha256=current, changed=current != original)
    if current != original:
        changes.append(name)
        assert name in CHANGED, 'Unexpected source change against implementation baseline: '+name
        diff = ''.join(difflib.unified_diff(frozen.read_text(encoding='utf-8').splitlines(True),
            live.read_text(encoding='utf-8').splitlines(True), fromfile='locked/'+name, tofile=name))
        (OUT/(live.stem+'.diff')).write_text(diff, encoding='utf-8')
assert set(changes) == CHANGED
receipt = dict(status='implementation_constructive_tests_only_independent_review_pending',
    python_version=platform.python_version(), source_lock_verified=True,
    plan_sha256=sha(STAGE/'PLAN.md'), design_review_sha256=sha(STAGE/'DESIGN_REVIEW.md'),
    source_files=rows, new_files={name:sha(ROOT/name) for name in NEW},
    package_versions_sha256=sha(OUT/'package_versions.json'),
    formal_matrices_executed=False, scientific_acceptance=False,
    logs={path.name:sha(path) for path in sorted(OUT.glob('pytest_*.log'))})
(OUT/'IMPLEMENTATION_CHECKS.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(dict(changed_existing=changes, new_files=receipt['new_files'],
                     frozen_source_count=len(rows), plan_sha256=receipt['plan_sha256']), indent=2))
