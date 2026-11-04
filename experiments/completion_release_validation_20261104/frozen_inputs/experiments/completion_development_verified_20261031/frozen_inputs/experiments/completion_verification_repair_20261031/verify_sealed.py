"""Isolated re-verification of unchanged models; no fitting or score writes."""
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.completion_historical_20261031 import study
from experiments.compositional_completion_20261031 import verification as original
from experiments.native_api_20261026.run import read, write, sha
from experiments.native_api_20261026.verify import require

HISTORICAL = study.HERE
TIMING = ('epoch_allowance_seconds', 'neural_until_seconds', 'gp_until_seconds')
ARTIFACTS = ('FREEZE.json', 'ALL_MODELS_FROZEN.json', 'SCORES.json', 'VERIFICATION.json',
             'SUMMARY.json', 'EXECUTION.json', 'FROZEN_INPUT_ARCHIVE.json',
             'POST_RUN_REVIEW.md', 'post_review_checks.json', 'POST_REVIEW_RECEIVED.json')


def normalized_model(model):
    """Classify three paths only after proving saved deadlines did not bind."""
    cfg = model['config']
    require(model['diagnostics']['deadline_reached'] is False, 'Global deadline reached.')
    history = model['history']
    budgets = [r for r in history if r['phase'] == 'phase_budget']
    require(len(budgets) == cfg['epochs'] and [r['step'] for r in budgets] == list(range(cfg['epochs'])), 'Phase budget coverage drift.')
    for row in budgets:
        values = [row[n] for n in TIMING]
        require(all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in values), 'Invalid phase time.')
        require(values[1] <= values[2] <= row['exploration_until_seconds'] <= cfg['wall_seconds'], 'Phase deadline order drift.')
        neural = [r for r in history if r['phase'] == 'neural' and r['step'] == row['step']]
        gp = [r for r in history if r['phase'] == 'gp' and r['step'] in range(row['step']*cfg['gp_generations'], (row['step']+1)*cfg['gp_generations'])]
        require(len(neural) == 1 and neural[0]['proposals'] == cfg['population_size'], 'Incomplete neural phase.')
        require(len(gp) == cfg['gp_generations'] and len({r['step'] for r in gp}) == len(gp), 'Incomplete GP phase.')
        for event, deadline in [(neural[0], values[1])] + [(r, values[2]) for r in gp]:
            elapsed = event['elapsed_seconds']
            require(type(elapsed) in (float, int) and math.isfinite(elapsed) and 0 <= elapsed < deadline, 'Phase deadline binding.')
    result = original.normalized(model)
    for row in result['history']:
        if row['phase'] == 'phase_budget':
            for name in TIMING:
                del row[name]
    return result


class SealedVerifier(original.Verifier):
    def control_identity(self, job, score):
        folder = self.here/'runs'/self.study.name(job)
        old_job = dict(job, method='repair')
        prior = self.study.PRIOR/'runs'/self.study.name(old_job)
        for filename in ('model.json', 'result.json', 'delivery.json', 'pool.json'):
            normalize = normalized_model if filename == 'model.json' else original.normalized
            require(normalize(read(folder/filename)) == normalize(read(prior/filename)), 'Archived control identity drift: '+filename)
        archive = read(self.study.PRIOR/'SCORES.json')['rows']
        require(len(archive) == len({original.key(r) for r in archive}) == self.study.EXPECTED and all(original.complete(r) for r in archive), 'Archived score matrix drift.')
        old_scores = [r for r in archive if original.key(r) == original.key(dict(job=old_job))]
        require(len(old_scores) == 1 and original.complete(old_scores[0]), 'Archived score denominator drift.')
        keep = lambda r: original.normalized({k:v for k,v in r.items() if k not in {'job', 'resources'}})
        require(keep(score) == keep(old_scores[0]), 'Archived control score drift.')
        model = read(folder/'model.json')
        if not model['diagnostics']['sharing_closure']['precision_repair']['triggered']:
            old_base = read(self.study.PRIOR/'runs'/self.study.name(dict(job, method='base'))/'model.json')
            identity = self.legacy.paired_identity(model, old_base)
            require(all(identity[k] for k in ('base_prefix_matches', 'base_cache_matches', 'skipped_pool_matches', 'skipped_choice_matches')), 'Control skip identity drift.')
        return True


def verify():
    manifest = read(HERE/'FREEZE.json')
    require(all(sha(ROOT/n) == h for n,h in manifest['files'].items()), 'Repair input drift.')
    require(not any((HERE/n).exists() for n in ('VERIFICATION.json', 'SUMMARY.json')), 'Keep repair results.')
    verifier = SealedVerifier(study)
    frozen, sealed = verifier.guards()
    table = read(HISTORICAL/'SCORES.json')
    require(table['all_models_sha256'] == sha(HISTORICAL/'ALL_MODELS_FROZEN.json') and sealed['frozen_unix'] <= table['scored_unix'], 'Original score chain drift.')
    require(read(HISTORICAL/'SUMMARY.json')['technical_pass'] is False and read(HISTORICAL/'SUMMARY.json')['adoption_pass'] is False, 'Original failure changed.')
    counts = Counter(original.key(r) for r in table['rows'])
    index = {original.key(r):r for r in table['rows']}
    rows = []
    for job in study.matrix():
        try:
            require(time.time() < manifest['deadline_unix'], 'Verification deadline expired.')
            key = original.key(dict(job=job));score = index.get(key, {})
            require(counts[key] == 1 and score.get('job') == job and original.complete(score), 'Invalid score cell.')
            row = verifier.base.verify_cell(job, score, frozen)
            row['closure'], supplement = verifier.closure_evidence(job, frozen)
            row['repair' if job['method'] == 'base' else 'completion'] = supplement
            if job['method'] == 'base':
                row['archived_control_identity'] = verifier.control_identity(job, score)
            require(time.time() < manifest['deadline_unix'], 'Verification deadline exceeded.')
        except Exception as error:
            row = dict(job=job, verified=False, error=repr(error))
        rows.append(row)
        print(study.name(job), row.get('error', 'verified'), flush=True)
    summary = verifier.summarize(table['rows'], rows)
    integrity = all(sha(ROOT/n) == h for n,h in manifest['files'].items())
    within = time.time() < manifest['deadline_unix']
    if not integrity or not within:
        summary.update(technical_pass=False, adoption_pass=False)
    summary.update(original_verdict='failed', verification_of_sealed_results=True, development_authorized=False)
    write(HERE/'VERIFICATION.json', dict(status='passed' if summary['technical_pass'] else 'failed',
        expected_systems=34, verified_systems=sum(r['verified'] for r in rows), rows=rows,
        budget_met=within, inputs_unchanged=integrity, verified_unix=time.time(),
        freeze_sha256=sha(HERE/'FREEZE.json'), original_sha256=manifest['original_sha256'], original_verdict='failed'))
    write(HERE/'SUMMARY.json', summary)


def run():
    require(not any((HERE/n).exists() for n in ('FREEZE.json', 'EXECUTION.json', 'verification.log')), 'Keep existing execution; no restart.')
    for name in ('PLAN.md', 'TEST_VALIDATION.json', 'PRE_RUN_REVIEW.md'):
        require((HERE/name).is_file(), 'Missing prerequisite: '+name)
    study.check_candidate_lock()
    start = time.time();deadline = min(start+600, study.budget())
    require(deadline > start, 'Original historical wall deadline expired.')
    inputs = [HISTORICAL/n for n in ARTIFACTS]
    inputs += [study.SHARED/'CANDIDATE_LOCK.json', study.SHARED/'verification.py', study.SHARED/'harness.py']
    inputs += [p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    original_hashes = {n:sha(HISTORICAL/n) for n in ARTIFACTS}
    write(HERE/'FREEZE.json', dict(frozen_unix=start, deadline_unix=deadline, verification_cap_seconds=600,
        original_deadline_unix=study.budget(), original_sha256=original_hashes,
        files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(inputs))},
        no_fit_search_or_rescoring=True, original_verdict='failed'))
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1')
    reason = None
    with (HERE/'verification.log').open('x', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, str(Path(__file__)), 'verify'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=max(.1, deadline-time.time()))
        except subprocess.TimeoutExpired:
            reason = 'verification_deadline';process.kill();code = process.wait()
    try:
        passed = read(HERE/'VERIFICATION.json')['status'] == 'passed'
    except (OSError, ValueError, KeyError, TypeError):
        passed = False
    write(HERE/'EXECUTION.json', dict(returncode=code, termination_reason=reason, wall_seconds=time.time()-start,
        verification_passed=passed, no_training_performed=True))
    if code or reason or not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    (verify if len(sys.argv) > 1 and sys.argv[1] == 'verify' else run)()
