"""Inspect sealed training records without fitting, selecting, or test scoring."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'experiments/new_task_validation_20261030'
sys.path.insert(0, str(ROOT / 'src'))
from dimensr_mimo import audit
from dimensr_mimo.graph import System


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(frozen):
    for name, expected in frozen['files'].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT), name
        assert sha(path) == expected, name
        assert sha(HERE / 'frozen_inputs' / name) == expected, name


def main():
    started = time.monotonic()
    output = HERE / 'DIAGNOSIS.json'
    if output.exists():
        raise RuntimeError('Preserve existing diagnosis.')
    frozen = read(HERE / 'FREEZE.json')
    check(frozen)
    seal = read(SOURCE / 'ALL_MODELS_FROZEN.json')
    assert seal['freeze_sha256'] == sha(SOURCE / 'FREEZE.json')
    rows = []
    for name in frozen['models']:
        assert time.monotonic() - started < 60, 'Read-only audit deadline.'
        path = ROOT / name
        assert seal['files'][path.relative_to(SOURCE).as_posix()] == sha(path)
        model = read(path)
        candidates = model['candidates']
        closure = model['diagnostics']['sharing_closure']
        count = closure['base_candidate_count']
        limits = model['config']['acceptance']
        relative = model['config']['degradation_relative']
        absolute = model['config']['degradation_absolute']
        training_bounds = [v * (1 + relative) + absolute for v in model['reference_train']]
        baseline_bounds = model['post_selection']['base_training_bounds']
        records = []
        for index, candidate in enumerate(candidates):
            graph = System.from_dict(candidate['system'])
            shape = audit.complexity(graph)
            errors = candidate['errors']
            assert len(errors) == 2 and all(0 <= v < float('inf') for v in errors)
            accurate = all(a <= b for a, b in zip(errors, limits['nrmse']))
            simple = all(shape[k] <= limits[k] for k in limits if k != 'nrmse')
            protected = all(a <= b for a, b in zip(errors, training_bounds))
            protected &= all(a <= b for a, b in zip(errors, baseline_bounds))
            records.append(dict(index=index, training_errors=errors, shape=shape,
                                accurate=accurate, simple=simple, protected=protected))

        def counts(group):
            result = dict(total=len(group), training_accuracy=sum(r['accurate'] for r in group),
                          structure=sum(r['simple'] for r in group),
                          training_accuracy_and_structure=sum(r['accurate'] and r['simple'] for r in group),
                          training_accuracy_structure_and_protection=sum(
                              r['accurate'] and r['simple'] and r['protected'] for r in group))
            assert 0 <= result['training_accuracy_structure_and_protection'] <= result['training_accuracy_and_structure']
            assert result['training_accuracy_and_structure'] <= min(result['training_accuracy'], result['structure']) <= len(group)
            return result

        best_index = min(range(len(candidates)), key=lambda i: (
            sum(candidates[i]['errors']) / 2, candidates[i]['complexity'],
            json.dumps(candidates[i]['system'], sort_keys=True)))
        best = candidates[best_index]
        fixed = model['post_selection']['candidate_index']
        assert candidates[fixed]['system'] == model['best']['system']
        assert candidates[fixed]['constants'] == model['best']['constants']
        repair = closure.get('precision_repair', {})
        base_counts = counts(records[:count])
        if model['config']['precision_repair']:
            assert repair['training_eligible'] == base_counts['training_accuracy']
            assert repair['triggered'] == (base_counts['training_accuracy'] == 0)
        rows.append(dict(name=path.parent.name, model_sha256=sha(path),
            base=base_counts, full=counts(records),
            final_selection_status=model['post_selection']['status'],
            final_selection=records[fixed],
            training_minimum=records[best_index],
            training_minimum_formulas=System.from_dict(best['system']).formulas(best['constants'], model['feature_names']),
            selected_formulas=System.from_dict(model['best']['system']).formulas(model['best']['constants'], model['feature_names']),
            repair={k:repair.get(k) for k in ('triggered','training_eligible','new_evaluations','valid','protected','stop_reason')},
            training_bounds=training_bounds, base_choice_training_bounds=baseline_bounds))
    assert len(rows) == len({r['name'] for r in rows}) == 32
    check(frozen)
    elapsed = time.monotonic() - started
    assert elapsed < 60, 'Read-only audit deadline.'
    result = dict(status='passed', freeze_sha256=sha(HERE / 'FREEZE.json'),
                  elapsed_seconds=elapsed, validation_metrics_used=False, test_scores_read=False,
                  new_fits=0, new_searches=0, production_changes=0, rows=rows,
                  scope='Stored training errors verified in the source batch; graph metrics recomputed here. No new candidate selection or validation/test scoring.')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], models=len(rows), elapsed_seconds=elapsed)))


if __name__ == '__main__':
    main()
