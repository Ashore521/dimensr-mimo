"""Read-only release artifact/index/receipt audit. Never fits or scores a model."""
import base64
import csv
from datetime import datetime, timezone
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
REL=ROOT/'experiments/release_170_20261104'


def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(p.read_bytes())
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


lock=read(ROOT/'experiments/compositional_completion_20261031/CANDIDATE_LOCK.json')
assert sha(ROOT/'experiments/compositional_completion_20261031/CANDIDATE_LOCK.json')=='51103c4e7085da90695f7845df5a312ca6ff693dc686bbc651772bf63c74b452'
assert {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (ROOT/'src').rglob('*.py')}==lock['source_files']
assert sha(REL/'COMPATIBILITY_REVIEW.md')=='9e29bf393d1c0a9193e4e7d666984d2d14dabcaaf55abb43a39b99a2e2b897f2'
assert sha(REL/'compatibility_review_checks.json')=='e372dc1879a33ca2e9a9cc3f45ecc03dd3b28404303d164364e420e428cb0a56'
validation=read(REL/'RELEASE_VALIDATION.json');packaging=read(REL/'PACKAGING.json')
manifest=read(REL/'INSTALLATION_MANIFEST.json')
assert sha(REL/'PACKAGING.json')==validation['packaging_sha256']
assert sha(REL/'INSTALLATION_MANIFEST.json')==validation['installation_manifest_sha256']==packaging['manifest_sha256']
assert manifest['source_files']==lock['source_files']
for rel,h in validation['logs'].items():assert sha(ROOT/rel)==h,rel
assert sha(REL/'check_installation.py')==packaging['script_sha256']
raw=read(ROOT/'outputs/release_170_20261104/installation_check_02/INSTALLATION_CHECK.json')
assert sha(ROOT/'outputs/release_170_20261104/installation_check_02/INSTALLATION_CHECK.json')==packaging['raw_installation_receipt_sha256']
for key in ('status','version','checked_model_count','legacy_native_checked','experiment_imports_blocked',
        'default_configuration_passed','installed_source_count','native_physical_fixed_point_predictions_exact',
        'prediction_probe','manifest_sha256','script_sha256'):
    assert raw[key]==packaging[key],key
assert len(raw['checked_models'])==len(manifest['models'])==135
for a,b in zip(raw['checked_models'],manifest['models']):
    for key in ('native','native_sha256','physical_sha256'):assert a[key]==b[key]
    for key in ('native','physical'):assert sha(ROOT/b[key])==b[key+'_sha256']
for row in manifest['legacy_native']:assert sha(ROOT/row['native'])==row['native_sha256']
installed=ROOT/'outputs/r170installed'
assert Path(raw['installed_package_path']).resolve()==(installed/'dimensr_mimo/__init__.py').resolve()
for rel,h in lock['source_files'].items():assert sha(installed/rel[4:])==h

for stage,field in (('completion_development_verified_20261031','observed_development'),('completion_release_validation_20261104','locked_new_output_pairs')):
    folder=ROOT/'experiments'/stage;evidence=validation['scientific_evidence'][field]
    for name,key in (('SUMMARY.json','summary_sha256'),('POST_RUN_REVIEW.md','post_run_review_sha256'),
            ('post_review_checks.json','post_review_checks_sha256')):
        assert sha(folder/name)==evidence[key],name
    s=read(folder/'SUMMARY.json')
    assert s['technical_pass'] is evidence['technical_pass'] is True
    assert s['adoption_pass'] is evidence['scientific_adoption_pass']
    assert s['methods']['base']['both_accepted']==evidence['control_accepted']
    assert s['methods']['completion']['both_accepted']==evidence['candidate_accepted']
dev=ROOT/'experiments/completion_development_verified_20261031'
dev_accept=read(dev/'POST_REVIEW_ACCEPTANCE.json')
assert dev_accept['status']=='accepted_for_release_validation' and len(dev_accept['sha256'])==8
for rel,h in dev_accept['sha256'].items():assert sha(dev/rel)==h
new=ROOT/'experiments/completion_release_validation_20261104'
new_received=read(new/'POST_REVIEW_RECEIVED.json')
assert new_received['scientific_adoption_pass'] is False
assert sha(new/'POST_RUN_REVIEW.md')==new_received['post_run_review_sha256']
assert sha(new/'post_review_checks.json')==new_received['post_review_checks_sha256']

forbidden_suffixes={'.pdf','.doc','.docx','.npy','.npz','.parquet','.xlsx','.xls','.pem','.key'}
forbidden_parts={'outputs','.venv','__pycache__','.pytest_cache','.env','.aws'}


def safe_names(names):
    for name in names:
        p=PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in name,name
        assert p.suffix.lower() not in forbidden_suffixes,name
        assert not(forbidden_parts & {v.lower() for v in p.parts}),name


artifact_checks={}
wheel=ROOT/'dist/dimensr_mimo-1.7.0-py3-none-any.whl'
sdist=ROOT/'dist/dimensr_mimo-1.7.0.tar.gz'
for p in (wheel,sdist):assert sha(p)==validation['artifacts'][p.name]
with zipfile.ZipFile(wheel) as z:
    names=z.namelist();assert len(names)==len(set(names));safe_names(names)
    assert all(n.startswith(('dimensr_mimo/','dimensr_mimo-1.7.0.dist-info/')) for n in names)
    sources={f'src/{n}':digest(z.read(n)) for n in names if n.startswith('dimensr_mimo/') and n.endswith('.py')}
    assert sources==lock['source_files']
    metadata=BytesParser().parsebytes(z.read('dimensr_mimo-1.7.0.dist-info/METADATA'))
    assert metadata['Version']=='1.7.0' and metadata['Name']=='dimensr-mimo'
    assert metadata.get_payload(decode=True).decode('utf-8').replace('\r\n','\n').strip()==(ROOT/'README.md').read_text(encoding='utf-8').strip()
    records=list(csv.reader(io.StringIO(z.read('dimensr_mimo-1.7.0.dist-info/RECORD').decode())))
    assert {r[0] for r in records}==set(names)
    for name,h,size in records:
        if not h:continue
        content=z.read(name)
        assert h=='sha256='+base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode()
        assert int(size)==len(content)
    artifact_checks['wheel']=dict(sha256=sha(wheel),members=len(names),source_files=41,record_hashes_verified=True,metadata_readme_exact=True)
with tarfile.open(sdist,'r:gz') as t:
    members=t.getmembers();names=[m.name for m in members];assert len(names)==len(set(names));safe_names(names)
    assert all(m.isfile() or m.isdir() for m in members)
    assert all(n.startswith('dimensr_mimo-1.7.0') for n in names)
    prefix='dimensr_mimo-1.7.0/'
    sources={n[len(prefix):]:digest(t.extractfile(n).read()) for n in names if n.startswith(prefix+'src/dimensr_mimo/') and n.endswith('.py')}
    assert sources==lock['source_files']
    exact_docs=['README.md','docs/RELEASE_1.7.0.zh-CN.md','pyproject.toml','MANIFEST.in','conftest.py','experiments/conftest.py']
    for relative in exact_docs:
        assert t.extractfile(prefix+relative).read()==(ROOT/relative).read_bytes(),relative
    metadata=BytesParser().parsebytes(t.extractfile(prefix+'PKG-INFO').read())
    assert metadata['Version']=='1.7.0' and metadata['Name']=='dimensr-mimo'
    artifact_checks['sdist']=dict(sha256=sha(sdist),members=len(names),source_files=41,exact_documents=exact_docs)

index_before=git('ls-files','--stage','-z')
index={}
for entry in index_before.split(b'\0'):
    if not entry:continue
    details,path=entry.split(b'\t',1);mode,blob,stage=details.split()
    assert stage==b'0'
    index[path.decode('utf-8')]=(mode.decode(),blob.decode())
safe_names(index)
tree={}
for name,(mode,blob) in index.items():
    node=tree;parts=name.split('/')
    for part in parts[:-1]:node=node.setdefault(part,{})
    node[parts[-1]]=(mode,blob)


def tree_hash(node):
    content=b''
    for name,value in sorted(node.items(),key=lambda pair:(pair[0]+('/' if isinstance(pair[1],dict) else '')).encode()):
        mode,blob=('40000',tree_hash(value)) if isinstance(value,dict) else value
        content+=mode.encode()+b' '+name.encode()+b'\0'+bytes.fromhex(blob)
    return hashlib.sha1(b'tree '+str(len(content)).encode()+b'\0'+content).hexdigest()


index_tree=tree_hash(tree)
assert index_tree=='2446f0a1796c82ac59030e0f807edf1ab655619e',index_tree
review_paths=list(lock['source_files'])+['README.md','docs/RELEASE_1.7.0.zh-CN.md','pyproject.toml','MANIFEST.in','.gitattributes','.github/workflows/ci.yml',
    'experiments/release_170_20261104/RELEASE_VALIDATION.json','experiments/release_170_20261104/PACKAGING.json',
    'experiments/release_170_20261104/INSTALLATION_MANIFEST.json','experiments/release_170_20261104/PUBLICATION_AUDIT.json',
    'experiments/completion_release_validation_20261104/POST_REVIEW_RECEIVED.json']
for stage in ('completion_development_verified_20261031','completion_release_validation_20261104'):
    review_paths.extend('experiments/'+stage+'/'+name for name in ('FREEZE.json','ALL_MODELS_FROZEN.json','SCORES.json','VERIFICATION.json','SUMMARY.json','POST_RUN_REVIEW.md','post_review_checks.json'))
review_paths.extend('experiments/release_170_20261104/validation/'+n for n in ('ci_source_tests_06.log','sdist_tests_02.log','installation_check_02.log'))
index_verified={};text_normalizations=[]
for rel in review_paths:
    content=git('cat-file','blob',index[rel][1])
    work=(ROOT/rel).read_bytes()
    if content!=work:
        assert rel in ('README.md','docs/RELEASE_1.7.0.zh-CN.md','pyproject.toml','MANIFEST.in','.gitattributes','.github/workflows/ci.yml'),rel
        assert content.replace(b'\r\n',b'\n')==work.replace(b'\r\n',b'\n'),rel
        text_normalizations.append(rel)
    index_verified[rel]=digest(content)
assert git('ls-files','--stage','-z')==index_before
whitespace=read(ROOT/'outputs/release_170_20261104/whitespace_audit.json')
assert whitespace['git_diff_check_exit_code']==2 and whitespace['issue_count']==2306
assert set(whitespace['extensions'])=={'.log','.diff','.md'}
metadata_path=Path(r'C:\Users\608\.codex\sessions\2026\11\04\rollout-2026-11-04T21-48-48-01a1a6d1-c925-7992-8ad8-da5462267a2a.jsonl')
contexts=[]
for line in metadata_path.read_text(encoding='utf-8').splitlines():
    row=json.loads(line)
    if row['type']=='turn_context':
        p=row['payload'];contexts.append(dict(timestamp=row['timestamp'],turn_id=p['turn_id'],model=p['model'],effort=p['effort']))
assert contexts[-1]['model']=='gpt-6.1-sol' and contexts[-1]['effort']=='high'
result=dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),artifacts=artifact_checks,
    candidate_lock_sha256=validation['candidate_lock_sha256'],source_count=41,installation_models_manifest_verified=135,
    installed_source_hashes_verified=41,legacy_model_manifest_verified=1,metadata_context=contexts[-1],
    index_tree_computed_read_only=index_tree,index_paths=len(index),index_path_scan_passed=True,
    index_paths_blob_verified=len(index_verified),index_file_hashes=index_verified,index_snapshot_unchanged=True,
    nonfrozen_text_line_ending_normalizations=text_normalizations,
    scientific_receipts_bound=True,new_task_scientific_adoption=False,compatibility_pre_review_bytes_unchanged=True,
    script_sha256=sha(Path(__file__)),independent_numerical_replay=False)
(OUT/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='index_file_hashes'}))
