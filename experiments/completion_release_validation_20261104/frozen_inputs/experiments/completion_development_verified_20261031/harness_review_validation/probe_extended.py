"""Prove local Win32 extended-path copy/read on an owned constructed target."""
import hashlib,json,os,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.completion_development_verified_20261031 import study

def extended(path):
    text=str(path.absolute())
    return Path('\\\\?\\'+text) if os.name=='nt' and not text.startswith('\\\\?\\') else path

source=max((p for p in study.freeze_inputs() if p.is_file()),key=lambda p:len(str(p)))
logical=HERE/'harness_review_validation'/'extended_probe'/source.relative_to(ROOT)
assert logical.is_relative_to(HERE/'harness_review_validation')
target=extended(logical)
target.parent.mkdir(parents=True,exist_ok=True)
shutil.copyfile(extended(source),target)
source_hash=hashlib.sha256(extended(source).read_bytes()).hexdigest()
target_hash=hashlib.sha256(target.read_bytes()).hexdigest()
result=dict(status='passed' if source_hash==target_hash else 'failed',source=str(source),logical_target=str(logical),logical_length=len(str(logical)),extended_length=len(str(target)),sha256=target_hash,scope='One constructed owned file copy/read, not a formal archive or freeze.')
target.unlink()
(HERE/'harness_review_validation/extended_probe.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
