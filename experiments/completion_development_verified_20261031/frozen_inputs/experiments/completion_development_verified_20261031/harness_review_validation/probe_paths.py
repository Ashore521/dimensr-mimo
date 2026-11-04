"""Read-only dependency hashes and an owned, immediately removed copy probe."""
import hashlib,json,shutil,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.completion_development_verified_20261031 import study,gate
from experiments.compositional_completion_20261031 import harness

inherited=set(harness.freeze_inputs(study));current=set(study.freeze_inputs())
expected={HERE/n for n in ['HARNESS_REVIEW.md','TEST_VALIDATION.json','PREFLIGHT.json']}
missing=[];bad=[];hashed=0
for p in sorted(current):
    if not p.is_file():missing.append(str(p));continue
    try:hashlib.sha256(p.read_bytes()).hexdigest();hashed+=1
    except OSError as error:bad.append(dict(path=str(p),error=repr(error)))
future=sorted([(len(str(HERE/'frozen_inputs'/p.relative_to(ROOT))),p,HERE/'frozen_inputs'/p.relative_to(ROOT)) for p in current if p.is_file()],reverse=True)
probes=[]
for length,source,dest in future[:5]:
    probe=HERE/'harness_review_validation'/'copy_probe'/source.relative_to(ROOT)
    row=dict(source=str(source),source_length=len(str(source)),future_destination=str(dest),future_length=length,probe_destination=str(probe),probe_length=len(str(probe)))
    try:
        probe.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,probe)
        row['copied_and_readable']=hashlib.sha256(probe.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
    except OSError as error:row.update(copied_and_readable=False,error=repr(error))
    finally:
        if probe.is_file():probe.unlink()
    probes.append(row)
result=dict(inherited_count=len(inherited),current_count=len(current),removed=[str(p) for p in inherited-current],hashed=hashed,missing=missing,hash_read_errors=bad,future_destination_lengths_over_259=sum(q[0]>259 for q in future),maximum_future_destination_length=future[0][0],copy_probes=probes)
(HERE/'harness_review_validation/path_probe.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='copy_probes'},indent=2));print(json.dumps(probes[:1],indent=2))
