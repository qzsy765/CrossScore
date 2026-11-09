"""Independent CSV, provenance and seed-level contrast audit after raw-score audit."""
from pathlib import Path
import json,csv,itertools,hashlib,math,ast
import numpy as np
from scipy.stats import t
import argparse
args=argparse.ArgumentParser(description='Recompute archived factorial summaries on CPU')
args.add_argument('--evidence-dir',type=Path,required=True)
args.add_argument('--output-dir',type=Path,required=True)
a=args.parse_args()
P=a.evidence_dir.resolve()
O=a.output_dir.resolve();O.mkdir(parents=True,exist_ok=True)
R=json.loads((P/'FACTORIAL_SCIENTIFIC_RESULT.json').read_text())
plan=json.loads((P/'FROZEN_FACTORIAL_PROTOCOL.json').read_text())
gate=json.loads((P/'ALL_40_ID_FROZEN_GATE.json').read_text())
proof=json.loads((P/'audit/INDEPENDENT_RAW_METRIC_AUDIT_PASS.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert proof['all_pass'] and proof['n_metric_checks']==3600
assert sha(P/'audit/INDEPENDENT_RAW_METRIC_AUDIT_PASS.json')==json.loads((P/'ALL_DONE.json').read_text())['audit_sha256']
assert R['protocol_sha256']==sha(P/'FROZEN_FACTORIAL_PROTOCOL.json')
assert not gate['new_OOD_started_before_gate']
rows=list(csv.DictReader((P/'ALL_DOMAIN_METRICS.csv').open()))
assert len(rows)==1200
domains=['ssb_hard','ninco','inaturalist','texture','openimage_o']
seeds=[42,43,44,45,46];metrics=['AUROC','AUPR','FPR95_complete_ties']
scores=sorted({x['score'] for x in rows});assert len(scores)==6
models={};checks=[]
for q in (P/'model_evaluation_records').glob('*/metrics.json'):
 m=json.loads(q.read_text());done=json.loads((q.parent/'COMPLETE.json').read_text());run=q.parent.name
 assert sha(q)==done['metrics_sha256']
 a=next(x for x in proof['inputs'] if x['run']==run)
 assert done['raw_sha256']==a['raw_sha256'] and a['metrics_sha256']==sha(q)
 g=next(x for x in gate['records'] if x['label']==m['label'] and x['seed']==m['seed'])
 assert m['checkpoint_sha256']==g['checkpoint_sha256'] and m['protocol_sha256']==sha(P/'FROZEN_FACTORIAL_PROTOCOL.json')
 assert m['all_40_gate_sha256']==sha(P/'ALL_40_ID_FROZEN_GATE.json') and m['no_OOD_selection']
 models[(tuple(m['components']),m['seed'])]=m
assert len(models)==40
cube={}
for score in scores:
 for metric in metrics:
  for bits in itertools.product([0,1],repeat=3):
   for seed in seeds:
    values=[]
    for domain in domains:
     matches=[r for r in rows if tuple(int(r[k]) for k in ['C','E','R'])==bits and int(r['seed'])==seed and r['score']==score and r['domain']==domain]
     assert len(matches)==1
     value=float(matches[0][metric]);expected=models[bits,seed]['ood_metrics'][score][domain][metric]
     assert abs(value-expected)<=1e-15;values.append(value)
    cube[score,metric,bits,seed]=float(np.mean(values))
def summary(values):
 x=np.asarray(values,dtype=float);mu=float(x.mean());sd=float(x.std(ddof=1));half=float(t.ppf(.975,4))*sd/math.sqrt(5)
 signs=np.asarray(list(itertools.product([-1,1],repeat=5)))
 p=float(np.mean(np.abs((signs*x).mean(axis=1))>=abs(mu)-1e-14))
 return {'n_seeds':5,'mean':mu,'sample_sd':sd,'t95_low':mu-half,'t95_high':mu+half,'two_sided_exact_sign_flip_p':p,'seed_values':x.tolist()}
def audit_one(label,derived,expected):
 for name in ['mean','sample_sd','t95_low','t95_high','two_sided_exact_sign_flip_p']:
  diff=abs(derived[name]-expected[name]);checks.append({'item':label+'/'+name,'abs_difference':diff});assert diff<=1e-12,checks[-1]
 assert np.max(np.abs(np.asarray(derived['seed_values'])-np.asarray(expected['seed_values'])))<=1e-12
for key,expected in R['cell_summaries'].items():
 score,metric,cell=key.split('/');bits=ast.literal_eval(cell);audit_one(key,summary([cube[score,metric,bits,s] for s in seeds]),expected)
for key,expected in R['factorial_effects'].items():
 score,metric,effect=key.split('/');axes=['CER'.index(x) for x in effect];other=[i for i in range(3) if i not in axes];values=[]
 for seed in seeds:
  stratified=[]
  for held in itertools.product([0,1],repeat=len(other)):
   terms=[]
   for varied in itertools.product([0,1],repeat=len(axes)):
    bits=[0,0,0]
    for a,v in zip(other,held):bits[a]=v
    for a,v in zip(axes,varied):bits[a]=v
    sign=(-1)**(len(axes)-sum(varied));terms.append(sign*cube[score,metric,tuple(bits),seed])
   stratified.append(sum(terms))
  values.append(float(np.mean(stratified)))
 audit_one(key,summary(values),expected)
contrasts={}
def paired(name,left,right):
 contrasts[name]={metric:summary([cube['energy',metric,left,s]-cube['energy',metric,right,s] for s in seeds]) for metric in metrics}
paired('R_only_minus_CE',(0,0,1),(0,0,0))
paired('E_only_minus_CE',(0,1,0),(0,0,0))
paired('C_only_minus_CE',(1,0,0),(0,0,0))
paired('Full_minus_CE',(1,1,1),(0,0,0))
for c in [0,1]:
 for e in [0,1]:paired('R_effect_at_C%dE%d'%(c,e),(c,e,1),(c,e,0))
contrasts['R_effect_E0_average_C']={metric:summary([float(np.mean([cube['energy',metric,(c,0,1),s]-cube['energy',metric,(c,0,0),s] for c in [0,1]])) for s in seeds]) for metric in metrics}
contrasts['R_effect_E1_average_C']={metric:summary([float(np.mean([cube['energy',metric,(c,1,1),s]-cube['energy',metric,(c,1,0),s] for c in [0,1]])) for s in seeds]) for metric in metrics}
diag={}
for bits in itertools.product([0,1],repeat=3):
 d=[models[bits,s] for s in seeds];diag[str(bits)]={
 'ID_test_top1':summary([m['id_test_top1'] for m in d]),
 'ID_energy_sd':summary([m['distribution_statistics']['test']['energy']['sd'] for m in d]),
 'ID_entropy_mean':summary([m['distribution_statistics']['test']['entropy']['mean'] for m in d]),
 'centered_energy_AUROC':summary([cube['centered_energy','AUROC',bits,s] for s in seeds]),
 'aux_uncert_AUROC':summary([cube['aux_uncert','AUROC',bits,s] for s in seeds])}
result={'status':'PASS_RAW_AUDIT_AND_INDEPENDENT_FACTORIAL_STATISTICS','n_models':40,'n_domain_rows':1200,'server_raw_metric_checks':3600,'n_summary_scalar_checks':len(checks),'max_summary_abs_difference':max(x['abs_difference'] for x in checks),'historical_models':12,'new_models':28,'primary':'Energy AUROC, equal-domain macro, paired five seeds','raw_metric_audit':proof,'scientific_result_sha256':sha(P/'FACTORIAL_SCIENTIFIC_RESULT.json'),'domain_csv_sha256':sha(P/'ALL_DOMAIN_METRICS.csv'),'independent_source_sha256':sha(Path(__file__)),'paired_energy_contrasts':contrasts,'ID_diagnostics':diag,'scope':'Local audit recomputed CSV summaries and contrasts independently; raw per-image metrics were independently recomputed on the server. All diagnostics are descriptive and exploratory.'}
(O/'INDEPENDENT_FACTORIAL_STATISTICS_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
(O/'ALL_SUMMARY_SCALAR_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'status':result['status'],'n_scalar_checks':len(checks),'max_abs_difference':result['max_summary_abs_difference'],'paired_AUROC':{n:v['AUROC'] for n,v in contrasts.items()},'ID_diagnostics':{n:{k:round(v['mean'],6) for k,v in d.items()} for n,d in diag.items()}}))
