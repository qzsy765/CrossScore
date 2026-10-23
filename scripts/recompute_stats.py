import csv,statistics,itertools
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'evidence/paired_comparison.csv'
r=list(csv.DictReader(p.open()))
for ds in ('cifar10','cifar100'):
 d=[float(x['CS_minus_RFKD_AUROC']) for x in r if x['dataset']==ds]
 assert len(d)==5
 mean=statistics.mean(d)
 perms=[sum(a*s for a,s in zip(d,z))/5 for z in itertools.product((-1,1),repeat=5)]
 pval=sum(abs(x)>=abs(mean)-1e-12 for x in perms)/32
 print(ds,'mean_delta',round(mean,6),'exact_p',pval)
