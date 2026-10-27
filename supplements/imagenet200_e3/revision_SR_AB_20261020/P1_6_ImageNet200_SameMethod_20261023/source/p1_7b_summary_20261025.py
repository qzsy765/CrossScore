#!/usr/bin/env python3
"""Aggregate frozen five-seed P1-7B multi-score diagnostic metrics. Exploratory only."""
from pathlib import Path
import os
import json,csv,itertools,math
import numpy as np
from scipy.stats import t as student_t,ttest_1samp
EXP=Path(os.environ.get('CROSSSCORE_PROJECT_ROOT','/SET_PROJECT_ROOT'))/'revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023'
OUT=EXP/'P1_7B_score_family_20261025'
SEEDS=[42,43,44,45,46]
MODES=['CE','CrossScore_T6_Corrected']
DOMAINS=['ssb_hard','ninco','inaturalist','texture','openimage_o']
keys=['AUROC','AUPR','FPR95_complete_ties']
def main():
    records={}
    for mode in MODES:
        for seed in SEEDS:
            run=f'{mode}_seed{seed}'
            p=OUT/run
            assert (p/'COMPLETE.json').is_file(),f'incomplete {run}'
            records[(mode,seed)]=json.loads((p/'metrics.json').read_text())
    first=records[(MODES[0],SEEDS[0])]
    mean_sd=[];paired=[]
    for score in first['score_family']:
        for domain in DOMAINS+['near_macro','far_macro','five_macro']:
            for metric in keys:
                def value(rec):
                    if domain=='near_macro': return float(np.mean([rec['ood_metrics'][score][d][metric] for d in DOMAINS[:2]]))
                    if domain=='far_macro': return float(np.mean([rec['ood_metrics'][score][d][metric] for d in DOMAINS[2:]]))
                    if domain=='five_macro': return float(np.mean([rec['ood_metrics'][score][d][metric] for d in DOMAINS]))
                    return float(rec['ood_metrics'][score][domain][metric])
                vals={}
                for mode in MODES:
                    v=np.array([value(records[(mode,s)]) for s in SEEDS])
                    vals[mode]=v
                    mean_sd.append({'score':score,'domain':domain,'metric':metric,'mode':mode,'n':5,'mean':float(v.mean()),'sd':float(v.std(ddof=1))})
                dif=vals[MODES[1]]-vals[MODES[0]]
                mu=float(dif.mean());se=float(dif.std(ddof=1)/math.sqrt(5));tcrit=float(student_t.ppf(.975,4))
                ttest=ttest_1samp(dif,0)
                flips=[abs(np.mean(dif*np.array(signs))) for signs in itertools.product([-1,1],repeat=5)]
                pexact=sum(v>=abs(mu)-1e-12 for v in flips)/len(flips)
                paired.append({'score':score,'domain':domain,'metric':metric,'crossscore_minus_ce':mu,
                 'ci95_low':mu-tcrit*se,'ci95_high':mu+tcrit*se,
                 'p_t_two_sided':float(ttest.pvalue),'exact_sign_flip_p_two_sided':pexact,
                 'number_positive':int((dif>0).sum()),'seed_differences':';'.join(f'{s}:{v:+.6f}' for s,v in zip(SEEDS,dif))})
    for filename,rows in [('score_family_mean_sd.csv',mean_sd),('score_family_paired.csv',paired)]:
        out=OUT/filename
        assert not out.exists(),f'refuse overwrite {out}'
        with out.open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    focus=[]
    for score in first['score_family']:
        key='AUROC'
        d=next(x for x in paired if x['score']==score and x['domain']=='five_macro' and x['metric']==key)
        ce=next(x for x in mean_sd if x['score']==score and x['domain']=='five_macro' and x['metric']==key and x['mode']=='CE')
        cs=next(x for x in mean_sd if x['score']==score and x['domain']=='five_macro' and x['metric']==key and x['mode']=='CrossScore_T6_Corrected')
        focus.append({'score':score,'CE_5macro_AUROC':ce['mean'],'CrossScore_5macro_AUROC':cs['mean'],'delta':d['crossscore_minus_ce'],'exact_signflip_p':d['exact_sign_flip_p_two_sided']})
    summary={'stage':'P1-7B exploratory frozen-checkpoint multi-score diagnostic',
             'priority':'mechanism diagnosis, NOT an independent confirmatory test; no OOD-specific model selection',
             'seeds':SEEDS,'n_models':10,'domains':DOMAINS,'crossscore_version':'corrected-T6 on ImageNet-200',
             'primary_formal_energy_result_location':str(EXP/'paired_comparison.csv'),
             'methods':MODES,'metric_higher_better':['AUROC','AUPR'],'metric_lower_better':['FPR95_complete_ties'],
             'multiple_comparisons':'Many correlated exploratory comparisons. Nominal p values are descriptive, not multiplicity-controlled.',
             'focus':focus,'mean_sd_file':str(OUT/'score_family_mean_sd.csv'),'paired_file':str(OUT/'score_family_paired.csv')}
    path=OUT/'score_family_summary.json'
    assert not path.exists()
    path.write_text(json.dumps(summary,indent=2)+'\n')
    print('SCORE_FAMILY_DIAGNOSTIC_COMPLETE',json.dumps(focus,indent=2),flush=True)
if __name__=='__main__':main()

