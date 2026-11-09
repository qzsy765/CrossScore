from pathlib import Path
import json,csv,itertools,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description='Plot audited seed-level factorial values')
p.add_argument('--evidence-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();SRC=a.output_dir;SRC.mkdir(exist_ok=True,parents=True)
R=json.loads((a.evidence_dir/'FACTORIAL_SCIENTIFIC_RESULT.json').read_text());V=json.loads((a.evidence_dir/'INDEPENDENT_FACTORIAL_STATISTICS_AUDIT.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(10.5,3.7),gridspec_kw={'width_ratios':[1.7,1]})
bits=list(itertools.product([0,1],repeat=3));labs=['CE','R','E','E+R','C','C+R','C+E','C+E+R'];source=[]
for i,(b,lab) in enumerate(zip(bits,labs)):
 x=R['cell_summaries'][f'energy/AUROC/{b}'];ys=np.array(x['seed_values']);axs[0].scatter(i+np.linspace(-.12,.12,5),ys,s=19,facecolors='white',edgecolors='black',linewidth=.8,zorder=3)
 axs[0].errorbar(i,x['mean'],yerr=x['sample_sd'],fmt='s',color='black',markersize=4,capsize=3,linewidth=1.1,zorder=4)
 for seed,value in zip(range(42,47),ys):source.append({'panel':'A','configuration':lab,'seed':seed,'Energy_AUROC':float(value)})
axs[0].set_xticks(range(8),labs,rotation=30);axs[0].set_ylabel('Energy AUROC');axs[0].set_ylim(.575,.91);axs[0].set_title('A  Component combinations',loc='left',fontweight='bold')
lo=V['paired_energy_contrasts']['R_effect_E0_average_C']['AUROC'];hi=V['paired_energy_contrasts']['R_effect_E1_average_C']['AUROC']
for i,seed in enumerate(range(42,47)):
 y=[lo['seed_values'][i],hi['seed_values'][i]];axs[1].plot([0,1],y,color='.6',linewidth=.8,marker='o',markersize=4,markerfacecolor='white',markeredgecolor='black')
 for setting,value in zip(['E off','E on'],y):source.append({'panel':'B','configuration':setting,'seed':seed,'Energy_AUROC':float(value)})
axs[1].plot([0,1],[lo['mean'],hi['mean']],color='black',linewidth=2,marker='s',markersize=5);axs[1].axhline(0,color='black',linestyle='--',linewidth=.8);axs[1].set_xticks([0,1],['E off','E on']);axs[1].set_xlim(-.25,1.25);axs[1].set_ylabel('Increment from adding R');axs[1].set_title('B  Ranking depends on entropy',loc='left',fontweight='bold')
for ax in axs:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='.9',linewidth=.6)
fig.tight_layout(pad=1.5);fig.savefig(SRC/'factorial_seed_results.pdf');fig.savefig(SRC/'factorial_seed_results.png',dpi=300);plt.close(fig)
with (SRC/'factorial_figure_source.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=['panel','configuration','seed','Energy_AUROC']);w.writeheader();w.writerows(source)
