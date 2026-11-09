#!/usr/bin/env python3
"""P2C1 full paired temperature run controller. One fixed seed per GPU, no concurrent training."""
from pathlib import Path
import os,sys,time,json,subprocess,hashlib,socket,datetime
P=Path('/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020/P1_6_ImageNet200_SameMethod_20261023/P2_C1_gradient_norm_matched_IDonly_20261027')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def main():
 seed=int(sys.argv[1]);assert seed in (44,45,46)
 meta=json.loads((P/'FROZEN_CONTROLLER_PLAN.json').read_text())
 assert sha(Path(__file__))==meta['controller_source_sha256']
 assert sha(P/'train_p2c1_normmatch_v2.py')==meta['trainer_source_sha256']
 assert sha(P/'p2c1_id_only_eval.py')==meta['ID_eval_source_sha256']
 assert sha(P/'FROZEN_PROTOCOL.json')==meta['science_protocol_sha256']
 assert sha(P/'FROZEN_AMENDMENT_01.json')==meta['amendment_sha256']
 smoke=json.loads((P/'FROZEN_V2_SMOKE_GATE.json').read_text())
 assert smoke['status']=='SMOKE_V2_PASS_BEFORE_FORMAL'
 lock=P/f'.node_seed{seed}_locked'
 lock.mkdir(exist_ok=False)
 try:
  for arm in ['T1_E1_NORMMATCH','T6_E1_NORMMATCH']:
   run=f'{arm}_seed{seed}'
   out=P/run
   assert not out.exists(), 'refuse duplicate or incomplete path:'+str(out)
   assert not (P/f'node_seed{seed}_ERROR.json').exists()
   for attempt in range(600):
    g=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    assert len(g)==1
    mem,util=[int(v.strip()) for v in g[0].split(',')]
    if mem<300 and util<35:break
    if attempt%20==0: print('WAIT_GPU_FREE',seed,arm,datetime.datetime.now(datetime.timezone.utc).isoformat(),g,flush=True)
    time.sleep(90)
   else:raise RuntimeError(('GPU_UNAVAILABLE_15_HOURS',run))
   print('P2C1_TRAIN_START',run,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
   cmd=['/root/private_data/envs/illumood/bin/python','-u',str(P/'train_p2c1_normmatch_v2.py'),
     '--mode','CrossScore_T6_Corrected','--variant',arm,'--seed',str(seed),
     '--epochs','60','--batch-size','512','--workers','16','--out',str(out)]
   env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CROSSSCORE_IM200_TRAINROOT':'/dev/shm/crossscore_im200/train'}
   with (P/f'train_{run}.log').open('x') as log:
    ret=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
   if ret.returncode:raise RuntimeError(('TRAIN_FAILED',run,ret.returncode))
   assert (out/'COMPLETE.json').is_file()
   record=json.loads((out/'COMPLETE.json').read_text())
   assert record['run_type']=='P2C1_NORM_MATCHED_ID_ONLY' and record['complete_epochs']==60
   print('P2C1_TRAIN_COMPLETE',run,record['best_val_acc'],flush=True)
   with (P/f'id_eval_{run}.log').open('x') as log:
    ret=subprocess.run(['/root/private_data/envs/illumood/bin/python','-u',str(P/'p2c1_id_only_eval.py'),'--run',run],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
   if ret.returncode:raise RuntimeError(('ID_EVAL_FAILED',run,ret.returncode))
   assert (out/'ID_ONLY_DIAGNOSTICS.json').exists()
   print('P2C1_ID_EVAL_COMPLETE',run,flush=True)
  (P/f'node_seed{seed}_DONE.json').write_text(json.dumps({'seed':seed,'status':'DONE_BOTH_ARMS','node':socket.gethostname(),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
  print('P2C1_SEED_DONE',seed,flush=True)
 except BaseException as e:
  (P/f'node_seed{seed}_ERROR.json').write_text(json.dumps({'seed':seed,'host':socket.gethostname(),'error':repr(e),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
  raise
 finally:
  lock.rmdir()
if __name__=='__main__':main()
