#!/usr/bin/env python3
"""Serial C/E/R study; wait for recovery evaluation and preserve all failures."""
import concurrent.futures,csv,datetime,fcntl,hashlib,json,math,os,shutil,socket,subprocess,sys,time
from pathlib import Path
STUDY=Path(__file__).resolve().parent
ROOT=Path('/root/private_data/projects/OOD/cross_score_v2')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(name,v):
 tmp=STUDY/(name+'.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(STUDY/name)
def heartbeat(phase,**kw):write('HEARTBEAT.json',{'pid':os.getpid(),'utc':now(),'phase':phase,**kw})
def idle_gpu():
 while True:
  s=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
  if not s:return
  heartbeat('WAIT_EXTERNAL_GPU_PROCESS',gpu_processes=s);time.sleep(30)
def child(cmd,logname,phase):
 idle_gpu();lp=STUDY/'logs'/logname;lp.parent.mkdir(exist_ok=True)
 with lp.open('x') as log:
  p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'1','CROSSSCORE_IM200_TRAINROOT':str(PLAN['train_cache'])})
  while p.poll() is None:heartbeat(phase,child_pid=p.pid,log=str(lp));time.sleep(10)
  if p.returncode:raise RuntimeError({'phase':phase,'returncode':p.returncode,'log':str(lp)})
def stage_training_cache():
 source=Path(PLAN['persistent_train_root']);dest=Path(PLAN['train_cache']);assert not dest.exists(),dest
 rows=[x.rsplit(' ',1)[0] for x in Path(PLAN['train_list']).read_text().splitlines()];assert len(rows)==258951
 dest.mkdir(parents=True,exist_ok=False);started=time.time();used=0
 def copy(p):
  rel=Path(*Path(p).parts[-2:]);src=source/rel;dst=dest/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  b=src.read_bytes();digest=hashlib.sha256(b).hexdigest()
  with dst.open('xb') as f:f.write(b)
  assert sha(dst)==digest,rel
  return str(rel),len(b),digest
 with (STUDY/'TRAIN_CACHE_SHA256.csv').open('x',newline='') as f,concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
  wr=csv.writer(f);wr.writerow(['relative_path','size_bytes','source_and_destination_sha256'])
  # Bounded submissions prevent 258951 futures from retaining image bytes.
  for start in range(0,len(rows),256):
   for record in pool.map(copy,rows[start:start+256]):wr.writerow(record);used+=record[1]
   cgroup=Path('/sys/fs/cgroup');limit=(cgroup/'memory.max').read_text().strip() if (cgroup/'memory.max').exists() else 'max'
   headroom=int(limit)-int((cgroup/'memory.current').read_text()) if limit!='max' else 1<<60
   if used>80*(1<<30) or shutil.disk_usage('/dev/shm').free<24*(1<<30) or headroom<32*(1<<30):raise RuntimeError('Training cache exceeded frozen resource budget; preserve partial cache')
   if start%2560==0:heartbeat('STAGING_VERIFIED_TRAIN_CACHE',n_copied=min(start+256,len(rows)),bytes=used)
 write('TRAIN_CACHE_COMPLETE.json',{'n':len(rows),'bytes':used,'elapsed_seconds':time.time()-started,'csv_sha256':sha(STUDY/'TRAIN_CACHE_SHA256.csv'),'every_source_and_destination_sha256_equal':True})
def validate_new(r,smoke=False):
 p=Path(r['path']);m=json.loads((p/'manifest.json').read_text());done=json.loads((p/'COMPLETE.json').read_text());hist=list(csv.DictReader((p/'history.csv').open()))
 assert m['components']==dict(zip(['C','E','R'],r['components'])) and m['seed']==r['seed']
 assert m['trainer_sha256']==PLAN['trainer_sha256'] and m['teacher_temperature']==6 and m['student_probability_temperature']==1
 n=1 if smoke else 60;assert done['complete_epochs']==n and [int(x['epoch']) for x in hist]==list(range(1,n+1))
 assert all(math.isfinite(float(v)) for row in hist for v in row.values())
 skipped=sum(int(x['skipped_amp_steps']) for x in hist)
 if smoke:assert skipped==0,('SMOKE_AMP_SKIPPED',r)
 return {**r,'checkpoint_sha256':sha(p/'best.pt'),'manifest_sha256':sha(p/'manifest.json'),'history_sha256':sha(p/'history.csv'),'complete_sha256':sha(p/'COMPLETE.json'),'amp_skipped_steps':skipped,'initial_state_sha256':m['initial_state_sha256'],'best_val_acc':done['best_val_acc']}
def train_cmd(r,smoke=False):
 c,e,k=r['components'];cmd=[sys.executable,'-u',str(STUDY/'r2_component_factorial_train.py'),'--mode','CrossScore_T6_Corrected','--seed',str(r['seed']),'--conf',str(c),'--entropy',str(e),'--rank',str(k),'--out',r['path']]
 if smoke:cmd+=['--epochs','1','--batch-size','32','--workers','2','--max-train-batches','2','--max-val-batches','1']
 return cmd
def main():
 global PLAN
 PLAN=json.loads((STUDY/'FROZEN_FACTORIAL_PROTOCOL.json').read_text());assert sha(Path(__file__))==PLAN['controller_sha256']
 assert not (STUDY/'CONTROLLER_STARTED.json').exists();write('CONTROLLER_STARTED.json',{'pid':os.getpid(),'utc':now(),'host':socket.gethostname(),'protocol_sha256':sha(STUDY/'FROZEN_FACTORIAL_PROTOCOL.json')})
 recovery=Path(PLAN['recovery_study'])
 while not (recovery/'ALL_SIX_OOD_COMPLETE.json').exists():
  if (recovery/'CONTROLLER_ERROR.json').exists():raise RuntimeError('Predecessor P2C1 evaluation failed; no training started')
  heartbeat('WAIT_P2C1_OOD_COMPLETE');time.sleep(30)
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==PLAN['gpu_uuid']
 locks=ROOT/'.gpu_device_locks';locks.mkdir(exist_ok=True);lock=(locks/(uuid+'.lock')).open('a+')
 while True:
  try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:heartbeat('WAIT_SHARED_GPU_LOCK');time.sleep(30)
 owner=locks/(uuid+'.owner.json');owner.write_text(json.dumps({'state':'RUNNING','study':str(STUDY),'pid':os.getpid(),'utc':now()},indent=2)+'\n')
 try:
  idle_gpu()
  for r in PLAN['source_dependencies']+PLAN['dataset_lists']:assert sha(r['path'])==r['sha256'],r['path']
  child([sys.executable,str(STUDY/'r2_component_factorial_preflight.py')],'cpu_preflight_repeat.log','REVALIDATE_CPU_AND_ANCHORS')
  stage_training_cache()
  smokes=[]
  for components in [[1,0,0],[0,1,0],[0,0,1]]:
   label='C%dE%dR%d'%tuple(components);r={'label':label,'components':components,'seed':42,'path':str(STUDY/'smokes'/label)}
   child(train_cmd(r,True),'smoke_'+label+'.log','SMOKE_'+label);smokes.append(validate_new(r,True))
  write('THREE_SINGLE_COMPONENT_SMOKES_PASS.json',{'utc':now(),'records':smokes,'not_formal_results':True})
  records=[]
  for r in PLAN['new_runs']:
   child(train_cmd(r),r['label']+'_seed'+str(r['seed'])+'.log','TRAIN_'+r['label']+'_seed'+str(r['seed']));records.append(validate_new(r));write('NEW_TRAINING_PROGRESS.json',{'n_complete':len(records),'n_planned':28,'records':records,'utc':now()})
  anchors=json.loads((STUDY/'CPU_PREFLIGHT_PASS.json').read_text())['anchors']
  from r2_component_factorial_preflight import audit_run
  anchors=[audit_run(r) for r in anchors]
  for r in anchors:r['checkpoint_sha256']=r['frozen_hashes']['best.pt']
  allrecords=anchors+records;assert len(allrecords)==40 and len({(tuple(r['components']),r['seed']) for r in allrecords})==40
  for seed in PLAN['seeds']:assert len({r['initial_state_sha256'] for r in records if r['seed']==seed})==1
  write('ALL_40_ID_FROZEN_GATE.json',{'status':'ALL_40_ID_TRAINING_FROZEN','utc':now(),'records':allrecords,'n_old':12,'n_new':28,'no_classification_performance_exclusion':True,'same_previously_examined_OOD_domains':True,'new_OOD_started_before_gate':False})
  for r in sorted(allrecords,key=lambda r:(r['seed'],r['components'])):
   child([sys.executable,'-u',str(STUDY/'r2_component_factorial_evaluate.py'),'--label',r['label'],'--seed',str(r['seed'])],'eval_'+r['label']+'_seed'+str(r['seed'])+'.log','OOD_'+r['label']+'_seed'+str(r['seed']))
  child([sys.executable,str(STUDY/'r2_component_factorial_analyze.py')],'factorial_analysis.log','FACTORIAL_ANALYSIS')
  write('ALL_DONE.json',{'status':'TRAINING_AND_OOD_AND_FACTORIAL_ANALYSIS_COMPLETE','utc':now(),'must_audit_before_manuscript_use':True})
 finally:
  owner.write_text(json.dumps({'state':'RELEASED','study':str(STUDY),'pid':os.getpid(),'utc':now()},indent=2)+'\n');fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
if __name__=='__main__':
 try:main()
 except BaseException as exc:write('CONTROLLER_ERROR.json',{'error':repr(exc),'utc':now(),'partial_outputs_preserved':True});raise
