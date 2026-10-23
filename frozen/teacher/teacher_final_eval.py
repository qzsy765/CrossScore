#!/usr/bin/env python3
"""CrossScore V1.2 final epoch200 teacher evaluation, strictly read-only checkpoints.
Official ID test and all six OOD datasets used only after fixed teacher model chosen.
Independent of training and without changing any hyperparameter.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import sys,json,hashlib,socket,datetime,argparse
from pathlib import Path
import numpy as np
import torch
from b2_protocol import Teacher,NUM_CLASSES,eval_ood_scores,full_sha256,OOD
ROOT=Path("/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020")
sys.path.insert(0,str(ROOT/"audit_scripts"))
from eval_uploaded_checkpoints import id_loader,loader_for,scores_and_correct
STAGE=ROOT/"B2_V1.2_teacher_20261020"
expected={
"cifar100":"8ffbd65a1f3457abfaf46d548ccb582ad9f0d3c83cc296807f9dc7e06667f8c6",
"cifar10":"80ea46669bb0eb8582746315358e97b0a404960363e86fe90ab64bb8ba26a41b"}
host_by_id={
"cifar100":"crdnotebook-2101132525233999873-scn26lemcn-98190",
"cifar10":"crdnotebook-2111244007934251010-scn26lemcn-74356"}
def write_new(path,obj):
    if path.exists():raise FileExistsError(path)
    temp=path.with_name(path.name+".pending."+str(os.getpid()))
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
    if path.exists():raise FileExistsError(path)
    os.link(temp,path)
    temp.unlink()
p=argparse.ArgumentParser()
p.add_argument("--dataset",choices=["cifar100","cifar10"],required=True)
p.add_argument("--workers",type=int,default=4)
p.add_argument("--batch-size",type=int,default=128)
a=p.parse_args()
if socket.gethostname()!=host_by_id[a.dataset]:raise RuntimeError("Wrong host for "+a.dataset)
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
torch.backends.cudnn.benchmark=True
j=STAGE/"formal_runs"/(a.dataset+"__TEACHER200")
checkpoint=j/"epoch200_final.pth"
done=json.loads((j/"DONE.json").read_text())
if len((j/"history.jsonl").read_text().splitlines())!=200:raise RuntimeError("Not 200 epochs")
if done["epoch"]!=200 or done["mode"]!="formal" or done["selection"]!="EPOCH200_FINAL_ONLY":raise RuntimeError("Not final epoch200")
ck_sha=full_sha256(checkpoint)
if ck_sha!=done["weights_sha256"] or ck_sha!=expected[a.dataset]:raise RuntimeError("Teacher checkpoint hash mismatch")
model=Teacher(NUM_CLASSES[a.dataset])
weights=torch.load(checkpoint,map_location="cpu",weights_only=True)
model.load_state_dict(weights,strict=True)
print("STRICT_LOAD_OK",a.dataset,"nkeys",len(weights),"sha256",ck_sha,flush=True)
model=model.cuda().eval()
types=("energy_neg","msp_neg","entropy")
out=STAGE/"teacher_final_independent_eval"/a.dataset
out.mkdir(parents=True,exist_ok=False)
ids,nid=id_loader(a.dataset,a.batch_size,a.workers)
sid,acc,nread=scores_and_correct(model,ids,torch.device("cuda"),types)
assert nid==nread==10000
print("OFFICIAL_ID_TEST_FINISHED",a.dataset,"n",nid,"acc",acc,flush=True)
metrics={}
counts={}
np.savez_compressed(out/"ID_test_scores.npz",**sid)
for od in OOD:
    ld,n=loader_for(od,a.batch_size,a.workers)
    so,_,nread=scores_and_correct(model,ld,torch.device("cuda"),types)
    if nread!=n:raise RuntimeError("OOD count incomplete "+od)
    counts[od]=n
    metrics[od]={k:eval_ood_scores(sid[k],so[k]) for k in types}
    np.savez_compressed(out/(od+"__scores.npz"),**so)
    print("OOD_COMPLETE",a.dataset,od,n,"energy_auroc",metrics[od]["energy_neg"]["OOD_positive"]["AUROC"],flush=True)
macro={k:{positive:{metric:float(np.mean([metrics[od][k][positive][metric] for od in OOD])) for metric in ("AUROC","AUPR","FPR95_complete_ties")} for positive in ("OOD_positive","ID_positive")} for k in types}
result={"protocol":"B2-V1.2-TEACHER-EPOCH200-LOCKED","dataset":a.dataset,
  "teacher_file":str(checkpoint),"teacher_sha256":ck_sha,"strict_load":True,
  "full_epoch_history":200,"checkpoints_preselected_from_training_only":True,
  "note":"Official test and six OOD evaluated only after epoch200 final checkpoint fixed; no model selection or hyperparameter tuning",
  "id_count":nid,"id_test_acc":acc,"ood_counts":counts,"metrics":metrics,"macro":macro,
  "scores_orientation":"larger means more OOD","fpr95":"unique thresholds, all tied samples grouped",
  "ood_transform":"legacy Resize(32);CenterCrop(32);Normalize CIFAR10 means",
  "timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
write_new(out/"results.json",result)
write_new(out/"DONE.json",{"dataset":a.dataset,"teacher_sha256":ck_sha,"six_ood":len(counts)==6,
         "results_sha256":full_sha256(out/"results.json"),"id_acc":acc})
print("COMPLETE",a.dataset,"test_acc",acc,"macro_energy",macro["energy_neg"]["OOD_positive"],flush=True)
