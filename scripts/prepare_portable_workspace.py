#!/usr/bin/env python3
"""Prepare separate path-adapted runnable copies; keep frozen/ unmodified."""
from pathlib import Path
import argparse,shutil,sys
p=argparse.ArgumentParser()
p.add_argument('--workspace',required=True,help='new nonexisting working directory')
p.add_argument('--datasets',required=True,help='existing external dataset root')
a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
workspace=Path(a.workspace).expanduser().resolve()
data=Path(a.datasets).expanduser().resolve()
if workspace.exists():sys.exit('Refusing overwrite of existing workspace')
if not data.is_dir():sys.exit('Dataset directory not found')
root=workspace/'revision_SR_AB_20261020'
source_map={
 'frozen/b2':'B2_V1.2_student_20261020/frozen_src',
 'frozen/rfkd':'P1_SR_ReviewerEvidence_20261022/P1_5_RFKD_Mechanism_Comparator_20261022/frozen_src',
 'frozen/teacher':'B2_V1.2_teacher_20261020/frozen_src',
 'frozen/audit':'audit_scripts'}
for src,dest in source_map.items():
 dst=root/dest
 shutil.copytree(repo/src,dst)
 for py in dst.glob('*.py'):
  content=py.read_text(encoding='utf8')
  content=content.replace('/root/private_data/projects/OOD/cross_score_v2/revision_SR_AB_20261020',str(root))
  content=content.replace('/root/private_data/CrossScore_SR_AB_upload_20261020',str(workspace/'external_original_checkpoints'))
  content=content.replace('/root/private_data/datasets/public',str(data))
  py.write_text(content,encoding='utf8')
print('WORKSPACE_CREATED',root)
print('Supply data, CIFAR-style teacher weights and release gate files per README. Adapted files have different hashes from frozen originals.')
