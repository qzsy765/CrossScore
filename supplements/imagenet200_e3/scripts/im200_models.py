from pathlib import Path
import os
import torch
from torchvision.models import resnet18,resnet50
from torchvision.datasets import ImageFolder
from im200_data import LABEL2WNID
ROOT=Path(os.environ.get('CROSSSCORE_PROJECT_ROOT','/SET_PROJECT_ROOT'))
CROOT=ROOT/'results/im200_checkpoints'
BASE=CROOT/'imagenet200_resnet18_224x224_base_e90_lr0.1_default'
TEACHER=CROOT/'resnet50-0676ba61.pth'
def student(seed):
    m=resnet18(weights=None,num_classes=200)
    p=BASE/f's{seed}'/'best.ckpt'
    m.load_state_dict(torch.load(p,map_location='cpu'))
    return m,p
def teacher_restricted():
    m=resnet50(weights=None,num_classes=1000)
    m.load_state_dict(torch.load(TEACHER,map_location='cpu'))
    idset=ImageFolder(str(Path(os.environ.get('CROSSSCORE_PUBLIC_DATA_ROOT','/SET_PUBLIC_ROOT'))/'imagenet1k/val'))
    idx=torch.tensor([idset.class_to_idx[LABEL2WNID[i]] for i in range(200)],dtype=torch.long)
    return m,idx
