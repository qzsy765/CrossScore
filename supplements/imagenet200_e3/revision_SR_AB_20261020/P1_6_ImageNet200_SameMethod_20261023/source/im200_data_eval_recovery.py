from pathlib import Path
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PUBLIC=Path(os.environ.get('CROSSSCORE_PUBLIC_DATA_ROOT','/SET_PUBLIC_ROOT'))
LISTROOT=PUBLIC/'openood_imglists/benchmark_imglist/imagenet200'
TRAINROOT=Path(os.environ.get('CROSSSCORE_IM200_TRAINROOT','/SET_TRAIN_ROOT'))
IM1K=PUBLIC/'imagenet1k'
DOWN=Path(os.environ.get('CROSSSCORE_OPENOOD_IMAGE_ROOT','/SET_OPENOOD_ROOT'))
MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)

def train_tf():
    return transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
def eval_tf():
    return transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])

def read_list(name):
    rows=[]
    with open(LISTROOT/name) as f:
        for line in f:
            p,y=line.rstrip().rsplit(' ',1); rows.append((p,int(y)))
    return rows

def label_to_wnid():
    m={}
    for p,y in read_list('train_imagenet200.txt'):
        wn=p.split('/')[2]
        if y in m and m[y]!=wn: raise RuntimeError((y,m[y],wn))
        m[y]=wn
    if sorted(m)!=list(range(200)): raise RuntimeError(f'labels={len(m)} range={min(m)}..{max(m)}')
    return m
LABEL2WNID=label_to_wnid()

def resolve_path(kind,p,y):
    parts=p.split('/')
    if kind=='train': return TRAINROOT/parts[-2]/parts[-1]
    if kind in ('val','test'):
        return IM1K/'val'/LABEL2WNID[y]/parts[-1]
    if kind=='ssb_hard': return DOWN/p
    if kind=='inaturalist': return DOWN/p
    if kind=='ninco':
        rel=parts[1:].copy()
        renames={'Caracal_caracal_caracal':'Caracal caracal caracal',
                 'high_heels':'high heels'}
        if rel[0] in renames:
            oldname=rel[0]
            rel[0]=renames[oldname]
            rel[-1]=rel[-1].replace(oldname+'_',rel[0]+'_',1)
        return PUBLIC/'ninco/NINCO/NINCO_OOD_classes'/Path(*rel)
    if kind=='openimage_o': return PUBLIC/'openood_openimage_o'/Path(*parts[1:])
    if kind=='texture': return PUBLIC/'texture'/Path(*parts[1:])
    raise ValueError(kind)

class ImgList(Dataset):
    def __init__(self,listname,kind,transform):
        self.rows=read_list(listname); self.kind=kind; self.transform=transform
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        p,y=self.rows[i]; fp=resolve_path(self.kind,p,y)
        with Image.open(fp) as im: x=self.transform(im.convert('RGB'))
        return x,y

def dataset(kind,train=False):
    table={
      'train':('train_imagenet200.txt','train'), 'val':('val_imagenet200.txt','val'), 'test':('test_imagenet200.txt','test'),
      'ssb_hard':('test_ssb_hard.txt','ssb_hard'), 'ninco':('test_ninco.txt','ninco'), 'inaturalist':('test_inaturalist.txt','inaturalist'),
      'openimage_o':('test_openimage_o.txt','openimage_o'), 'texture':('test_textures.txt','texture')}
    ln,k=table[kind]; return ImgList(ln,k,train_tf() if train else eval_tf())
