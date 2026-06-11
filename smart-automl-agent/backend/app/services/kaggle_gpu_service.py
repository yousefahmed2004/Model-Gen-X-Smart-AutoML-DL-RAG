"""
Kaggle GPU Training Service — pure subprocess CLI.
"""
from __future__ import annotations
import json, logging, os, shutil, subprocess, tempfile, time, uuid
from pathlib import Path
from typing import Any
from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_env() -> dict:
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = settings.kaggle_username or ""
    env["KAGGLE_KEY"]      = settings.kaggle_key or ""
    return env


def _ensure_kaggle_json() -> None:
    if not settings.kaggle_username or not settings.kaggle_key:
        raise RuntimeError("KAGGLE_USERNAME and KAGGLE_KEY not set in .env")
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(parents=True, exist_ok=True)
    kfile = kdir / "kaggle.json"
    kfile.write_text(json.dumps({"username": settings.kaggle_username, "key": settings.kaggle_key}))
    try: kfile.chmod(0o600)
    except Exception: pass


def _kaggle_available() -> bool:
    try:
        r = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False


# ── Notebook templates stored as plain strings (no f-string) ────────────

_NB_IMAGE = """
import os,torch,numpy as np
from torchvision import datasets,models,transforms
from torch import nn
from torch.utils.data import DataLoader,SubsetRandomSampler
from sklearn.metrics import accuracy_score,f1_score
import json

ARCH=__ARCH__;EPOCHS=__EPOCHS__;IMG_SIZE=__IMG_SIZE__;BATCH=32
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
print("Device:",DEVICE)

input_root="/kaggle/input"
dirs=[d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root,d))]
data_root=os.path.join(input_root,dirs[0])
print("Data root:",data_root)

has_split=os.path.exists(os.path.join(data_root,"train"))
train_dir=os.path.join(data_root,"train") if has_split else data_root

tfm_t=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.2,.2,.2),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
tfm_v=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

full=datasets.ImageFolder(train_dir,transform=tfm_t)
n_cls=len(full.classes)
print("Classes:",full.classes)

if has_split:
    val_dir=os.path.join(data_root,"val") if os.path.exists(os.path.join(data_root,"val")) else os.path.join(data_root,"test")
    train_loader=DataLoader(full,batch_size=BATCH,shuffle=True,num_workers=2,pin_memory=True)
    val_loader=DataLoader(datasets.ImageFolder(val_dir,transform=tfm_v),batch_size=BATCH,num_workers=2)
else:
    idxs=list(range(len(full)));np.random.shuffle(idxs);sp=int(.8*len(full))
    train_loader=DataLoader(full,batch_size=BATCH,sampler=SubsetRandomSampler(idxs[:sp]),num_workers=2,pin_memory=True)
    val_loader=DataLoader(datasets.ImageFolder(train_dir,transform=tfm_v),batch_size=BATCH,sampler=SubsetRandomSampler(idxs[sp:]),num_workers=2)

if ARCH=="resnet18":
    m=models.resnet18(pretrained=True);m.fc=nn.Linear(m.fc.in_features,n_cls)
elif ARCH=="mobilenet":
    m=models.mobilenet_v2(pretrained=True);m.classifier[1]=nn.Linear(m.classifier[1].in_features,n_cls)
else:
    m=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(64*(IMG_SIZE//4)**2,256),nn.ReLU(),nn.Dropout(.5),nn.Linear(256,n_cls))

m=m.to(DEVICE)
opt=torch.optim.Adam(m.parameters(),lr=1e-3)
sched=torch.optim.lr_scheduler.StepLR(opt,step_size=5,gamma=.5)
best_acc=0;best_w=None

for ep in range(1,EPOCHS+1):
    m.train()
    for xb,yb in train_loader:
        xb,yb=xb.to(DEVICE),yb.to(DEVICE);opt.zero_grad();nn.CrossEntropyLoss()(m(xb),yb).backward();opt.step()
    sched.step();m.eval();c=t=0
    with torch.no_grad():
        for xb,yb in val_loader:
            p=m(xb.to(DEVICE)).argmax(1).cpu();c+=(p==yb).sum().item();t+=len(yb)
    acc=c/t;print(f"Epoch {ep:2d}/{EPOCHS} val_acc={acc:.4f}")
    if acc>best_acc:
        best_acc=acc
        best_w=dict((k,v.cpu().clone()) for k,v in m.state_dict().items())

if best_w: m.load_state_dict(best_w)
torch.save({"model_state":best_w,"meta":{"type":"image_classification","arch":ARCH,"n_classes":n_cls,"class_names":full.classes,"img_size":IMG_SIZE}},"/kaggle/working/model.pt")
avg="binary" if n_cls==2 else "weighted"
all_p,all_t=[],[]
with torch.no_grad():
    for xb,yb in val_loader:
        all_p.extend(m(xb.to(DEVICE)).argmax(1).cpu().numpy());all_t.extend(yb.numpy())
metrics={"accuracy":float(accuracy_score(all_t,all_p)),"f1":float(f1_score(all_t,all_p,average=avg,zero_division=0)),"best_model":ARCH,"n_classes":n_cls,"task_type":"image_classification"}
json.dump(metrics,open("/kaggle/working/metrics.json","w"))
print("METRICS:",json.dumps(metrics))
print("DONE")
"""

_NB_ML = """
import pandas as pd,numpy as np,json,glob
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import accuracy_score,f1_score,r2_score,mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

TARGET="__TARGET__";TASK="__TASK__"
csvs=glob.glob("/kaggle/input/**/*.csv",recursive=True)
print("CSV:",csvs)
df=pd.read_csv(csvs[0]).dropna(subset=[TARGET])
print(df.shape)
y=df[TARGET];X=df.drop(columns=[TARGET])
for c in list(X.columns):
    if X[c].dtype=="O" and X[c].nunique()==len(X): X=X.drop(columns=[c])

num=X.select_dtypes("number").columns.tolist()
cat=[c for c in X.columns if c not in num]
low=[c for c in cat if X[c].nunique()<=50];high=[c for c in cat if c not in low]
if sum(X[c].nunique() for c in low)>500: high=cat;low=[]

np_=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler(with_mean=False))])
op_=Pipeline([("i",SimpleImputer(strategy="most_frequent")),("e",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])
hp_=Pipeline([("i",SimpleImputer(strategy="most_frequent")),("e",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))])
trs=[]
if num: trs.append(("n",np_,num))
if low: trs.append(("l",op_,low))
if high: trs.append(("h",hp_,high))
pre=ColumnTransformer(trs,remainder="drop")

is_cls=TASK=="ml_classification";le=None
if is_cls:
    le=LabelEncoder();y=le.fit_transform(y.astype(str))
    cands=[("Random Forest",RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)),("Logistic Regression",LogisticRegression(max_iter=1000,solver="saga",n_jobs=-1)),("Gradient Boosting",GradientBoostingClassifier(n_estimators=100,random_state=42)),("Decision Tree",DecisionTreeClassifier(random_state=42,max_depth=12))]
    sc="accuracy"
else:
    cands=[("Random Forest",RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1)),("Ridge",Ridge()),("Gradient Boosting",GradientBoostingRegressor(n_estimators=100,random_state=42)),("Decision Tree",DecisionTreeRegressor(random_state=42,max_depth=12))]
    sc="r2"

Xt,Xe,yt,ye=train_test_split(X,y,test_size=.2,random_state=42,stratify=y if is_cls and np.unique(y).size>1 else None)
bn=None;bs=-9999;bp=None
for name,mdl in cands:
    p=Pipeline([("pre",pre),("m",mdl)])
    try:
        s=cross_val_score(p,Xt,yt,cv=3,scoring=sc,n_jobs=-1).mean()
        print(f"  {name:<22} cv={s:.4f}")
        if s>bs: bs=s;bn=name;bp=p
    except Exception as e: print(f"  {name:<22} FAILED:{e}")

bp.fit(Xt,yt);yp=bp.predict(Xe)
if is_cls:
    avg="binary" if len(np.unique(y))==2 else "weighted"
    metrics={"accuracy":float(accuracy_score(ye,yp)),"f1":float(f1_score(ye,yp,average=avg,zero_division=0)),"task_type":TASK,"best_model":bn}
else:
    metrics={"r2":float(r2_score(ye,yp)),"mae":float(mean_absolute_error(ye,yp)),"task_type":TASK,"best_model":bn}

joblib.dump({"pipeline":bp,"feature_columns":list(X.columns),"target_column":TARGET,"task_type":TASK,"classes":list(le.classes_) if le else None},"/kaggle/working/model.joblib")
json.dump(metrics,open("/kaggle/working/metrics.json","w"))
print("METRICS:",json.dumps(metrics))
print("DONE")
"""

_NB_DL_TABULAR = """
import pandas as pd,numpy as np,torch,torch.nn as nn,json,glob
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,f1_score

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
TARGET="__TARGET__";EPOCHS=__EPOCHS__
print("Device:",DEVICE)
csvs=glob.glob("/kaggle/input/**/*.csv",recursive=True)
df=pd.read_csv(csvs[0]).dropna(subset=[TARGET])
y_r=df[TARGET];X_r=df.drop(columns=[TARGET])
for c in X_r.select_dtypes("object").columns:
    le=LabelEncoder();X_r[c]=le.fit_transform(X_r[c].astype(str))
imp=SimpleImputer(strategy="median");X=imp.fit_transform(X_r)
sc=StandardScaler();X=sc.fit_transform(X)
le_y=LabelEncoder();y=le_y.fit_transform(y_r.astype(str));n=len(le_y.classes_)
Xt,Xe,yt,ye=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
Xt=torch.FloatTensor(Xt);yt=torch.LongTensor(yt)
Xe=torch.FloatTensor(Xe);ye_np=np.array(ye)
loader=DataLoader(TensorDataset(Xt,yt),batch_size=256,shuffle=True)
model=nn.Sequential(nn.Linear(X.shape[1],256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(.3),nn.Linear(256,128),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,n)).to(DEVICE)
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
best_acc=0;best_w=None
for ep in range(1,EPOCHS+1):
    model.train()
    for xb,yb in loader:
        xb,yb=xb.to(DEVICE),yb.to(DEVICE);opt.zero_grad();nn.CrossEntropyLoss()(model(xb),yb).backward();opt.step()
    model.eval()
    with torch.no_grad(): pred=model(Xe.to(DEVICE)).argmax(1).cpu().numpy()
    acc=float(accuracy_score(ye_np,pred));print(f"Epoch {ep:2d}/{EPOCHS} val_acc={acc:.4f}")
    if acc>best_acc:
        best_acc=acc
        best_w=dict((k,v.cpu().clone()) for k,v in model.state_dict().items())
if best_w: model.load_state_dict(best_w)
torch.save({"model_state":best_w,"meta":{"type":"tabular_mlp","n_classes":n,"classes":list(le_y.classes_),"input_dim":int(X.shape[1])}},"/kaggle/working/model.pt")
with torch.no_grad(): pred=model(Xe.to(DEVICE)).argmax(1).cpu().numpy()
avg="binary" if n==2 else "weighted"
metrics={"accuracy":float(accuracy_score(ye_np,pred)),"f1":float(f1_score(ye_np,pred,average=avg,zero_division=0)),"task_type":"tabular_dl","best_model":"MLP"}
json.dump(metrics,open("/kaggle/working/metrics.json","w"))
print("METRICS:",json.dumps(metrics));print("DONE")
"""


def _build_notebook(task, target_column, model_arch, epochs, img_size) -> str:
    if task == "dl_image":
        code = (_NB_IMAGE
                .replace("__ARCH__", f'"{model_arch}"')
                .replace("__EPOCHS__", str(epochs))
                .replace("__IMG_SIZE__", str(img_size)))
    elif task in ("ml_classification", "ml_regression"):
        code = (_NB_ML
                .replace("__TARGET__", target_column)
                .replace("__TASK__", task))
    else:
        code = (_NB_DL_TABULAR
                .replace("__TARGET__", target_column)
                .replace("__EPOCHS__", str(epochs)))

    nb = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 4,
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.strip()}],
    }
    return json.dumps(nb)


def train_on_kaggle(
    local_dataset_path: str | Path,
    task: str,
    target_column: str = "",
    text_column: str   = "",
    model_arch: str    = "resnet18",
    epochs: int        = 10,
    img_size: int      = 128,
    dataset_slug: str  = "",
    log_fn=None,
) -> dict[str, Any]:

    def _log(msg):
        logger.info(msg)
        if log_fn: log_fn(msg)

    if not _kaggle_available():
        raise RuntimeError("Kaggle CLI not found. Run: pip install kaggle")

    _ensure_kaggle_json()
    env  = _get_env()
    user = settings.kaggle_username
    uid  = uuid.uuid4().hex[:8]

    # ── Step 1: Upload dataset ───────────────────────────────────────────
    if not dataset_slug:
        _log("📤 Uploading dataset to Kaggle…")
        ds_slug = f"mgx-data-{uid}"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = Path(local_dataset_path)
            if src.is_file():
                shutil.copy2(src, tmp_path / src.name)
            else:
                shutil.copytree(str(src), str(tmp_path / src.name))

            (tmp_path / "dataset-metadata.json").write_text(json.dumps({
                "title":    f"MGX Dataset {uid}",
                "id":       f"{user}/{ds_slug}",
                "licenses": [{"name": "CC0-1.0"}],
            }))

            r = subprocess.run(
                ["kaggle", "datasets", "create", "-p", str(tmp_path), "--dir-mode", "zip"],
                capture_output=True, text=True, env=env
            )
            if r.returncode != 0 and "already exists" not in (r.stdout + r.stderr).lower():
                raise RuntimeError(f"Dataset upload failed:\n{r.stdout}\n{r.stderr}")

        dataset_slug = f"{user}/{ds_slug}"
        _log(f"✓ Dataset uploaded: {dataset_slug}")
    else:
        _log(f"✓ Using existing dataset: {dataset_slug}")

    # ── Step 2: Submit kernel ────────────────────────────────────────────
    kernel_name = f"mgx-train-{uid}"
    kernel_id   = f"{user}/{kernel_name}"
    _log(f"⚡ Submitting to Kaggle GPU: {kernel_id}")

    notebook_json = _build_notebook(
        task=task, target_column=target_column,
        model_arch=model_arch, epochs=epochs, img_size=img_size,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "training.ipynb").write_text(notebook_json)
        (tmp_path / "kernel-metadata.json").write_text(json.dumps({
            "id":                  kernel_id,
            "title":               f"MGX Training {uid}",
            "code_file":           "training.ipynb",
            "language":            "python",
            "kernel_type":         "notebook",
            "is_private":          True,
            "enable_gpu":          True,
            "enable_internet":     True,
            "dataset_sources":     [dataset_slug],
            "competition_sources": [],
            "kernel_sources":      [],
        }))
        r = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(tmp_path)],
            capture_output=True, text=True, env=env
        )
        _log(f"  Kernel push stdout: {r.stdout.strip()}")
        _log(f"  Kernel push stderr: {r.stderr.strip()}")
        if r.returncode != 0:
            raise RuntimeError(f"Kernel push failed:\n{r.stdout}\n{r.stderr}")

    _log("✓ Kernel submitted — waiting for Kaggle GPU to finish…")
    _log(f"  👉 You can monitor at: https://www.kaggle.com/code/{kernel_id}")

    # ── Step 3: Poll ─────────────────────────────────────────────────────
    max_wait = 7200; poll_wait = 30; elapsed = 0
    while elapsed < max_wait:
        time.sleep(poll_wait)
        elapsed += poll_wait
        r = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            capture_output=True, text=True, env=env
        )
        out = (r.stdout + r.stderr).lower()
        _log(f"  Status ({elapsed//60}m): {r.stdout.strip()}")

        if "403" in out or "forbidden" in out or "permission" in out:
            raise RuntimeError(
                "403 Forbidden — الـ API key مش عنده صلاحية على الـ Kernels.\n"
                "الحل:\n"
                "1. افتح kaggle.com/settings\n"
                "2. API section → Expire API Token → Create New API Token\n"
                "3. حدّث KAGGLE_KEY في الـ .env بالـ key الجديد"
            )
        if "complete" in out: break
        if any(w in out for w in ("error","failed","cancel")):
            raise RuntimeError(f"Kaggle kernel failed:\n{r.stdout}\n{r.stderr}")
    else:
        raise RuntimeError("Training timed out after 2 hours")

    # ── Step 4: Download ─────────────────────────────────────────────────
    _log("📥 Downloading model from Kaggle…")
    out_dir = settings.model_path / f"kaggle_{uid}"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        ["kaggle", "kernels", "output", kernel_id, "-p", str(out_dir)],
        capture_output=True, text=True, env=env
    )
    if r.returncode != 0:
        raise RuntimeError(f"Output download failed:\n{r.stdout}\n{r.stderr}")

    model_file = None
    for pat in ["model.pt","model.joblib","*.pt","*.joblib"]:
        hits = list(out_dir.rglob(pat))
        if hits: model_file = hits[0]; break

    if not model_file:
        raise RuntimeError(f"No model file found. Files: {list(out_dir.rglob('*'))}")

    metrics = {}
    for mf in out_dir.rglob("metrics.json"):
        try: metrics = json.loads(mf.read_text())
        except Exception: pass
        break

    final_path = settings.model_path / f"kaggle_{uid}{model_file.suffix}"
    shutil.move(str(model_file), str(final_path))
    _log(f"✅ Done! {final_path} | Metrics: {metrics}")

    return {
        "best_model_name": metrics.get("best_model", "Kaggle-GPU"),
        "framework":       "pytorch" if final_path.suffix == ".pt" else "sklearn",
        "task_type":       metrics.get("task_type", task),
        "metrics":         metrics,
        "confusion_matrix": None,
        "feature_columns": [],
        "target_column":   target_column,
        "artifact_path":   str(final_path),
        "log":             f"Kaggle GPU kernel: {kernel_id}\nDataset: {dataset_slug}\nMetrics: {json.dumps(metrics,indent=2)}",
    }