"""
Deep Learning Engine — PyTorch-based.

Supports:
  1. Image Classification  — CNN (custom) or pretrained ResNet18/MobileNetV2
  2. Text Classification   — LSTM or DistilBERT (if transformers installed)
  3. Tabular DL            — MLP (Multi-Layer Perceptron) via PyTorch

Input:
  - Image: ZIP file with subfolders per class  e.g. cats/dog.jpg  dogs/cat.jpg
  - Text:  CSV with a text column + label column
  - Tabular: CSV (same as ML engine but uses MLP)

All models saved as .pt files alongside a metadata JSON.
"""
from __future__ import annotations

import io
import json
import logging
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Check available libraries ────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset as TorchDataset, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — DL training unavailable. pip install torch torchvision")

try:
    from torchvision import datasets, models, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════
# 0. SHARED HELPER — rebuild image model architecture
# ══════════════════════════════════════════════════════════════════════════

def get_image_model(arch: str, n_classes: int, img_size: int = 128):
    """
    Rebuild an image classification model from saved metadata.

    Mirrors the exact architecture used in train_image_classification so that
    predict_image_endpoint (and any other caller) can reconstruct the network
    from a checkpoint without duplicating the architecture logic.

    Args:
        arch:      Architecture name: 'resnet18', 'resnet50', 'mobilenet' /
                   'mobilenetv2', or anything else for the custom CNN.
        n_classes: Number of output classes.
        img_size:  Input image size (used only by the custom CNN).

    Returns:
        torch.nn.Module — weights NOT loaded, call load_state_dict() yourself.
    """
    if not TORCH_AVAILABLE or not TORCHVISION_AVAILABLE:
        raise RuntimeError(
            "PyTorch + torchvision required. Run: pip install torch torchvision"
        )

    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    elif arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    elif arch in ("mobilenet", "mobilenetv2"):
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    else:
        # Custom CNN — must match the architecture in train_image_classification exactly
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * (img_size // 8) * (img_size // 8), 256),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    return model


# ══════════════════════════════════════════════════════════════════════════
# 1. TABULAR MLP
# ══════════════════════════════════════════════════════════════════════════

class TabularMLP(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_tabular_dl(
    dataset_path: str | Path,
    target_column: str,
    task_type: str = "auto",
    epochs: int = 30,
    hidden_dims: list[int] | None = None,
) -> dict[str, Any]:
    """Train an MLP on tabular CSV data using PyTorch."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed. Run: pip install torch")

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

    log_buf = io.StringIO()
    t0 = time.time()

    def _log(msg):
        print(msg, file=log_buf)
        logger.info(msg)

    if hidden_dims is None:
        hidden_dims = [256, 128, 64]

    _log(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path) if str(dataset_path).endswith('.csv') else pd.read_excel(dataset_path)
    df = df.dropna(subset=[target_column])
    _log(f"Loaded {len(df)} rows × {df.shape[1]} cols")

    y_raw = df[target_column]
    X_raw = df.drop(columns=[target_column])

    # Drop ID cols
    for c in list(X_raw.columns):
        if X_raw[c].dtype == 'O' and X_raw[c].nunique() == len(X_raw):
            X_raw = X_raw.drop(columns=[c])

    # Encode categoricals
    for c in X_raw.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_raw[c] = le.fit_transform(X_raw[c].astype(str))

    # Impute + scale
    imp = SimpleImputer(strategy='median')
    X_np = imp.fit_transform(X_raw)
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)

    # Detect task
    from app.ml.automl_engine import detect_task_type
    detected = detect_task_type(y_raw) if task_type == 'auto' else task_type

    label_encoder = None
    if detected == 'classification':
        label_encoder = LabelEncoder()
        y_np = label_encoder.fit_transform(y_raw.astype(str))
        n_classes = len(label_encoder.classes_)
        output_dim = n_classes
    else:
        y_np = y_raw.values.astype(np.float32)
        output_dim = 1

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42,
        stratify=y_np if detected == 'classification' else None
    )

    # Tensors
    Xt = torch.FloatTensor(X_train)
    yt = torch.LongTensor(y_train) if detected == 'classification' else torch.FloatTensor(y_train)
    Xv = torch.FloatTensor(X_test)
    yv = torch.LongTensor(y_test) if detected == 'classification' else torch.FloatTensor(y_test)

    train_ds = TensorDataset(Xt, yt)
    loader   = DataLoader(train_ds, batch_size=min(256, len(Xt)), shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _log(f"Device: {device}  |  Task: {detected}  |  Classes: {output_dim}")

    model = TabularMLP(X_np.shape[1], hidden_dims, output_dim).to(device)

    if detected == 'classification':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            if detected == 'regression':
                loss = criterion(out.squeeze(), yb)
            else:
                loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(Xv.to(device))
            if detected == 'regression':
                val_loss = criterion(val_out.squeeze(), yv.to(device)).item()
            else:
                val_loss = criterion(val_out, yv.to(device)).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            _log(f"  Epoch {epoch:3d}/{epochs}  train_loss={total_loss/len(loader):.4f}  val_loss={val_loss:.4f}")

    # Load best weights
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds_raw = model(Xv.to(device))
        if detected == 'classification':
            y_pred = preds_raw.argmax(dim=1).cpu().numpy()
            y_true = yv.numpy()
            avg = 'binary' if n_classes == 2 else 'weighted'
            metrics = {
                'accuracy':  float(accuracy_score(y_true, y_pred)),
                'f1':        float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
                'task_type': 'classification',
                'best_model': 'MLP',
            }
            _log(f"Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}")
        else:
            y_pred = preds_raw.squeeze().cpu().numpy()
            y_true = yv.numpy()
            metrics = {
                'r2':  float(r2_score(y_true, y_pred)),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'task_type': 'regression',
                'best_model': 'MLP',
            }
            _log(f"R²: {metrics['r2']:.4f}  MAE: {metrics['mae']:.4f}")

    # Save
    settings.model_path.mkdir(parents=True, exist_ok=True)
    artifact_name = f"dl_mlp_{uuid.uuid4().hex[:12]}.pt"
    artifact_path = settings.model_path / artifact_name

    meta = {
        'type': 'tabular_mlp',
        'task_type': detected,
        'input_dim': int(X_np.shape[1]),
        'hidden_dims': hidden_dims,
        'output_dim': int(output_dim),
        'feature_columns': list(X_raw.columns),
        'target_column': target_column,
        'classes': list(label_encoder.classes_) if label_encoder else None,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'imputer_stats': imp.statistics_.tolist(),
    }
    torch.save({'model_state': best_state or model.state_dict(), 'meta': meta}, artifact_path)
    _log(f"Saved → {artifact_path}")
    _log(f"Done in {time.time()-t0:.1f}s")

    return {
        'best_model_name': 'MLP (PyTorch)',
        'framework': 'pytorch',
        'task_type': detected,
        'metrics': metrics,
        'confusion_matrix': None,
        'feature_columns': list(X_raw.columns),
        'target_column': target_column,
        'artifact_path': str(artifact_path),
        'log': log_buf.getvalue(),
    }


# ══════════════════════════════════════════════════════════════════════════
# 2. IMAGE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════

def _prepare_image_dataset(data_root: Path, img_size: int, batch_size: int,
                            log_fn) -> tuple:
    """
    Smart dataset builder — handles ANY folder structure automatically.

    Supported layouts
    ─────────────────
    1. Standard ImageFolder  →  root/cat/1.jpg  root/dog/2.jpg
    2. train/val split       →  root/train/cat/  root/val/cat/
    3. train/test split      →  root/train/cat/  root/test/cat/
    4. Single flat folder    →  root/images/*.jpg  (labels from filename prefix  cat_001.jpg)
    5. CSV + images          →  root/labels.csv [filename, label]  root/images/*.jpg
    6. Nested single folder  →  root/images/cat/  root/images/dog/  (unwrap one level)
    """
    import shutil
    import tempfile
    import pandas as pd
    from torch.utils.data import SubsetRandomSampler

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tfm_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def _is_image(p: Path) -> bool:
        return p.suffix.lower() in IMG_EXTS

    def _count_images(folder: Path) -> int:
        return sum(1 for f in folder.rglob('*') if _is_image(f))

    def _make_loaders_from_imagefolder(train_dir, val_dir=None):
        train_ds = datasets.ImageFolder(str(train_dir), transform=tfm_train)
        if val_dir:
            val_ds = datasets.ImageFolder(str(val_dir), transform=tfm_val)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
        else:
            # 80/20 random split
            n    = len(train_ds)
            idxs = list(range(n))
            np.random.shuffle(idxs)
            split = max(1, int(0.8 * n))
            val_ds = datasets.ImageFolder(str(train_dir), transform=tfm_val)
            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      sampler=SubsetRandomSampler(idxs[:split]),  num_workers=0)
            val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                      sampler=SubsetRandomSampler(idxs[split:]), num_workers=0)
        return train_loader, val_loader, train_ds.classes

    subdirs   = [d for d in data_root.iterdir() if d.is_dir()]
    dir_names = {d.name.lower() for d in subdirs}
    csv_files = list(data_root.glob('*.csv'))

    # ── Layout 2/3: train/val or train/test ──────────────────────────────
    if 'train' in dir_names:
        train_dir = data_root / 'train'
        val_dir   = (data_root / 'val')   if (data_root / 'val').exists()  else \
                    (data_root / 'test')  if (data_root / 'test').exists() else None
        log_fn(f"Layout: train/val split  →  train={train_dir}  val={val_dir}")
        # Check train has class subfolders
        train_subdirs = [d for d in train_dir.iterdir() if d.is_dir()]
        if train_subdirs:
            train_loader, val_loader, class_names = _make_loaders_from_imagefolder(train_dir, val_dir)
            log_fn(f"Classes ({len(class_names)}): {class_names}")
            return train_loader, val_loader, class_names

    # ── Layout 1: root has class subfolders with images ──────────────────
    class_subdirs = [d for d in subdirs if _count_images(d) > 0]
    if len(class_subdirs) >= 2:
        log_fn(f"Layout: ImageFolder  →  {len(class_subdirs)} class folders")
        train_loader, val_loader, class_names = _make_loaders_from_imagefolder(data_root)
        log_fn(f"Classes ({len(class_names)}): {class_names}")
        return train_loader, val_loader, class_names

    # ── Layout 6: single wrapper folder (e.g. images/cat/ images/dog/) ───
    if len(subdirs) == 1:
        inner = subdirs[0]
        inner_subdirs = [d for d in inner.iterdir() if d.is_dir()]
        if inner_subdirs and all(_count_images(d) > 0 for d in inner_subdirs):
            log_fn(f"Layout: nested single folder  →  unwrapping {inner.name}/")
            train_loader, val_loader, class_names = _make_loaders_from_imagefolder(inner)
            log_fn(f"Classes ({len(class_names)}): {class_names}")
            return train_loader, val_loader, class_names

    # ── Layout 5: CSV + images folder ────────────────────────────────────
    if csv_files:
        csv_path = csv_files[0]
        log_fn(f"Layout: CSV + images  →  {csv_path.name}")
        df = pd.read_csv(csv_path)

        # Detect filename col and label col
        filename_col = next((c for c in df.columns if 'file' in c.lower() or 'image' in c.lower() or 'name' in c.lower()), df.columns[0])
        label_col    = next((c for c in df.columns if 'label' in c.lower() or 'class' in c.lower() or 'category' in c.lower()), df.columns[-1])
        log_fn(f"CSV cols → filename: '{filename_col}'  label: '{label_col}'")

        # Find images root
        img_root = data_root
        for candidate in ['images', 'imgs', 'data', 'photos']:
            if (data_root / candidate).exists():
                img_root = data_root / candidate
                break

        # Build ImageFolder-compatible structure in a temp dir
        tmp_dir = Path(tempfile.mkdtemp())
        classes = df[label_col].astype(str).unique().tolist()
        for cls in classes:
            (tmp_dir / cls).mkdir(parents=True, exist_ok=True)

        missing = 0
        for _, row in df.iterrows():
            fname = str(row[filename_col])
            label = str(row[label_col])
            src   = img_root / fname
            if not src.exists():
                # Try without subfolder
                src = img_root / Path(fname).name
            if src.exists():
                shutil.copy2(src, tmp_dir / label / src.name)
            else:
                missing += 1

        if missing:
            log_fn(f"Warning: {missing} images not found and skipped")

        log_fn(f"Classes ({len(classes)}): {classes}")
        train_loader, val_loader, class_names = _make_loaders_from_imagefolder(tmp_dir)
        return train_loader, val_loader, class_names

    # ── Layout 4: flat folder — labels from filename prefix ──────────────
    all_images = [f for f in data_root.rglob('*') if _is_image(f)]
    if all_images:
        log_fn(f"Layout: flat folder  →  {len(all_images)} images, extracting labels from filenames")

        # Extract label = everything before first digit or underscore+digit
        import re
        tmp_dir = Path(tempfile.mkdtemp())

        label_map: dict[str, list[Path]] = {}
        for img in all_images:
            stem  = img.stem
            # e.g. cat_001 → cat | dog001 → dog | 001_cat → cat
            match = re.match(r'^([a-zA-Z][a-zA-Z\s\-]*?)[\s_\-]?\d', stem)
            if match:
                label = match.group(1).strip().lower()
            else:
                # fallback: take alphabetic prefix
                label = re.sub(r'[^a-zA-Z]', '', stem).lower() or 'unknown'
            label_map.setdefault(label, []).append(img)

        if len(label_map) < 2:
            raise ValueError(
                "Could not detect multiple classes from filenames. "
                "Please organize images into subfolders named after each class "
                "(e.g. cats/, dogs/) or provide a CSV with labels."
            )

        for label, imgs in label_map.items():
            (tmp_dir / label).mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, tmp_dir / label / img.name)

        log_fn(f"Detected classes: {list(label_map.keys())}")
        train_loader, val_loader, class_names = _make_loaders_from_imagefolder(tmp_dir)
        return train_loader, val_loader, class_names

    raise ValueError(
        "Could not find any images in the uploaded dataset. "
        "Supported formats: subfolders per class, train/val split, CSV + images, "
        "or flat folder with class-prefixed filenames."
    )


def train_image_classification(
    zip_path: str | Path,
    model_arch: str = "resnet18",
    epochs: int = 10,
    img_size: int = 128,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Train image classifier.

    Accepts ANY dataset structure — see _prepare_image_dataset for details.
    """
    if not TORCH_AVAILABLE or not TORCHVISION_AVAILABLE:
        raise RuntimeError(
            "PyTorch + torchvision required. Run: pip install torch torchvision"
        )

    import tempfile
    from sklearn.metrics import accuracy_score, f1_score

    log_buf = io.StringIO()
    t0 = time.time()

    def _log(msg):
        print(msg, file=log_buf)
        logger.info(msg)

    zip_path = Path(zip_path)

    # ── Detect input type and extract ────────────────────────────────────
    tmp_ctx = None

    if zip_path.suffix.lower() == ".json":
        meta      = json.loads(zip_path.read_text())
        data_root = Path(meta["local_path"])
        _log(f"Loading from Kaggle local path: {data_root}")
    elif zip_path.is_dir():
        data_root = zip_path
    elif zip_path.suffix.lower() == ".zip":
        tmp_ctx   = tempfile.TemporaryDirectory()
        data_root = Path(tmp_ctx.name)
        _log(f"Extracting ZIP: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_root)
        # Unwrap __MACOSX and hidden junk
        for junk in data_root.rglob('__MACOSX'):
            import shutil as _sh
            _sh.rmtree(junk, ignore_errors=True)
    else:
        raise ValueError(f"Unsupported input: {zip_path}")

    try:
        _log(f"Data root: {data_root}")

        # ── Smart dataset detection ──────────────────────────────────────
        train_loader, val_loader, class_names = _prepare_image_dataset(
            data_root, img_size, batch_size, _log
        )

        n_classes = len(class_names)
        _log(f"Total classes: {n_classes}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _log(f"Device: {device}  Arch: {model_arch}  Epochs: {epochs}")

        # ── Build model ──────────────────────────────────────────────────
        if model_arch == 'resnet18':
            model = models.resnet18(weights='DEFAULT')
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        elif model_arch == 'resnet50':
            model = models.resnet50(weights='DEFAULT')
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        elif model_arch == 'mobilenet':
            model = models.mobilenet_v2(weights='DEFAULT')
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
        else:
            model = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128 * (img_size // 8) * (img_size // 8), 256),
                nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(256, n_classes),
            )

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        best_acc, best_state = 0.0, None

        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                nn.CrossEntropyLoss()(model(xb), yb).backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    pred = model(xb.to(device)).argmax(1).cpu()
                    correct += (pred == yb).sum().item()
                    total   += len(yb)
            acc = correct / total if total else 0
            _log(f"  Epoch {epoch:2d}/{epochs}  val_acc={acc:.4f}")
            if acc > best_acc:
                best_acc   = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                all_pred.extend(model(xb.to(device)).argmax(1).cpu().numpy())
                all_true.extend(yb.numpy())

        avg = 'binary' if n_classes == 2 else 'weighted'
        acc = float(accuracy_score(all_true, all_pred))
        f1  = float(f1_score(all_true, all_pred, average=avg, zero_division=0))
        metrics = {
            'accuracy': acc,
            'f1': f1,
            'task_type': 'image_classification',
            'best_model': model_arch,
            'n_classes': n_classes,
        }
        _log(f"Best val_acc={acc:.4f}  F1={f1:.4f}")

        settings.model_path.mkdir(parents=True, exist_ok=True)
        artifact_name = f"dl_img_{uuid.uuid4().hex[:12]}.pt"
        artifact_path = settings.model_path / artifact_name
        torch.save(
            {
                'model_state': best_state or model.state_dict(),
                'meta': {
                    'type':        'image_classification',
                    'arch':        model_arch,
                    'n_classes':   n_classes,
                    'class_names': class_names,
                    'img_size':    img_size,
                },
            },
            artifact_path,
        )
        _log(f"Saved → {artifact_path}")
        _log(f"Done in {time.time()-t0:.1f}s")

    finally:
        if tmp_ctx:
            tmp_ctx.cleanup()

    return {
        'best_model_name': f'CNN-{model_arch}',
        'framework': 'pytorch',
        'task_type': 'image_classification',
        'metrics': metrics,
        'confusion_matrix': None,
        'feature_columns': [],
        'target_column': 'image_class',
        'artifact_path': str(artifact_path),
        'log': log_buf.getvalue(),
    }


# ══════════════════════════════════════════════════════════════════════════
# 3. TEXT CLASSIFICATION (LSTM)
# ══════════════════════════════════════════════════════════════════════════

class TextLSTM(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, n_layers=2, dropout=0.3):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm    = nn.LSTM(embed_dim, hidden_dim, n_layers,
                               batch_first=True, dropout=dropout, bidirectional=True)
        self.fc      = nn.Linear(hidden_dim * 2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        emb = self.dropout(self.embed(x))
        out, (h, _) = self.lstm(emb)
        h = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(self.dropout(h))


def train_text_classification(
    dataset_path: str | Path,
    text_column: str,
    label_column: str,
    epochs: int = 10,
    max_len: int = 128,
    vocab_size: int = 10000,
) -> dict[str, Any]:
    """Train LSTM text classifier from CSV."""

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed. Run: pip install torch")

    import pandas as pd
    import re
    from collections import Counter
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score

    log_buf = io.StringIO()
    t0 = time.time()

    def _log(msg):
        print(msg, file=log_buf)
        logger.info(msg)

    df = pd.read_csv(dataset_path).dropna(subset=[text_column, label_column])
    _log(f"Loaded {len(df)} rows  text_col={text_column}  label_col={label_column}")

    def tokenize(text):
        text = re.sub(r'[^a-zA-Z\u0600-\u06FF\s]', ' ', str(text).lower())
        return text.split()

    # Build vocab
    counter = Counter()
    all_tokens = df[text_column].apply(tokenize)
    for tokens in all_tokens:
        counter.update(tokens)

    vocab = {tok: idx + 2 for idx, (tok, _) in enumerate(counter.most_common(vocab_size - 2))}
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1

    def encode(text):
        tokens = tokenize(text)[:max_len]
        ids    = [vocab.get(t, 1) for t in tokens]
        ids   += [0] * (max_len - len(ids))
        return ids

    X = np.array([encode(t) for t in df[text_column]])
    le = LabelEncoder()
    y  = le.fit_transform(df[label_column].astype(str))
    n_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xt = torch.LongTensor(X_train)
    yt = torch.LongTensor(y_train)
    Xv = torch.LongTensor(X_test)
    yv = torch.LongTensor(y_test)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _log(f"Device: {device}  Classes: {n_classes}  Vocab: {len(vocab)}")

    model     = TextLSTM(len(vocab), 128, 128, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(Xv.to(device)).argmax(1).cpu().numpy()
        acc = float(accuracy_score(yv.numpy(), pred))
        _log(f"  Epoch {epoch:2d}/{epochs}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(Xv.to(device)).argmax(1).cpu().numpy()

    avg = 'binary' if n_classes == 2 else 'weighted'
    metrics = {
        'accuracy':   float(accuracy_score(yv.numpy(), y_pred)),
        'f1':         float(f1_score(yv.numpy(), y_pred, average=avg, zero_division=0)),
        'task_type':  'text_classification',
        'best_model': 'BiLSTM',
    }
    _log(f"Accuracy: {metrics['accuracy']:.4f}  F1: {metrics['f1']:.4f}")

    settings.model_path.mkdir(parents=True, exist_ok=True)
    artifact_name = f"dl_lstm_{uuid.uuid4().hex[:12]}.pt"
    artifact_path = settings.model_path / artifact_name
    meta = {
        'type':         'text_classification',
        'vocab':        vocab,
        'max_len':      max_len,
        'n_classes':    n_classes,
        'class_names':  list(le.classes_),
        'text_column':  text_column,
        'label_column': label_column,
        'embed_dim':    128,
        'hidden_dim':   128,
    }
    torch.save({'model_state': best_state or model.state_dict(), 'meta': meta}, artifact_path)
    _log(f"Saved → {artifact_path}")
    _log(f"Done in {time.time()-t0:.1f}s")

    return {
        'best_model_name': 'BiLSTM (PyTorch)',
        'framework':       'pytorch',
        'task_type':       'text_classification',
        'metrics':         metrics,
        'confusion_matrix': None,
        'feature_columns': [text_column],
        'target_column':   label_column,
        'artifact_path':   str(artifact_path),
        'log':             log_buf.getvalue(),
    }