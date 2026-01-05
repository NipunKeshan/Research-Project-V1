# =========================
# 0) Config
# =========================
import os
from dataclasses import dataclass

@dataclass
class CFG:
    # Data
    CSV_PATH: str = "light_curves.csv"          # <-- change if needed
    ID_COL: str = "oid"                         # object id column (optional but recommended)
    LABEL_COL: str | None = None                # e.g. "transient_type" (set to None if unlabeled)

    # Columns
    # If FEATURE_COLS is None, we auto-select numeric columns except ID/LABEL.
    FEATURE_COLS: list[str] | None = None
    CATEGORICAL_COLS: list[str] = ("fid",)      # columns treated as categorical
    BINARY_COLS: list[str] = ("isdiffpos",)     # columns treated as binary (-1/1 or 0/1)

    # SimCLR training
    batch_size: int = 512
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-6
    temperature: float = 0.2

    # Augmentations (continuous only)
    noise_std: float = 0.05
    drop_prob: float = 0.10
    scale_jitter: float = 0.05

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Model
    hidden: int = 256
    embed_dim: int = 128
    proj_dim: int = 128
    dropout: float = 0.1

    # Outputs
    OUT_DIR: str = "outputs_simclr_clean"
    MODEL_PATH: str = "simclr_encoder.pt"
    SCALER_PATH: str = "scaler.pkl"
    EMB_PATH: str = "embeddings.npy"

cfg = CFG()

os.makedirs(cfg.OUT_DIR, exist_ok=True)
print("Output dir:", cfg.OUT_DIR)


# =========================
# 1) Imports
# =========================
import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


# =========================
# 2) Load + preprocess data
# =========================
df = pd.read_csv(cfg.CSV_PATH)
print("Raw shape:", df.shape)
print("Columns:", list(df.columns))

# --- basic cleanup ---
df = df.drop_duplicates()
print("After drop_duplicates:", df.shape)

# Decide feature columns
excluded = set([c for c in [cfg.ID_COL, cfg.LABEL_COL] if c is not None])

if cfg.FEATURE_COLS is None:
    # use numeric cols by default (safe for tabular SimCLR)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    FEATURE_COLS = [c for c in num_cols if c not in excluded]
else:
    FEATURE_COLS = list(cfg.FEATURE_COLS)

# Ensure categorical/binary are included if they exist
for c in cfg.CATEGORICAL_COLS:
    if c in df.columns and c not in FEATURE_COLS:
        FEATURE_COLS.append(c)
for c in cfg.BINARY_COLS:
    if c in df.columns and c not in FEATURE_COLS:
        FEATURE_COLS.append(c)

print("Using features (n=%d):" % len(FEATURE_COLS), FEATURE_COLS)

# --- Handle missing ---
# Light-touch: fill numeric NaNs with column median
for c in FEATURE_COLS:
    if c not in df.columns:
        raise ValueError(f"Missing feature column in CSV: {c}")
    if pd.api.types.is_numeric_dtype(df[c]):
        df[c] = df[c].fillna(df[c].median())
    else:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else 0)

# --- Encode categorical (fid) one-hot ---
cat_cols = [c for c in cfg.CATEGORICAL_COLS if c in df.columns]
bin_cols = [c for c in cfg.BINARY_COLS if c in df.columns]

df_proc = df.copy()

# Binary columns to 0/1
for c in bin_cols:
    if pd.api.types.is_numeric_dtype(df_proc[c]):
        # common cases: {-1,1} or {0,1}
        vals = set(pd.unique(df_proc[c].dropna()))
        if vals.issubset({-1, 1}):
            df_proc[c] = (df_proc[c] == 1).astype(int)
        else:
            df_proc[c] = (df_proc[c] > 0).astype(int)
    else:
        df_proc[c] = df_proc[c].astype(str).str.lower().isin(["1","true","t","yes","y","pos","positive"]).astype(int)

# One-hot for categorical columns
if len(cat_cols) > 0:
    df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=False)

# Build final feature matrix
feature_cols_final = [c for c in df_proc.columns if c not in excluded]
# keep only the ones derived from requested features
# (we allow new one-hot columns generated from categorical features)
# Filter rule: include numeric columns and one-hot columns; exclude obviously non-feature columns
if cfg.ID_COL in df_proc.columns:
    feature_cols_final = [c for c in feature_cols_final if c != cfg.ID_COL]
if cfg.LABEL_COL in df_proc.columns and cfg.LABEL_COL is not None:
    feature_cols_final = [c for c in feature_cols_final if c != cfg.LABEL_COL]

X = df_proc[feature_cols_final].to_numpy().astype(np.float32)
print("Processed X shape:", X.shape)

# Labels (optional)
y = None
if cfg.LABEL_COL is not None:
    if cfg.LABEL_COL not in df.columns:
        raise ValueError(f"LABEL_COL='{cfg.LABEL_COL}' not found in CSV.")
    y = df[cfg.LABEL_COL].astype(str).to_numpy()
    print("Labels:", pd.Series(y).value_counts().head())

# Train/val split for SimCLR (labels not required)
idx = np.arange(len(df_proc))
idx_train, idx_val = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X[idx_train])
X_val   = scaler.transform(X[idx_val])

# Save scaler
with open(os.path.join(cfg.OUT_DIR, cfg.SCALER_PATH), "wb") as f:
    pickle.dump({"scaler": scaler, "feature_cols": feature_cols_final}, f)

print("Saved scaler to:", os.path.join(cfg.OUT_DIR, cfg.SCALER_PATH))


# =========================
# 3) SimCLR dataset + augmentations (continuous only)
# =========================
class SimCLRTabularDataset(Dataset):
    def __init__(self, X: np.ndarray, noise_std: float, drop_prob: float, scale_jitter: float):
        self.X = X.astype(np.float32)
        self.noise_std = noise_std
        self.drop_prob = drop_prob
        self.scale_jitter = scale_jitter

    def _augment(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()

        # Gaussian noise
        x += np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)

        # Feature dropout (mask to 0)
        drop_mask = (np.random.rand(*x.shape) > self.drop_prob).astype(np.float32)
        x *= drop_mask

        # Multiplicative jitter
        x *= (1.0 + np.random.normal(0.0, self.scale_jitter, size=x.shape).astype(np.float32))

        return x

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        v1 = self._augment(x)
        v2 = self._augment(x)
        return torch.from_numpy(v1), torch.from_numpy(v2)

train_ds = SimCLRTabularDataset(X_train, cfg.noise_std, cfg.drop_prob, cfg.scale_jitter)
val_ds   = SimCLRTabularDataset(X_val,   cfg.noise_std, cfg.drop_prob, cfg.scale_jitter)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=True)

print("Batches (train/val):", len(train_loader), len(val_loader))


# =========================
# 4) Encoder + projection head
# =========================
class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        return self.net(x)

class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int, proj_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, z):
        return self.net(z)

class SimCLR(nn.Module):
    def __init__(self, in_dim: int, hidden: int, embed_dim: int, proj_dim: int, dropout: float):
        super().__init__()
        self.encoder = MLPEncoder(in_dim, hidden, embed_dim, dropout)
        self.proj = ProjectionHead(embed_dim, proj_dim)

    def forward(self, x):
        h = self.encoder(x)
        z = self.proj(h)
        z = F.normalize(z, dim=1)
        return h, z

in_dim = X_train.shape[1]
model = SimCLR(in_dim, cfg.hidden, cfg.embed_dim, cfg.proj_dim, cfg.dropout).to(DEVICE)
print("Model params:", sum(p.numel() for p in model.parameters()))


# =========================
# 5) NT-Xent loss
# =========================
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """NT-Xent for a batch. z1,z2 are L2-normalized."""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)
    # mask out self similarity
    mask = torch.eye(2 * batch_size, device=sim.device).bool()
    sim.masked_fill_(mask, -1e9)

    # positives: i-th sample in z1 matches i-th in z2
    positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)  # (2B,)

    # denominator: logsumexp over all except self
    loss = -positives + torch.logsumexp(sim, dim=1)
    return loss.mean()

# quick sanity check
x1, x2 = next(iter(train_loader))
with torch.no_grad():
    _, z1 = model(x1.to(DEVICE))
    _, z2 = model(x2.to(DEVICE))
print("Sanity loss:", float(nt_xent_loss(z1, z2, cfg.temperature)))


# =========================
# 6) Train loop (early stopping + collapse check)
# =========================
from tqdm.auto import tqdm

opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

best_val = float("inf")
epochs_no_improve = 0
history = {"train_loss": [], "val_loss": [], "embed_std": []}

def embedding_collapse_std(model: SimCLR, loader: DataLoader, n_batches: int = 5) -> float:
    """If std gets too close to 0, embeddings may have collapsed."""
    model.eval()
    zs = []
    with torch.no_grad():
        for i, (a, b) in enumerate(loader):
            if i >= n_batches:
                break
            _, z = model(a.to(DEVICE))
            zs.append(z.detach().cpu())
    z_all = torch.cat(zs, dim=0)
    return float(z_all.std(dim=0).mean())

for epoch in range(1, cfg.epochs + 1):
    # ---- train ----
    model.train()
    tr_losses = []
    for v1, v2 in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False):
        v1 = v1.to(DEVICE)
        v2 = v2.to(DEVICE)

        _, z1 = model(v1)
        _, z2 = model(v2)
        loss = nt_xent_loss(z1, z2, cfg.temperature)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        tr_losses.append(float(loss.detach().cpu()))

    tr_loss = float(np.mean(tr_losses))

    # ---- val ----
    model.eval()
    val_losses = []
    with torch.no_grad():
        for v1, v2 in val_loader:
            v1 = v1.to(DEVICE)
            v2 = v2.to(DEVICE)
            _, z1 = model(v1)
            _, z2 = model(v2)
            val_losses.append(float(nt_xent_loss(z1, z2, cfg.temperature).detach().cpu()))
    val_loss = float(np.mean(val_losses))

    # collapse check
    e_std = embedding_collapse_std(model, val_loader)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(val_loss)
    history["embed_std"].append(e_std)

    print(f"Epoch {epoch:03d} | train {tr_loss:.4f} | val {val_loss:.4f} | embed_std {e_std:.4f}")

    # early stopping
    if val_loss < best_val - cfg.min_delta:
        best_val = val_loss
        epochs_no_improve = 0
        torch.save(model.encoder.state_dict(), os.path.join(cfg.OUT_DIR, cfg.MODEL_PATH))
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= cfg.patience:
            print(f"Early stopping: no val improvement for {cfg.patience} epochs.")
            break

print("Saved best encoder to:", os.path.join(cfg.OUT_DIR, cfg.MODEL_PATH))


# =========================
# 7) Extract embeddings for ALL samples
# =========================
# Reload best encoder weights (optional but recommended)
encoder = MLPEncoder(in_dim, cfg.hidden, cfg.embed_dim, cfg.dropout).to(DEVICE)
encoder.load_state_dict(torch.load(os.path.join(cfg.OUT_DIR, cfg.MODEL_PATH), map_location=DEVICE))
encoder.eval()

X_all_scaled = scaler.transform(X).astype(np.float32)

with torch.no_grad():
    emb = []
    bs = 2048
    for i in range(0, len(X_all_scaled), bs):
        xb = torch.from_numpy(X_all_scaled[i:i+bs]).to(DEVICE)
        hb = encoder(xb).detach().cpu().numpy()
        emb.append(hb)
emb = np.vstack(emb)

np.save(os.path.join(cfg.OUT_DIR, cfg.EMB_PATH), emb)
print("Embeddings shape:", emb.shape)
print("Saved embeddings to:", os.path.join(cfg.OUT_DIR, cfg.EMB_PATH))


# =========================
# 8) Classifier on embeddings (optional)
# =========================
if y is None:
    print("No labels provided. Set cfg.LABEL_COL to enable classifier training + metrics.")
else:
    # train/test split with labels
    X_tr, X_te, y_tr, y_te = train_test_split(emb, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=2000, n_jobs=None)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)

    # --- metrics ---
    acc = accuracy_score(y_te, y_pred)
    prec_macro = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score(y_te, y_pred, average="macro", zero_division=0)
    f1_macro   = f1_score(y_te, y_pred, average="macro", zero_division=0)

    prec_w = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec_w  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1_w   = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    print("Accuracy:", acc)
    print("Macro   Precision/Recall/F1:", prec_macro, rec_macro, f1_macro)
    print("Weighted Precision/Recall/F1:", prec_w, rec_w, f1_w)

    # Confusion matrix
    labels_sorted = np.unique(y_te)
    cm = confusion_matrix(y_te, y_pred, labels=labels_sorted)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted))

    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, zero_division=0))

    # ROC-AUC for binary only
    if len(labels_sorted) == 2:
        proba = clf.predict_proba(X_te)[:, 1]
        # Need to binarize y for roc_auc_score
        y_bin = (y_te == labels_sorted[1]).astype(int)
        auc = roc_auc_score(y_bin, proba)
        print("ROC-AUC:", auc)

    # Save classifier
    with open(os.path.join(cfg.OUT_DIR, "classifier_logreg.pkl"), "wb") as f:
        pickle.dump({"clf": clf, "classes": clf.classes_}, f)
    print("Saved classifier to:", os.path.join(cfg.OUT_DIR, "classifier_logreg.pkl"))
