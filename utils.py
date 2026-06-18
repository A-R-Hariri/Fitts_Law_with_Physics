import os, copy, time, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

import torch;
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn; import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import GradScaler, autocast
from torch.utils.data import (DataLoader, TensorDataset, Sampler)
from sklearn.preprocessing import StandardScaler
from libemg.feature_extractor import FeatureExtractor
from torch.nn.utils import clip_grad_norm_


NAME = 'test'


def is_notebook():
    try:
        from IPython import get_ipython; shell = get_ipython()
        if shell is None: return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except: return False

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

DTYPE = np.float32
PICKLE_PATH = 'checkpoints'; FIGURE_PATH = 'figures'; 
CHECKPOINT_PATH = PICKLE_PATH; 
SEQ = 40; INC = 2; CH = 8; CLASSES = 5; VAL_CUTOFF = 332
WORKERS = 8; PRE_FETCH = 2; VERBOSE=True; DEVICE = 'cuda'
UPDATE_EVERY = 2; PRESIST_WORKER = True; PIN_MEMORY = True

_GESTURE_LABELS = {0: "NM", 1: "HC", 2: "FX", 3: "EX", 4: "HO"}

EPOCHS = 300; BATCH_SIZE = 128; DROPOUT = 0.2; PATIENCE = 15
LR_FACTOR = 0.6; LR_PATIENCE = 7; LR_INIT = 5e-4; LR_MIN = 1e-6

RN_PRIOR_WEIGHT = 0
N_SUB     = 4      

N_SUBJECTS = 306; MARGIN = 0.5; W_HARD = 1.0; W_SOFT = 0.0
ALPHA_START = 0.01; ALPHA_END = 0.25; WARMUP = 25
TAU = float('inf')

FITTS_PATH = os.path.join('fitts_logs', NAME)
DATA_PATH = os.path.join('emg_logs', NAME)
SGT_PATH = os.path.join('user_sgt', NAME)

TOTAL_REPS = 6

FEAT_LIST = ['WENG'] 
SAMPLING_RATE = 200
FEATURE_DIC = {'WENG_fs': SAMPLING_RATE}

PARAMS = {
        'frame_rate': 60, 'mode': 'B', 'hold_frames_required': 45,
        'target_timeout_frames': 480, 'max_targets': 12,
        'target_radius_list': [30, 21, 12], 'target_distance_range': [200, 400],
        'ring_radius_list': [300, 375, 450], 'screen_size': (1690, 980),
        'physics': {'enabled': False, 'mass': 5, 
                    'max_acceleration': 0.08, 'damping': 1.0},
        'c_vel': 1, 'use_test_input': False, 'snap_back' : False,
        }


mapping = {0: 1, 1: 4, 2: 0, 3: 3, 4: 2}
def remap_labels(labels, mapping=mapping):
    return np.array([mapping[x] for x in labels])

# ======== MODELS, TRAINING & DATASETS ========
def count_params(m): 
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ======== DATA LOADER ========
def create_loader(x, y, s, batch=BATCH_SIZE, shuffle=False, 
                  workers=WORKERS, prefetch_factor=PRE_FETCH,
                  persistent_workers=PRESIST_WORKER,
                  pin_memory=PIN_MEMORY):
    return DataLoader(
            TensorDataset(torch.from_numpy(x.astype(DTYPE)), 
                            torch.from_numpy(y.astype(np.int64)),
                            torch.from_numpy(s.astype(np.int64))),
            batch_size=batch,
            shuffle=shuffle,
            num_workers=workers,
            prefetch_factor=prefetch_factor if workers > 0 else None,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            drop_last=False)


@torch.no_grad()
def eval_within(model, loader, meta,
                multi_head=None,
                device=DEVICE):

    model.to(device)
    model.eval()
    results = {}

    def run(loader, meta):
        N = len(loader.dataset)
        # Pre-allocate on GPU to avoid dynamic growth
        preds = torch.empty(N, dtype=torch.long, device=device)
        ptr = 0
        for xb, *_ in loader:
            b = xb.size(0)
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
                out = model(xb)
                if multi_head is not None:
                    out = out[multi_head]
            preds[ptr:ptr+b] = out.argmax(1)
            ptr += b
        
        # Single sync point
        preds = preds.cpu().numpy()
        labels = np.asarray(meta['classes'])
        
        ps, ls = preds, labels

        f1 = f1_score(ls, ps, average='macro')

        # CA (Classification Accuracy)
        acc = (ps == ls).mean()

        # AER logic (Active Error Rate / Active Accuracy)
        act_mask = (ls != 0)
        if act_mask.any():
            act_acc = (ps[act_mask] == ls[act_mask]).mean()

        # Vectorized Balanced Accuracy
        # Efficiently calculates recall for all classes at once
        cm = confusion_matrix(ls, ps, labels=np.arange(CLASSES))
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class = np.diag(cm) / cm.sum(axis=1)
            bal_acc = np.nanmean(per_class)

        acc, act_acc, bal_acc = acc * 100, act_acc * 100, bal_acc * 100

        return {"acc_mean": acc.mean(),
                "act_acc_mean": act_acc.mean(),
                "bal_acc_mean": bal_acc.mean(),
                "F1": f1}

    # Iterate through provided loaders (raw, segmented, relabeled)
    results = run(loader, meta)

    return results


@torch.no_grad()
def eval_within_lda(model, x, meta):

    results = {}

    def run(x, meta):
        preds = model.predict(x)
        labels = np.asarray(meta['classes'])
        ps, ls = preds, labels

        # CA (Classification Accuracy)
        acc = (ps == ls).mean()

        # AER logic (Active Error Rate / Active Accuracy)
        act_mask = (ls != 0)
        if act_mask.any():
            act_acc = (ps[act_mask] == ls[act_mask]).mean()

        # Vectorized Balanced Accuracy
        # Efficiently calculates recall for all classes at once
        cm = confusion_matrix(ls, ps, labels=np.arange(CLASSES))
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class = np.diag(cm) / cm.sum(axis=1)
            bal_acc = np.nanmean(per_class)

        acc, act_acc, bal_acc = acc * 100, act_acc * 100, bal_acc * 100

        return {"acc_mean": acc.mean(),
                "act_acc_mean": act_acc.mean(),
                "bal_acc_mean": bal_acc.mean()}

    # Iterate through provided loaders (raw, segmented, relabeled)
    results = run(x, meta)

    return results


def extract_full(windows, feat_list, feat_dic):
    """Full window features. Returns (N, CH*n_feats) → (N, F)"""
    fe = FeatureExtractor()
    return fe.extract_features(feat_list, windows, array=True,
                               fix_feature_errors=True,
                               feature_dic=feat_dic).reshape(windows.shape[0], -1)

def extract_sub(windows, feat_list, feat_dic, n_sub=N_SUB):
    """
    Split each (N, CH, T) window into n_sub sub-windows of (N, CH, T//n_sub),
    extract features on each, return (N, n_sub, F).
    """
    fe = FeatureExtractor()
    N, CH, T = windows.shape
    assert T % n_sub == 0, f"T={T} not divisible by n_sub={n_sub}"
    sub_len = T // n_sub
    # reshape to (N*n_sub, CH, sub_len) so libemg processes all at once
    subs = windows.reshape(N * n_sub, CH, sub_len)
    feats = fe.extract_features(feat_list, subs, array=True,
                                fix_feature_errors=True,
                                feature_dic=feat_dic)
    feats = feats.reshape(N * n_sub, -1)  # (N*n_sub, F)
    return feats.reshape(N, n_sub, -1)    # (N, n_sub, F)


def normalize_features(tr, va, te):
    """
    Fit StandardScaler on training data only.
    Apply to val and test without leaking test statistics.

    Works for both:
      full window : (N, F)
      sub-windowed: (N, N_SUB, F) — reshape, scale, reshape back
    """
    shape_tr = tr.shape
    shape_va = va.shape
    shape_te = te.shape

    # flatten to 2D for scaler: (N, F) or (N*N_SUB, F)
    tr_2d = tr.reshape(-1, shape_tr[-1])
    va_2d = va.reshape(-1, shape_va[-1])
    te_2d = te.reshape(-1, shape_te[-1])

    scaler = StandardScaler()
    tr_2d  = scaler.fit_transform(tr_2d)   # fit + transform on train
    va_2d  = scaler.transform(va_2d)       # transform only
    te_2d  = scaler.transform(te_2d)       # transform only

    return (np.nan_to_num(tr_2d.reshape(shape_tr), nan=0.0, 
            posinf=0.0, neginf=0.0).astype(np.float32),
            np.nan_to_num(va_2d.reshape(shape_va), nan=0.0, 
            posinf=0.0, neginf=0.0).astype(np.float32),
            np.nan_to_num(te_2d.reshape(shape_te), nan=0.0, 
            posinf=0.0, neginf=0.0).astype(np.float32))


def population_channel_stats(windows, batch=200_000):
    """Compute raw per-channel mean and std. Pass raw (non-normalized) windows."""
    N, C, T = windows.shape
    s1 = np.zeros(C, np.float64)
    s2 = np.zeros(C, np.float64)
    count = 0
    for i in range(0, N, batch):
        c = np.asarray(windows[i:i+batch], dtype=np.float64)
        s1    += c.sum(axis=(0, 2))
        s2    += (c * c).sum(axis=(0, 2))
        count += c.shape[0] * T
    mean = (s1 / count).astype(np.float32)
    std  = np.sqrt(np.clip(s2 / count - (s1/count)**2, 0, None)).astype(np.float32)
    return mean, std


def normalize_per_user(windows, subjects, eps=1e-6):
    out = np.array(windows, dtype=np.float32)  # materialize mmap into writable copy
    
    for s in np.unique(subjects):
        mask = subjects == s
        u = out[mask]                                          # (N_s, C, T)
        mean = u.mean(axis=(0, 2), keepdims=True)              # (1, C, 1)
        std  = u.std(axis=(0, 2), keepdims=True)               # (1, C, 1)
        out[mask] = (u - mean) / (std + eps)
    
    return out

class RunningNorm(nn.Module):
    def __init__(self, num_channels, tau,
                 init_mean=None, init_std=None,
                 eps=1e-6, prior_weight=RN_PRIOR_WEIGHT):   
        """
        tau: time constant in windows.
             tau=inf  → exact cumulative mean (session ceiling).
             tau=N    → EMA that closes 63% of a step change in N windows.
        alpha per-window = 1 - exp(-1/tau), batch-size invariant by construction.
        """
        super().__init__()
        self.tau = tau
        self.eps = eps

        im  = torch.zeros(num_channels) if init_mean is None \
              else torch.as_tensor(init_mean, dtype=torch.float32)
        ist = torch.ones(num_channels)  if init_std  is None \
              else torch.as_tensor(init_std,  dtype=torch.float32)
        im  = im.view(1, -1, 1)
        ist = ist.view(1, -1, 1)

        self.register_buffer("init_mean",    im.clone())
        self.register_buffer("init_sq",      (ist**2 + im**2).clone())
        self.register_buffer("running_mean", im.clone())
        self.register_buffer("running_sq",   (ist**2 + im**2).clone())
        self.register_buffer("n_updates", torch.tensor([float(prior_weight)], dtype=torch.float32))
        self.prior_weight = float(prior_weight)

    @torch.no_grad()
    def _update(self, x):
        B = x.size(0)
        n = self.n_updates.item()
        if self.tau == float('inf'):
            a = B / (n + B)
        else:
            a = 1.0 - math.exp(-B / self.tau)   # simplified: equivalent to 1-(1-alpha)^B

        self.running_mean.mul_(1 - a).add_(
            x.mean(dim=(0, 2), keepdim=True), alpha=a)
        self.running_sq.mul_(1 - a).add_(
            (x * x).mean(dim=(0, 2), keepdim=True), alpha=a)
        self.n_updates.add_(B)

    def forward(self, x):
        if self.training:
            return x
        xf = x.float()
        self._update(xf)
        var = (self.running_sq - self.running_mean ** 2).clamp_min(0.0)
        return ((xf - self.running_mean) / torch.sqrt(var + self.eps) * 128.0).to(x.dtype)

    def reset(self):
        self.running_mean.copy_(self.init_mean)
        self.running_sq.copy_(self.init_sq)
        self.n_updates.fill_(self.prior_weight) 



# ======== TRAINING & VALIDATING ========
def train(model, train_loader, val_loader, name,
          loss_fn=nn.CrossEntropyLoss(),
          return_emb=False, return_logits=False,
          epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN,
          lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, 
          patience=PATIENCE, device=DEVICE,
          verbose=VERBOSE, save_chkp=False):

    model.to(device)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr)
    scaler = GradScaler(enabled=(device=="cuda"))

    best_val = 1e9
    best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    wait = 0
    best_epoch = 0

    if save_chkp:
        os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        total = 0
        step = 0
        pbar = tqdm(total=len(train_loader), desc=f"{name} | Ep {ep}", 
                    leave=True, dynamic_ncols=True, disable=not verbose)

        for xb, yb, *_ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                if return_emb and return_logits:
                    emb, logits = model(xb, return_emb, return_logits)
                    loss = loss_fn(emb, logits, yb)
                elif return_emb:
                    emb, logits = model(xb, return_emb, return_logits)
                    loss = loss_fn(emb, yb)
                else:
                    logits = model(xb)
                    loss = loss_fn(logits, yb)                    

            scaler.scale(loss).backward()
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            total_loss += loss.detach()
            correct += (logits.argmax(1) == yb).sum()
            total += yb.numel()
            step += 1

            if not(step % UPDATE_EVERY):
                pbar.update(UPDATE_EVERY)
                pbar.set_postfix(
                    loss=f"{total_loss.item() / step:10.8f}",
                    acc=f"{correct.item() / max(1, total):6.4f}",
                    LR=f"{opt.param_groups[0]['lr']:8.6f}")

        if step % UPDATE_EVERY:
            pbar.update(step % UPDATE_EVERY)

        val_acc, val_loss, val_bal, val_conf = evaluate(model, val_loader, loss_fn, 
                                            return_emb, return_logits, device)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
            best_epoch = ep
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    tqdm.write(f"{name} | Early stop")
                pbar.close()
                break

        pbar.set_postfix(
            loss=f"{total_loss.item() / max(1, len(train_loader)):10.6f}",
            acc=f"{correct.item() / max(1, total):6.4f}",
            val_loss=f"{val_loss:10.6f}",
            val_acc=f"{val_acc:6.4f}",
            val_bal = f"{val_bal:6.4f}", 
            LR=f"{opt.param_groups[0]['lr']:8.6f}",
            wait=f"{wait:3.0f}")
        pbar.close()

        if verbose:
            val_conf_norm = val_conf / val_conf.sum(dim=1, keepdim=True).clamp(min=1.0)
            for i, row in enumerate(val_conf_norm):
                print(f"  c{i}: [" + ", ".join([f"{v.item():.2f}" for v in row]) +
                        f"]  recall={row[i].item():.2f}")
            
        if save_chkp:
            checkpoint = {'epoch': ep,
                        'model_state_dict': model.state_dict()}
            torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/chkp_{ep:03d}.pt")

    model.load_state_dict(best_state)
    checkpoint = {'epoch': best_epoch,
                'model_state_dict': model.state_dict()}
    if save_chkp:
        torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/{name}.pt")

    return model


# ---- VALIDATION ----
@torch.no_grad()
def evaluate(model, loader, loss_fn, 
             return_emb=True, return_logits=False, device='cuda'):
    model.eval()
    # Initialize on GPU
    lsum = torch.tensor(0.0, device=device)
    cor = torch.tensor(0.0, device=device)
    tot = torch.tensor(0,   device=device, dtype=torch.long)
    class_cor   = torch.zeros(CLASSES, device=device)   # per-class correct
    class_tot   = torch.zeros(CLASSES, device=device)   # per-class total
    val_conf_matrix = torch.zeros((CLASSES, CLASSES), device=device)

    for xb, yb, *_ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
            if return_emb and return_logits:
                emb, logits = model(xb, return_emb, return_logits)
                loss = loss_fn(emb, logits, yb)
            else:
                logits = model(xb)
                loss = loss_fn(logits, yb)  

        preds = logits.argmax(1)
        lsum += loss.detach()
        cor += (preds == yb).sum()
        tot += yb.numel()
        hits = (preds == yb).float()
        class_cor.scatter_add_(0, yb, hits)
        class_tot.scatter_add_(0, yb, torch.ones_like(hits))
        idx = (yb * CLASSES + preds).clamp(0, CLASSES * CLASSES - 1)
        batch_counts = torch.bincount(idx, minlength=CLASSES * CLASSES).float().view(CLASSES, CLASSES)
        val_conf_matrix += batch_counts
        mask = class_tot > 0
        bal_acc = (class_cor[mask] / class_tot[mask]).mean().item()

    return (cor.item() / max(1, tot), 
            lsum.item() / max(1, len(loader)),
            bal_acc, val_conf_matrix)