import os, copy, time
from os.path import join
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch;
import torch.nn as nn; import torch.nn.functional as F
from torch.optim import Adam
from torch.amp import GradScaler, autocast
from torch.utils.data import (DataLoader,TensorDataset)

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
PICKLE_PATH = 'pickles'; FIGURE_PATH = 'figures'; 
PATH = PICKLE_PATH; PATH_MODELS = PATH
SEQ = 40; INC = 2; CH = 8; CLASSES = 5; VAL_CUTOFF = 332
WORKERS = 4; PRE_FETCH = 2; VERBOSE=True; DEVICE = 'cuda'
UPDATE_EVERY = 50; PRESIST_WORKER = True; PIN_MEMORY = True

EPOCHS = 200; BATCH_SIZE = 4096; DROPOUT = 0.2; PATIENCE = 10
LR_FACTOR = 0.8; LR_PATIENCE = 2; LR_INIT = 1e-3; LR_MIN = 1e-5

COLLECT = 0
NAME = 'test1'
FITTS_PATH = join('fitts_logs', NAME)
DATA_PATH = join('emg_logs', NAME)

# ======== MODELS, TRAINING & DATASETS ========
def count_params(m): 
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ======== DATA LOADER ========
def create_loader(x, y, batch=BATCH_SIZE, shuffle=False, 
                  workers=WORKERS, prefetch_factor=PRE_FETCH,
                  persistent_workers=PRESIST_WORKER):
    return DataLoader(
    TensorDataset(torch.from_numpy(x), 
                  torch.from_numpy(y)),
                #   torch.tensor(x), 
                #   torch.tensor(y)),
    batch_size=batch,
    shuffle=shuffle,
    num_workers=workers,
    prefetch_factor=prefetch_factor if workers > 0 else None,
    persistent_workers=persistent_workers,
    pin_memory=PIN_MEMORY,
    drop_last=False)


# ======== TRAINING & VALIDATING ========
def train(model, train_loader, val_loader, name,
          loss_fn=nn.CrossEntropyLoss(),
          epochs=EPOCHS, lr=LR_INIT, min_lr=LR_MIN,
          lr_factor=LR_FACTOR, lr_patience=LR_PATIENCE, 
          patience=PATIENCE, device=DEVICE, verbose=VERBOSE,
          save_chkp=False, emb_loader=None):

    model.to(device)
    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=lr_factor, patience=lr_patience, min_lr=min_lr)
    scaler = GradScaler(enabled=(device == "cuda"))

    best_val = 1e9
    best_state = {k: v.to("cpu", non_blocking=True).clone() 
                for k, v in model.state_dict().items()}
    wait = 0

    if save_chkp:
        os.makedirs(f"{PICKLE_PATH}/{name}/", exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        total = 0
        step = 0
        pbar = tqdm(total=len(train_loader), desc=f"{name} | Ep {ep}", 
                    leave=True, dynamic_ncols=True, disable=not verbose)

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(xb)
                loss = loss_fn(logits, yb)

            scaler.scale(loss).backward()
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

        val_acc, val_loss = evaluate(model, val_loader, loss_fn, device)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.to("cpu", non_blocking=True).clone() 
                        for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                tqdm.write(f"{name} | Early stop")
                pbar.close()
                break

        pbar.set_postfix(
            loss=f"{total_loss.item() / max(1, len(train_loader)):10.6f}",
            acc=f"{correct.item() / max(1, total):6.4f}",
            val_loss=f"{val_loss:10.6f}",
            val_acc=f"{val_acc:6.4f}",
            LR=f"{opt.param_groups[0]['lr']:8.6f}",
            wait=f"{wait:3.0f}")
        pbar.close()

        if save_chkp:
            checkpoint = {'epoch': ep,
                        'model_state_dict': model.state_dict()}
            torch.save(checkpoint, f"{PICKLE_PATH}/{name}/chkp_{ep:03d}.pt")

    model.load_state_dict(best_state)
    return model


# ---- EMBEDDING PCA CALLBACK ----
class PCA_GPU:
    def __init__(self, dims=2, device=DEVICE):
        self.device = device
        self.dims = dims
        self.mean_ = None
        self.components_ = None

    def fit(self, x):
        if isinstance(x, np.ndarray):
            X = torch.from_numpy(x).to(self.device)
        else:
            X = x.to(self.device)
        N = X.shape[0]
        self.mean_ = X.mean(dim=0, keepdim=True)
        Xc = X - self.mean_
        C = (Xc.T @ Xc) / (N - 1)
        eigvals, eigvecs = torch.linalg.eigh(C)
        idx = torch.argsort(eigvals, descending=True)
        self.components_ = eigvecs[:, idx[:self.dims]]
        return self

    def transform(self, x):
        if isinstance(x, np.ndarray):
            X = torch.from_numpy(x).to(self.device)
        else:
            X = x.to(self.device)
        Xc = X - self.mean_
        Z = Xc @ self.components_
        return Z

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

@torch.no_grad()
def collect_embeddings(model, loader, device):
    model.eval()
    N = len(loader.dataset)
    
    # Robustly infer embedding dimension
    sample_xb, _ = next(iter(loader))
    with autocast(device_type="cuda", enabled=(device == "cuda")):
        sample_emb = model(sample_xb.to(device), return_emb=True)
    D = sample_emb.shape[1]
    
    feats = torch.empty(N, D, device=device)
    labels = torch.empty(N, device=device, dtype=torch.long)
    
    ptr = 0
    for xb, yb in loader:
        b = xb.size(0)
        xb = xb.to(device, non_blocking=True)
        
        # Match training precision for speed
        with autocast(device_type="cuda", enabled=(device == "cuda")):
            emb = model(xb, return_emb=True)
            
        feats[ptr:ptr+b] = emb
        labels[ptr:ptr+b] = yb.to(device, non_blocking=True)
        ptr += b
        
    return feats, labels # Return on GPU for PCA fitting

@torch.no_grad()
def run_pca_sweep(model, loader, name, device=DEVICE):
    checkpoint_dir = f"{PICKLE_PATH}/{name}/"
    output_dir = f"{FIGURE_PATH}/{name}_PCAs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all checkpoint files and sort by epoch
    files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')])
    if not files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    model.to(device)
    model.eval()
    
    # 1. Collect all embeddings for all epochs first (to fit a global PCA for stable video/plots)
    # If memory is an issue, fit PCA only on the last epoch's embeddings
    all_epoch_data = []
    
    for f in files:
        ep_path = os.path.join(checkpoint_dir, f)
        checkpoint = torch.load(ep_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        
        feats, labels = collect_embeddings(model, loader, device)
        all_epoch_data.append({
            "epoch": epoch,
            "feats": feats,  # These are already on CPU from collect_embeddings
            "labels": labels
        })

    # 2. Fit PCA on the final epoch to define the coordinate space
    # (Using the last epoch ensures the most discriminative features define the axes)
    pca = PCA_GPU(dims=2, device=device)
    pca.fit(all_epoch_data[-1]["feats"])

    # 3. Transform and Plot
    for data in all_epoch_data:
        ep = data["epoch"]
        Z = pca.transform(data["feats"]).cpu().numpy()
        y = data["labels"].numpy()

        # Fast plotting using OO API
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100) # Lower DPI for faster saving
        scatter = ax.scatter(Z[:, 0], Z[:, 1], c=y, s=4, cmap="tab10", alpha=0.6)
        ax.set_title(f"{name} | Epoch {ep}")
        
        # Hide axes for cleaner visualization if preferred
        ax.set_xticks([]); ax.set_yticks([])
        
        fig.savefig(f"{output_dir}/ep_{ep:03d}.png", bbox_inches='tight')
        plt.close(fig) # Mandatory memory release

    print(f"PCA sweep completed. Figures saved to {output_dir}")

# ---- VALIDATION ----
@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    # Initialize on GPU
    lsum = torch.tensor(0.0, device=device)
    cor = torch.tensor(0.0, device=device)
    tot = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(xb)
            loss = loss_fn(logits, yb)
        lsum += loss
        cor += (logits.argmax(1) == yb).sum()
        tot += yb.numel()
    return cor.item() / max(1, tot), lsum.item() / max(1, len(loader))


# ======== GENERAL TESTING ========
@torch.no_grad()
def eval_test(model, loaders, metas, name,
              save=True, multi_head=None,
              device=DEVICE):

    model.to(device)
    model.eval()
    results = {}
    os.makedirs(f"{FIGURE_PATH}/{name}/", exist_ok=True)
    os.makedirs(f"{PICKLE_PATH}/{name}/", exist_ok=True)

    def run(loader, meta, tag):
        N = len(loader.dataset)
        # Pre-allocate on GPU to avoid dynamic growth
        preds = torch.empty(N, dtype=torch.long, device=device)
        ptr = 0
        for xb, _ in loader:
            b = xb.size(0)
            xb = xb.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                out = model(xb)
                if multi_head is not None:
                    out = out[multi_head]
            preds[ptr:ptr+b] = out.argmax(1)
            ptr += b
        
        # Single sync point
        preds = preds.cpu().numpy()
        subjects = np.asarray(meta['subjects'])
        labels = np.asarray(meta['classes'])
        unique_subjects = np.unique(subjects)
        n_subj = len(unique_subjects)
        
        acc, act_acc, bal_acc = np.zeros(n_subj), np.zeros(n_subj), np.zeros(n_subj)

        for i, s in enumerate(unique_subjects):
            mask = (subjects == s)
            ps, ls = preds[mask], labels[mask]

            # CA (Classification Accuracy)
            acc[i] = (ps == ls).mean()

            # AER logic (Active Error Rate / Active Accuracy)
            act_mask = (ls != 0)
            if act_mask.any():
                act_acc[i] = (ps[act_mask] == ls[act_mask]).mean()

            # Vectorized Balanced Accuracy
            # Efficiently calculates recall for all classes at once
            cm = confusion_matrix(ls, ps, labels=np.arange(CLASSES))
            with np.errstate(divide='ignore', invalid='ignore'):
                per_class = np.diag(cm) / cm.sum(axis=1)
                bal_acc[i] = np.nanmean(per_class)

        acc, act_acc, bal_acc = acc * 100, act_acc * 100, bal_acc * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 7), dpi=200)
        fig.suptitle(f"{tag} | Mean Acc {acc.mean():.2f} ± {np.std(acc):.2f} \
                     | Mean Actv {act_acc.mean():.2f} ± {np.std(act_acc):.2f}")
        
        ax1.bar(np.arange(n_subj), np.sort(acc))
        ax1.axhline(acc.mean(), color='red', linestyle='--')
        ax1.set_title('Per Subject Accuracy')
        
        ax2.bar(np.arange(n_subj), np.sort(act_acc))
        ax2.axhline(act_acc.mean(), color='red', linestyle='--')
        ax2.set_title('Per Subject Active Accuracy')

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(f"{FIGURE_PATH}/{name}/{tag}.jpg")
        fig.clf()
        plt.close(fig)

        if save:
            np.save(f"{PICKLE_PATH}/{name}/acc_{tag}.npy", acc)
            np.save(f"{PICKLE_PATH}/{name}/aer_{tag}.npy", act_acc)

        return {"acc_mean": acc.mean(), "acc_std": acc.std(),
                "act_acc_mean": act_acc.mean(), "act_acc_std": act_acc.std(),
                "bal_acc_mean": bal_acc.mean(), "bal_acc_std": bal_acc.std()}

    # Iterate through provided loaders (raw, segmented, relabeled)
    for tag in loaders.keys():
        results[tag] = run(loaders[tag], metas[tag], tag)

    # Atomic CSV logging (Fastest concurrent-safe method)
    csv_path = f"{FIGURE_PATH}/results.csv"
    rows = [{"model": name, "set": tag, **r} for tag, r in results.items()]
    df_new = pd.DataFrame(rows)
    df_new.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))

    return results

