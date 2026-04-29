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
WORKERS = 8; PRE_FETCH = 2; VERBOSE=True; DEVICE = 'cuda'
UPDATE_EVERY = 1; PRESIST_WORKER = True; PIN_MEMORY = True

EPOCHS = 100; BATCH_SIZE = 512; DROPOUT = 0.2; PATIENCE = 5
LR_FACTOR = 0.6; LR_PATIENCE = 4; LR_INIT = 1e-4; LR_MIN = 1e-5

NAME = 'Test'
FITTS_PATH = join('fitts_logs', NAME)
DATA_PATH = join('emg_logs', NAME)
SGT_PATH = join('user_sgt', NAME)

SAMPLING_RATE = 200
FEATURE_LIST = ['WENG']
FEATURE_DIC = {'WENG_fs': SAMPLING_RATE}

PARAMS = {
        'frame_rate': 60, 'mode': 'B', 'hold_frames_required': 30,
        'target_timeout_frames': 420, 'max_targets': 12,
        'target_radius_list': [20,10], 'target_distance_range': [200, 400],
        'ring_radius_list': [300,450], 'screen_size': (1690, 980),
        'physics': {'enabled': False, 'mass': 5, 
                    'max_acceleration': 0.08, 'damping': 1.0},
        'c_vel': 1, 'use_test_input': False, 'snap_back' : False,
        }

# ======== MODELS, TRAINING & DATASETS ========
def count_params(m): 
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

mapping = {0: 1, 1: 4, 2: 0, 3: 3, 4: 2}
def remap_labels(labels, mapping=mapping):
    return np.array([mapping[x] for x in labels])

# ======== DATA LOADER ========
def create_loader(x, y, batch=BATCH_SIZE, shuffle=False, 
                  workers=WORKERS, prefetch_factor=PRE_FETCH,
                  persistent_workers=PRESIST_WORKER):
    return DataLoader(
    TensorDataset(torch.from_numpy(x.astype(DTYPE)), 
                  torch.from_numpy(y.astype(np.int64))),
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

    # if save_chkp:
    #     os.makedirs(f"{CHECKPOINT_PATH}", exist_ok=True)
    #     os.makedirs(f"{CHECKPOINT_PATH}/{name}/", exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0.0, device=device)
        total = 0
        step = 0
        pbar = tqdm(total=len(train_loader), desc=f"{name} | Ep {ep}", 
                    leave=False, dynamic_ncols=True, disable=not verbose)

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device=="cuda")):
                if return_emb and return_logits:
                    emb, logits = model(xb, return_emb, return_logits)
                    loss = loss_fn(emb, logits, yb)
                else:
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

        val_acc, val_loss = evaluate(model, val_loader, loss_fn, 
                                     return_emb, return_logits, device)
        sch.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            wait = 0
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
            LR=f"{opt.param_groups[0]['lr']:8.6f}",
            wait=f"{wait:3.0f}")
        pbar.close()

        # if save_chkp:
        #     checkpoint = {'epoch': ep,
        #                 'model_state_dict': model.state_dict()}
        #     torch.save(checkpoint, f"{CHECKPOINT_PATH}/{name}/chkp_{ep:03d}.pt")

    model.load_state_dict(best_state)
    return model


# ---- VALIDATION ----
@torch.no_grad()
def evaluate(model, loader, 
             loss_fn=nn.CrossEntropyLoss(), 
             return_emb=False, return_logits=False,
             device='cuda'):
    model.eval()
    # Initialize on GPU
    lsum = torch.tensor(0.0, device=device)
    cor = torch.tensor(0.0, device=device)
    tot = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
            if return_emb and return_logits:
                emb, logits = model(xb, return_emb, return_logits)
                loss = loss_fn(emb, logits, yb)
            else:
                logits = model(xb)
                loss = loss_fn(logits, yb)  
        lsum += loss.detach()
        cor += (logits.argmax(1) == yb).sum()
        tot += yb.numel()
    return cor.item() / max(1, tot), lsum.item() / max(1, len(loader))
