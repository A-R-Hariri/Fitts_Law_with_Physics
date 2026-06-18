import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch, torch.nn as nn
import libemg
import numpy as np
import socket, random
from sklearn.utils.class_weight import compute_class_weight

from utils import * 
from models import *


SEED = 13
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
BATCH_SIZE=64

if __name__ == "__main__":
    if not os.path.exists(SGT_PATH) or not len(os.listdir(SGT_PATH)):
        p, smm = libemg.streamers.myo_streamer() 
        odh = libemg.data_handler.OnlineDataHandler(smm)
        
        os.makedirs(SGT_PATH, exist_ok=True)
        args = {'num_reps': TOTAL_REPS, 'rep_time': 3, 'rest_time': 3, 'media_folder': 'images/', 'data_folder': SGT_PATH}
        ui = libemg.gui.GUI(odh, args=args, width=700, height=700)
        ui.download_gestures([1,2,3,4,5], "images/")
        ui.start_gui()

    filters = [libemg.data_handler.RegexFilter(left_bound="C_", right_bound="_R", 
                                               values=["0","1","2","3","4"], description='classes'),
               libemg.data_handler.RegexFilter(left_bound="R_", right_bound="_emg.csv", 
                                               values=[str(r) for r in range(TOTAL_REPS)], description='reps')]
    offline_dh = libemg.data_handler.OfflineDataHandler()
    offline_dh.get_data(folder_location=SGT_PATH, regex_filters=filters, delimiter=',')

    for reps in [1, 5]:
        odh = offline_dh.isolate_data("reps", list(range(reps)), fast=True)
        train_windows, train_meta = odh.parse_windows(SEQ, INC)
        train_meta['classes'] = remap_labels(train_meta['classes'])
        train_loader = create_loader(train_windows, train_meta['classes'], np.zeros_like(train_meta['classes']), 
                                batch=BATCH_SIZE, shuffle=True, 
                                workers=WORKERS, persistent_workers=PRESIST_WORKER)
        
        odh_val = offline_dh.isolate_data("reps", [TOTAL_REPS - 1], fast=True)
        val_windows, val_meta = odh_val.parse_windows(SEQ, INC)
        val_meta['classes'] = remap_labels(val_meta['classes'])
        val_loader = create_loader(val_windows, val_meta['classes'], np.zeros_like(val_meta['classes']),
                                    batch=BATCH_SIZE, shuffle=False, 
                                    workers=WORKERS, persistent_workers=PRESIST_WORKER)

        weights = torch.tensor(compute_class_weight('balanced', 
                            classes=np.arange(CLASSES), 
                                y=train_meta['classes']),
                                dtype=torch.float32,
                                device=DEVICE)


        MODEL = f'within_mhcnn_raw_base-ft-{reps}'
        model = MHCNN()
        model.load_state_dict(torch.load(join(CHECKPOINT_PATH, 
                            "cross_mhcnn_raw_base.pt"))['model_state_dict'])
        train(model=model, name=NAME,
                        loss_fn=BaseLoss(),
                        train_loader=train_loader,
                        val_loader=val_loader,
                        save_chkp=False)
        torch.save({'model_state_dict': model.state_dict()}, 
                   join(SGT_PATH, f"{MODEL}.pt"))
        with open(join(SGT_PATH, "results.txt"), "a") as f:
            print(f"{MODEL}: {evaluate(model, val_loader, BaseLoss())}", file=f)
    

        if reps == 5:
            MODEL = f'within_cnnhcf_raw_base-{reps}'

            train_feats = extract_sub(train_windows, FEAT_LIST, FEATURE_DIC).transpose(0, 2, 1)
            train_feat_loader = create_loader(train_feats, train_meta['classes'], np.zeros_like(train_meta['classes']), 
                                batch=BATCH_SIZE, shuffle=True, 
                                workers=WORKERS, persistent_workers=PRESIST_WORKER)
            val_feats = extract_sub(val_windows, FEAT_LIST, FEATURE_DIC).transpose(0, 2, 1)
            val_feat_loader = create_loader(val_feats, val_meta['classes'], np.zeros_like(val_meta['classes']), 
                                batch=BATCH_SIZE, shuffle=False, 
                                workers=WORKERS, persistent_workers=PRESIST_WORKER)
            n_feat_sub = train_feats.shape[1]

            model = CNN_HCF(n_feat_sub)
            train(model=model, name=MODEL, 
                train_loader=train_feat_loader,
                val_loader=val_feat_loader,
                loss_fn=nn.CrossEntropyLoss(weight=weights),
                save_chkp=False, verbose=VERBOSE)
            torch.save({'model_state_dict': model.state_dict()}, 
                   join(SGT_PATH, f"{MODEL}.pt"))
            with open(join(SGT_PATH, "results.txt"), "a") as f:
                print(f"{MODEL}: {evaluate(model, val_feat_loader, BaseLoss())}", file=f)

            del train_feat_loader, val_feat_loader

        del train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()