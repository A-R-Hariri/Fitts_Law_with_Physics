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
from models import CNN, MLP


SEED = 13
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
BATCH_SIZE=128


if __name__ == "__main__":
    if not os.path.exists(SGT_PATH) or not os.listdir(SGT_PATH):
        p, smm = libemg.streamers.myo_streamer() 
        odh = libemg.data_handler.OnlineDataHandler(smm)
        
        os.mkdir(SGT_PATH)
        args = {'num_reps': 15, 'rep_time': 3, 'rest_time': 2, 'media_folder': 'images/', 'data_folder': SGT_PATH}
        ui = libemg.gui.GUI(odh, args=args, width=700, height=700)
        ui.download_gestures([1,2,3,4,5], "images/")
        ui.start_gui()

    filters = [libemg.data_handler.RegexFilter(left_bound="C_", right_bound="_R", 
                                               values=["0","1","2","3","4"], description='classes'),
               libemg.data_handler.RegexFilter(left_bound="R_", right_bound="_emg.csv", 
                                               values=[str(r) for r in range(15)], description='reps')]
    offline_dh = libemg.data_handler.OfflineDataHandler()
    offline_dh.get_data(folder_location=SGT_PATH, regex_filters=filters, delimiter=',')

    fe = libemg.feature_extractor.FeatureExtractor()

    for reps in [2, 5, 13]:
        odh = offline_dh.isolate_data("reps", list(range(reps)), fast=True)
        train_windows, train_meta = odh.parse_windows(SEQ, INC)
        train_meta['classes'] = remap_labels(train_meta['classes'])
        train_loader = create_loader(train_windows, train_meta['classes'], 
                                batch=BATCH_SIZE, shuffle=True, 
                                workers=WORKERS, persistent_workers=PRESIST_WORKER)
        train_feats = fe.extract_features(FEATURE_LIST, train_windows, array=True,
                                        fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
                                            train_windows.shape[0], -1))
        train_feat_loader = create_loader(train_feats, train_meta['classes'], 
                            batch=BATCH_SIZE, shuffle=True, 
                            workers=WORKERS, persistent_workers=PRESIST_WORKER)
        
        odh_val = offline_dh.isolate_data("reps", [13], fast=True)
        val_windows, val_meta = odh_val.parse_windows(SEQ, INC)
        val_meta['classes'] = remap_labels(val_meta['classes'])
        val_loader = create_loader(val_windows, val_meta['classes'], 
                                    batch=BATCH_SIZE, shuffle=False, 
                                    workers=WORKERS, persistent_workers=PRESIST_WORKER)
        val_feats = fe.extract_features(FEATURE_LIST, val_windows, array=True,
                                        fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
                                            val_windows.shape[0], -1))
        val_feat_loader = create_loader(val_feats, val_meta['classes'], 
                            batch=BATCH_SIZE, shuffle=False, 
                            workers=WORKERS, persistent_workers=PRESIST_WORKER)
        
        odh_test = offline_dh.isolate_data("reps", [14], fast=True)
        test_windows, test_meta = odh_test.parse_windows(SEQ, INC)
        test_meta['classes'] = remap_labels(test_meta['classes'])
        test_loader = create_loader(test_windows, test_meta['classes'], 
                                    batch=BATCH_SIZE, shuffle=False, 
                                    workers=0, persistent_workers=False)
        test_feats = fe.extract_features(FEATURE_LIST, test_windows, array=True,
                                        fix_feature_errors=False, feature_dic=FEATURE_DIC).reshape((
                                            test_windows.shape[0], -1))
        test_feat_loader = create_loader(test_feats, test_meta['classes'], 
                            batch=BATCH_SIZE, shuffle=False, 
                            workers=0, persistent_workers=False)
        
        n_features = train_feats.shape[1]

        weights = torch.tensor(compute_class_weight('balanced', 
                            classes=np.arange(CLASSES), 
                                y=train_meta['classes']),
                                dtype=torch.float32,
                                device=DEVICE)

        MODEL = f'cnn_within_ft_raw_{reps}'
        model = CNN()
        # model.load_state_dict(torch.load(join(PATH, "cnn_raw.pt")))
        train(model=model, name=MODEL, 
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=nn.CrossEntropyLoss(weight=weights),
            save_chkp=False, verbose=VERBOSE)
        torch.save(model.state_dict(), join(SGT_PATH, f"{MODEL}.pt"))
        print(f"{MODEL}: {evaluate(model, test_loader)[0]}")

        MODEL = f'mlp_within_raw_{reps}'
        model = MLP(48)
        # model.load_state_dict(torch.load(join(PATH, "mlp_raw.pt")))
        train(model=model, name=MODEL, 
            train_loader=train_feat_loader,
            val_loader=val_feat_loader,
            loss_fn=nn.CrossEntropyLoss(weight=weights),
            save_chkp=False, verbose=VERBOSE)
        torch.save(model.state_dict(), join(SGT_PATH, f"{MODEL}.pt"))
        print(f"{MODEL}: {evaluate(model, test_feat_loader)[0]}")
        
        del train_loader, val_loader, train_feat_loader, val_feat_loader
        torch.cuda.empty_cache()
        gc.collect()