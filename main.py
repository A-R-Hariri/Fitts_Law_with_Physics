import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch, torch.nn as nn
import libemg, random
import numpy as np
import socket, threading, random, select
from datetime import datetime
from multiprocessing import Manager

from utils import * 
from models import MHCNN, CNN_HCF, RunningNorm
from fitts import Dashboard, QApplication

# SEED = 13
# random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

pop_mean = np.array([-0.70885944, -0.74997824, -0.47742087, -0.73471236, -0.99069226, -0.9039961,
                    -0.8920331, -0.78142345])
pop_std = np.array([21.533228, 21.636055, 28.874157, 30.270008, 17.713427, 13.582925, 19.829926,
                    21.28387])
                                                                                    
dummy_feats = extract_sub(np.ones((10, CH, SEQ)), FEAT_LIST, FEATURE_DIC).transpose(0, 2, 1)
n_feat_sub = dummy_feats.shape[1]  # F per sub-window

class MultiModelWrapper(nn.Module):
    def __init__(self, models_dict, shared_context, 
                 feature_list=FEAT_LIST, feature_dic=FEATURE_DIC,
                 device=DEVICE, std=None, mean=None):
        super().__init__()
        self.models = nn.ModuleDict(models_dict)
        self.device = device
        self.sc = shared_context
        self.active_name = None
        self.active_model = None
        self.feature_list = feature_list
        self.feature_dic = feature_dic
        self.mean = mean
        self.std = std
        self.fe = libemg.feature_extractor.FeatureExtractor()

        self.rn= RunningNorm(CH, tau=float('inf'),
                    init_mean=pop_mean, init_std=pop_std).eval()

    def forward(self, x):
        name = self.sc.active_model_name
        if name not in self.models:
            return None
        
        if 'hcf' in name:
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            x = extract_sub(x, self.feature_list, self.feature_dic)  # (B, 4, F)
            x = x.transpose(0, 2, 1)                                 # (B, F, 4)
            # normalize using feature-level scaler
            x = (x - self.mean[:, None]) / self.std[:, None]  # (B, F, 4)
            x = torch.from_numpy(x).to(self.device, non_blocking=True).float()
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device, non_blocking=True).float()
        
        if name != self.active_name:
            self.active_name = name
            self.active_model = self.models[name]
            print(f"[MODEL CHANGE]: {name}")
            self.rn.reset()

        if 'rn' in self.sc.active_model_name:
            x = self.rn(x)

        return self.active_model(x)

    def predict_proba(self, x):
        if self.sc.active_model_name not in self.models: return np.zeros((1, 5))
        return self(x).detach().cpu().numpy()
        
    @torch.no_grad()
    def predict(self, x):
        if self.sc.active_model_name not in self.models: return np.zeros((1,))
        return self(x).detach().argmax(1).cpu().numpy()

def load_all_models(model_names):
    models = {}
    for name in model_names:
        if 'within' in name:
            w_path = join(SGT_PATH, f"{name}.pt")
        else:
            w_path = join(CHECKPOINT_PATH, 
                        #   f"{name}", 
                          f"{name}.pt")
        if 'cnnhcf' in name:
            m = CNN_HCF(n_feat_sub).to(DEVICE)
        else:
            m = MHCNN().to(DEVICE)
        m.load_state_dict(torch.load(w_path, map_location=DEVICE)['model_state_dict'])
        m.eval()
        models[name] = m
        print(f"[SUCCESS] Loaded: {name}")
    return models

def input_thread(sockets_dict, sc):
    print("Input thread started...")
    socks_list = list(sockets_dict.values())
    while True:
        try:
            readable, _, _ = select.select(socks_list, [], [])
            for sock in readable:
                data, _ = sock.recvfrom(1024)
                
                name = sc.active_model_name
                if 'within' in name: active_cat = 'within'
                else: active_cat = 'normal'
                
                if sock != sockets_dict.get(active_cat):
                    continue

                parts = data.decode("utf-8").strip().split(' ')
                if len(parts) >= 6:
                    probs_list = [float(p) for p in parts[:-2]]
                    raw_vel = float(parts[-2])          # [-1] is timestamp
                    gesture = np.argmax(np.array(probs_list))
                    
                    flip = sc.flip_lr
                    # speed_mult = sc.speed_multiplier * VEL_CONSTANT       # Applied in Fitts class instead
                    speed = np.clip(raw_vel, 0.0, 1.0) #* speed_mult

                    dx, dy = 0.0, 0.0
                    if gesture == 1: dy = 1
                    elif gesture == 4: dy = -1
                    elif gesture == 2: dx = 1 if flip else -1
                    elif gesture == 3: dx = -1 if flip else 1

                    sc.emg_x, sc.emg_y = dx * speed, dy * speed
                    sc.probs, sc.raw_velocity = probs_list, raw_vel
        except: pass

if __name__ == "__main__":

    filters = [libemg.data_handler.RegexFilter(left_bound="C_", right_bound="_R", 
                                            values=["0","1","2","3","4"], description='classes'),
            libemg.data_handler.RegexFilter(left_bound="R_", right_bound="_emg.csv", 
                                            values=[str(r) for r in range(TOTAL_REPS - 1)], description='reps')]
    offline_dh = libemg.data_handler.OfflineDataHandler()
    offline_dh.get_data(folder_location=SGT_PATH, regex_filters=filters, delimiter=',')
    offline_odh = offline_dh.isolate_data("reps", list(range(TOTAL_REPS - 1)), fast=True)
    train_windows, train_meta = offline_odh.parse_windows(SEQ, INC)

    train_feats = extract_sub(train_windows, FEAT_LIST, FEATURE_DIC).transpose(0, 2, 1)  # (N, F, 4)
    _feat_mean = train_feats.mean(axis=(0, 2))  # (F,)
    _feat_std  = train_feats.std(axis=(0, 2))   # (F,)
    train_meta['classes'] = remap_labels(train_meta['classes'])

    manager = Manager()
    SharedContext = manager.Namespace()
    
    # Initialize Namespace values BEFORE spawning libemg
    SharedContext.emg_x = 0.0
    SharedContext.emg_y = 0.0
    SharedContext.probs = [0.0]*5
    SharedContext.raw_velocity = 0.0
    SharedContext.flip_lr = False
    SharedContext.speed_multiplier = 1.0
    
    SharedContext.params = PARAMS

    model_names = [ 'cross_mhcnn_raw_base',
                    'within_cnnhcf_raw_base-5',
                    'cross_mhcnn_raw_1va',
                    'cross_mhcnn_raw_base-rn',
                    'cross_mhcnn_raw_rest',
                    'cross_mhcnn_raw_trp',
                    'cross_mhcnn_segmented_base',
                    'within_mhcnn_raw_base-ft-1',
                    'within_mhcnn_raw_base-ft-5',
                    ]
    
    random.shuffle(model_names)
        
    loaded_models = load_all_models(model_names)
    SharedContext.available_models = list(loaded_models.keys())
    SharedContext.active_model_name = model_names[0]

    dict_normal = {k: v for k, v in loaded_models.items() if 'within' not in k}
    dict_within_within = {k: v for k, v in loaded_models.items() if 'within' in k}

    wrapper_normal = MultiModelWrapper(dict_normal, SharedContext).to(DEVICE).eval()
    wrapper_within_within = MultiModelWrapper(dict_within_within, SharedContext, std=_feat_std, 
                                           mean=_feat_mean).to(DEVICE).eval()

    o_class_normal = libemg.emg_predictor.EMGClassifier(wrapper_normal)
    o_class_within_within = libemg.emg_predictor.EMGClassifier(wrapper_within_within)

    o_class_normal.add_velocity([], [])
    th_max_path = join(CHECKPOINT_PATH, 'th_max_dic.npy')
    th_min_path = join(CHECKPOINT_PATH, 'th_min_dic.npy')
    if os.path.exists(th_max_path):
        o_class_normal.th_max_dic = np.load(th_max_path, allow_pickle=True).item()
        o_class_normal.th_min_dic = np.load(th_min_path, allow_pickle=True).item()
    
    o_class_within_within.add_velocity(train_windows, train_meta['classes'])

    p, smm = libemg.streamers.myo_streamer()
    odh = libemg.data_handler.OnlineDataHandler(smm)
    p_norm, p_within = 12346, 12347
    on_normal = libemg.emg_predictor.OnlineEMGClassifier(o_class_normal, SEQ, INC, odh, 
                                                        ip='127.0.0.1', port=p_norm,
                                                        features=None, output_format='probabilities')
    on_within_within = libemg.emg_predictor.OnlineEMGClassifier(o_class_within_within, SEQ, INC, odh, 
                                                            ip='127.0.0.1', port=p_within,
                                                            features=None, output_format='probabilities')
    
    on_normal.run(block=False)
    on_within_within.run(block=False)

    s_norm = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s_norm.bind(('127.0.0.1', p_norm))
    s_within = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s_within.bind(('127.0.0.1', p_within))
    sockets_dict = {'normal': s_norm, 'within': s_within}

    threading.Thread(target=input_thread, args=(sockets_dict, SharedContext), daemon=True).start()

    if not os.path.exists(DATA_PATH): os.mkdir(DATA_PATH)
    odh.log_to_file(file_path=join(DATA_PATH, f"fitts_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"))

    app = QApplication(sys.argv)
    dash = Dashboard(SharedContext)
    dash.update_model_list(SharedContext.available_models)
    sys.exit(app.exec())