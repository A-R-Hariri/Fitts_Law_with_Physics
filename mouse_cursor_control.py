import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch, torch.nn as nn
import libemg
import numpy as np
import socket, threading, random, select
from datetime import datetime
import ctypes
from multiprocessing import Manager

from utils import * 
from models import CNN, CNN_GRL, MLP
from fitts import Dashboard, QApplication

SEED = 13
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class MultiModelWrapper(nn.Module):
    def __init__(self, models_dict, shared_context, 
                 feature_list=FEATURE_LIST, feature_dic=FEATURE_DIC,
                 device=DEVICE):
        super().__init__()
        self.models = nn.ModuleDict(models_dict)
        self.device = device
        self.sc = shared_context
        self.active_name = None
        self.active_model = None
        self.feature_list = feature_list
        self.feature_dic = feature_dic
        self.fe = libemg.feature_extractor.FeatureExtractor()

    def forward(self, x):
        name = self.sc.active_model_name
        if name not in self.models:
            return None
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device, non_blocking=True).float()
        
        name = self.sc.active_model_name
        if name != self.active_name:
            self.active_name = name
            self.active_model = self.models[name]
            print(f"[MODEL CHANGE]: {name}")
        return self.active_model(x)

    def predict_proba(self, x):
        if self.sc.active_model_name not in self.models: return np.zeros((1, 5))
        # if 'mlp' in self.sc.active_model_name:
        #     x = self.fe.extract_features(self.feature_list, x, array=True,
        #                                 fix_feature_errors=False, 
        #                                 feature_dic=self.feature_dic).reshape((
        #                                     x.shape[0], -1))
        return self(x).detach().cpu().numpy()
        
    @torch.no_grad()
    def predict(self, x):
        if self.sc.active_model_name not in self.models: return np.zeros((1,))
        # if 'mlp' in self.sc.active_model_name:
        #     x = self.fe.extract_features(self.feature_list, x, array=True,
        #                                 fix_feature_errors=False, 
        #                                 feature_dic=self.feature_dic).reshape((
        #                                     x.shape[0], -1))
        return self(x).detach().argmax(1).cpu().numpy()

def load_all_models(model_names):
    models = {}
    for name in model_names:
        if 'within' in name:
            w_path = join(SGT_PATH, f"{name}.pt")
        else:
            w_path = join(PATH, f"{name}.pt")
        if 'grl' in name:
            m = CNN_GRL().to(DEVICE)
        elif 'mlp' in name:
            m = MLP(48).to(DEVICE)
        else:
            m = CNN().to(DEVICE)
        m.load_state_dict(torch.load(w_path, map_location=DEVICE))
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
                if 'mlp' in name: active_cat = 'within_mlp'
                elif 'within' in name: active_cat = 'within_cnn'
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

                move_x = int(dx * speed * 40)
                move_y = int(dy * speed * 40)
                if move_x != 0 or move_y != 0:
                    ctypes.windll.user32.mouse_event(0x0001, move_x, move_y, 0, 0)
        except: pass

if __name__ == "__main__":
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

    model_names = [
        'cnn_raw',
    ]
        
    loaded_models = load_all_models(model_names)
    SharedContext.available_models = list(loaded_models.keys())
    SharedContext.active_model_name = model_names[0]

    dict_normal = {k: v for k, v in loaded_models.items() if 'within' not in k and 'mlp' not in k}
    # dict_within_cnn = {k: v for k, v in loaded_models.items() if 'within' in k and 'mlp' not in k}
    # dict_within_mlp = {k: v for k, v in loaded_models.items() if 'mlp' in k}

    wrapper_normal = MultiModelWrapper(dict_normal, SharedContext).to(DEVICE)
    # wrapper_within_cnn = MultiModelWrapper(dict_within_cnn, SharedContext).to(DEVICE)
    # wrapper_within_mlp = MultiModelWrapper(dict_within_mlp, SharedContext).to(DEVICE)

    o_class_normal = libemg.emg_predictor.EMGClassifier(wrapper_normal)
    # o_class_within_cnn = libemg.emg_predictor.EMGClassifier(wrapper_within_cnn)
    # o_class_within_mlp = libemg.emg_predictor.EMGClassifier(wrapper_within_mlp)

    o_class_normal.add_velocity([], [])
    th_max_path = join(PATH, 'th_max_dic.npy')
    th_min_path = join(PATH, 'th_min_dic.npy')
    if os.path.exists(th_max_path):
        o_class_normal.th_max_dic = np.load(th_max_path, allow_pickle=True).item()
        o_class_normal.th_min_dic = np.load(th_min_path, allow_pickle=True).item()

    # filters = [libemg.data_handler.RegexFilter(left_bound="C_", right_bound="_R", 
    #                                         values=["0","1","2","3","4"], description='classes'),
    #         libemg.data_handler.RegexFilter(left_bound="R_", right_bound="_emg.csv", 
    #                                         values=[str(r) for r in range(15)], description='reps')]
    # offline_dh = libemg.data_handler.OfflineDataHandler()
    # offline_dh.get_data(folder_location=SGT_PATH, regex_filters=filters, delimiter=',')
    # offline_odh = offline_dh.isolate_data("reps", list(range(14)), fast=True)
    # train_windows, train_meta = offline_odh.parse_windows(SEQ, INC)
    # train_meta['classes'] = remap_labels(train_meta['classes'])

    # o_class_within_cnn.add_velocity(train_windows, train_meta['classes'])
    
    # o_class_within_mlp.add_velocity(train_windows, train_meta['classes'])
    # o_class_within_mlp.feature_params = FEATURE_DIC

    p, smm = libemg.streamers.myo_streamer()
    odh = libemg.data_handler.OnlineDataHandler(smm)
    p_norm, p_wcnn, p_wmlp = 12346, 12347, 12348
    on_normal = libemg.emg_predictor.OnlineEMGClassifier(o_class_normal, SEQ, INC, odh, 
                                                        ip='127.0.0.1', port=p_norm,
                                                        features=None, output_format='probabilities')
    # on_within_cnn = libemg.emg_predictor.OnlineEMGClassifier(o_class_within_cnn, SEQ, INC, odh, 
    #                                                         ip='127.0.0.1', port=p_wcnn,
    #                                                         features=None, output_format='probabilities')
    # on_within_mlp = libemg.emg_predictor.OnlineEMGClassifier(o_class_within_mlp, SEQ, INC, odh, 
    #                                                         ip='127.0.0.1', port=p_wmlp,
    #                                                         features=FEATURE_LIST, output_format='probabilities')
    
    on_normal.run(block=False)
    # on_within_cnn.run(block=False)
    # on_within_mlp.run(block=False)

    s_norm = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s_norm.bind(('127.0.0.1', p_norm))
    s_wcnn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s_wcnn.bind(('127.0.0.1', p_wcnn))
    s_wmlp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s_wmlp.bind(('127.0.0.1', p_wmlp))
    sockets_dict = {'normal': s_norm, 'within_cnn': s_wcnn, 'within_mlp': s_wmlp}

    threading.Thread(target=input_thread, args=(sockets_dict, SharedContext), daemon=True).start()

    if not os.path.exists(DATA_PATH): os.mkdir(DATA_PATH)
    odh.log_to_file(file_path=join(DATA_PATH, f"fitts_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"))

    app = QApplication(sys.argv)
    dash = Dashboard(SharedContext)
    dash.update_model_list(SharedContext.available_models)
    sys.exit(app.exec())