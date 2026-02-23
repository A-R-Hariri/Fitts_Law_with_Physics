import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch, torch.nn as nn
import libemg
import numpy as np
import socket, threading, random
from datetime import datetime
from multiprocessing import Manager

from utils import * 
from models import CNN
from Fitts import Dashboard, QApplication

SEED = 13
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

class MultiModelWrapper(nn.Module):
    def __init__(self, models_dict, shared_context, device=DEVICE):
        super().__init__()
        self.models = nn.ModuleDict(models_dict)
        self.device = device
        self.sc = shared_context
        self.active_name = None
        self.active_model = None

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device, non_blocking=True).float()
        
        name = self.sc.active_model_name
        if name != self.active_name:
            if name in self.models:
                self.active_name = name
                self.active_model = self.models[name]
                print(f"[MODEL CHANGE]: {name}")
            else:
                print(f"[CRITICAL] Model '{name}' not found in wrapper.")
                os._exit(1)
        return self.active_model(x)

    @torch.no_grad()
    def predict_proba(self, x): return self(x).cpu().numpy()
    @torch.no_grad()
    def predict(self, x): return self(x).argmax(1).cpu().numpy()

def load_all_models(model_names):
    models = {}
    for name in model_names:
        w_path = join(PATH, f"{name}.pt")
        m = CNN().to(DEVICE)
        m.load_state_dict(torch.load(w_path, map_location=DEVICE))
        m.eval()
        models[name] = m
        print(f"[SUCCESS] Loaded: {name}")
    return models

def input_thread(sock, sc):
    print("Input thread started...")
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            parts = data.decode("utf-8").strip().split(' ')
            if len(parts) >= 6:
                probs_list = [float(p) for p in parts[:-2]]
                raw_vel = float(parts[-2])          # [-1] is timestamp
                gesture = np.argmax(np.array(probs_list))
                
                flip = sc.flip_lr
                # speed_mult = sc.speed_multiplier * VEL_CONSTANT
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

    model_names = ['cnn_raw', 'cnn_relabeled', 'cnn_segmented', 
                #    'cnn_raw_eq', 'cnn_relabeled_eq', 'cnn_segmented_eq',
                   'cnn_raw_rest', 'cnn_relabeled_rest', 'cnn_segmented_rest']
    loaded_models = load_all_models(model_names)
    SharedContext.available_models = list(loaded_models.keys())
    SharedContext.active_model_name = model_names[0]
    
    wrapper = MultiModelWrapper(loaded_models, SharedContext).to(DEVICE)
    o_classifier = libemg.emg_predictor.EMGClassifier(wrapper)

    o_classifier.add_velocity([], [])
    th_max_path = join(PATH, 'th_max_dic.npy')
    th_min_path = join(PATH, 'th_min_dic.npy')
    if os.path.exists(th_max_path):
        o_classifier.th_max_dic = np.load(th_max_path, allow_pickle=True).item()
        o_classifier.th_min_dic = np.load(th_min_path, allow_pickle=True).item()

    p, smm = libemg.streamers.myo_streamer()
    odh = libemg.data_handler.OnlineDataHandler(smm)
    classifier = libemg.emg_predictor.OnlineEMGClassifier(o_classifier, SEQ, INC, odh, None, output_format='probabilities')
    classifier.run(block=False)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); sock.bind(('127.0.0.1', 12346))
    threading.Thread(target=input_thread, args=(sock, SharedContext), daemon=True).start()

    if not os.path.exists(DATA_PATH): os.mkdir(DATA_PATH)
    odh.log_to_file(file_path=join(DATA_PATH, f"fitts_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"))

    app = QApplication(sys.argv)
    dash = Dashboard(SharedContext)
    dash.update_model_list(SharedContext.available_models)
    sys.exit(app.exec())