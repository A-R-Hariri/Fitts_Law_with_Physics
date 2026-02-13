import warnings, sys, os, gc
from os.path import join
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch, torch.nn as nn
import libemg
import numpy as np
import socket, threading, time, random
from datetime import datetime
from glob import glob


from utils import * 
from models import CNN
from Fitts import shared, Dashboard, QApplication, VEL_CONSTANT


SEED = 13; random.seed(SEED); np.random.seed(SEED)
GENERATOR = torch.manual_seed(SEED)


class MultiModelWrapper(nn.Module):
    def __init__(self, models_dict, device=DEVICE):
        super().__init__()
        self.models = nn.ModuleDict(models_dict)
        self.device = device
        self.active_name = None
        self.active_model = None

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device, non_blocking=True).float()

        name = shared.get_model()
        if name is None:
            name = list(self.models.keys())[0]
            shared.set_model(name)

        if name != self.active_name:
            if name in self.models:
                self.active_name = name
                self.active_model = self.models[name]
                print(f"[MODEl CHANGE]: {name}")
            else:
                print(f"[CRITICAL] Model '{name}' not found in: {list(self.models.keys())}")
                os._exit(1)
        return self.active_model(x)
    
    @torch.no_grad()
    def predict_proba(self, x):
        return self(x).cpu().numpy()
    
    @torch.no_grad()
    def predict(self, x):
        return self(x).argmax(1).cpu().numpy()

def load_all_models(model_names):
    models = {}
    for name in model_names:
        w_path = join(PATH, f"{name}.pt")
        if not os.path.exists(w_path):
            print(f"[ERROR] Required model file missing: {w_path}")
            sys.exit(1)
        
        try:
            m = CNN().to(DEVICE)
            m.load_state_dict(torch.load(w_path, map_location=DEVICE))
            m.eval()
            models[name] = m
            print(f"[SUCCESS] Loaded: {name}")
        except Exception as e:
            print(f"[FATAL] Failed to load {name}: {e}")
            sys.exit(1)
    return models

def input_thread(sock):
    print("Input thread started...")
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            data_str = str(data.decode("utf-8")).strip()
            parts = data_str.split(' ')
            
            if len(parts) >= 6:
                probs_list = [float(p) for p in parts[:-1]]
                raw_velocity = float(parts[-1])
                
                probs = np.array(probs_list)
                gesture = np.argmax(probs)
                
                # if probs[gesture] < 0.6: 
                #     gesture = 0
                
                with shared.lock:
                    flip = shared.flip_lr
                    speed_mult = shared.speed_multiplier * VEL_CONSTANT
                
                speed = np.clip(raw_velocity, 0.0, 1.0)
                speed *= speed_mult
                
                dx, dy = 0.0, 0.0
                if gesture == 1:
                    dx, dy = 0, 1 
                elif gesture == 4:
                    dx, dy = 0, -1
                elif gesture == 2:
                    dx, dy = (1, 0) if flip else (-1, 0)
                elif gesture == 3:
                    dx, dy = (-1, 0) if flip else (1, 0)
                
                final_x = dx * speed
                final_y = dy * speed
                
                shared.update_emg(final_x, final_y, probs_list, raw_velocity)
        except Exception as e:
            pass


if __name__ == "__main__":
    # Names of .pt files in the 'pickles' folder
    model_names = [
        'cnn_raw',
        'cnn_relabeled',
        'cnn_segmented',
        'cnn_raw_eq',
        'cnn_raw_rest'
    ]

    loaded_models = load_all_models(model_names)
    available_keys = list(loaded_models.keys())

    shared.set_models(list(loaded_models.keys()))
        
    wrapper = MultiModelWrapper(loaded_models).to(DEVICE)
    print(f"Models loaded into wrapper: {available_keys}")

    p, smm = libemg.streamers.myo_streamer() 
    odh = libemg.data_handler.OnlineDataHandler(smm)

    o_classifier = libemg.emg_predictor.EMGClassifier(wrapper)

    # Velocity threshold loading (User: "every data is in pickles/")
    # We look in PATH ('pickles') instead of dataset_folder.
    try:
        o_classifier.add_velocity([], [])
        
        th_max_path = join(PATH, 'th_max_dic.npy')
        th_min_path = join(PATH, 'th_min_dic.npy')

        if os.path.exists(th_max_path):
            o_classifier.th_max_dic = np.load(th_max_path, allow_pickle=True).item()
            o_classifier.th_min_dic = np.load(th_min_path, allow_pickle=True).item()
            print(f"Loaded velocity thresholds from: {th_max_path}")
        else:
            raise FileNotFoundError(f"Thresholds not found in {PATH}")
            
    except Exception as e:
        print(f"Re-calculating thresholds... ({e})")
        try:
            # Assumes training windows are also in pickles/ based on "every data is in pickles"
            # If not found there, you might need to revert to dataset_folder for these.
            train_windows = np.load(join(PATH, 'train_windows.npy'), mmap_mode="r")[::]
            train_meta = np.load(join(PATH, 'train_meta.npy'), allow_pickle=True).item()['classes'][::]
            o_classifier.add_velocity(train_windows, train_meta)
            
            # Save to pickles so they exist next time
            np.save(join(PATH, 'th_max_dic.npy'), o_classifier.th_max_dic)
            np.save(join(PATH, 'th_min_dic.npy'), o_classifier.th_min_dic)
            print(f"Calculated and saved thresholds to {PATH}")
        except:
            print(f"CRITICAL: Training data for velocity calibration not found in {PATH}. Velocity might be broken.")

    classifier = libemg.emg_predictor.OnlineEMGClassifier(
        o_classifier, SEQ, INC, odh, None, output_format='probabilities')
    classifier.run(block=False)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 12346))

    t = threading.Thread(target=input_thread, args=(sock,), daemon=True)
    t.start()

    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    odh.log_to_file(file_path=join(DATA_PATH, 
                    f"fitts_{datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')}"))

    app = QApplication(sys.argv)

    dash = Dashboard()
    dash.update_model_list(available_keys)

    sys.exit(app.exec())