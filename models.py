import torch
import torch.nn as nn; import torch.nn.functional as F

from utils import *

# ======== MODELS, TRAINING & DATASETS ========

# ======== MODEL 1: CNN ========
class CNN(nn.Module):
    def __init__(self, ch=CH, seq=SEQ, emb_dim=64, 
                 num_classes=CLASSES, dropout=DROPOUT):
        super().__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout

        self.conv1 = nn.Conv1d(ch, 32, 8, padding="same")
        self.conv2 = nn.Conv1d(32, 64, 4, padding="same")
        self.conv3 = nn.Conv1d(64, 128, 4, padding="same")

        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(128, 128)
        self.fc_emb = nn.Linear(128, emb_dim)

        self.drop = nn.Dropout(self.dropout)

        self.classifier = nn.Linear(self.emb_dim, 
                                    num_classes)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, return_emb=False, return_logits=False):
        x /= 128

        x = F.relu(self.conv1(x))
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = F.relu(self.conv3(x))
        x = self.drop(x)

        x = self.pool(x).squeeze(-1)

        x = self.fc1(x)
        x = F.gelu(x)
        emb = self.fc_emb(x)

        logits = self.classifier(emb)

        if return_emb and return_logits:
            return emb, logits
        if return_emb:
            return emb
        return logits

# ======== MODEL 2: TRANSFORMER ========
class EMGTransformer(nn.Module):
    def __init__(self, ch=CH, seq=SEQ, 
                 classes=CLASSES, dropout=DROPOUT,
                   d=64, heads=4, layers=2, ff=256):
        super().__init__()
        self.seq = seq
        self.inp = nn.Linear(ch, d)
        self.pos = nn.Parameter(torch.zeros(1, seq, d))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.mlp = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(32, classes)

        self.apply(self._init)
        print(count_params(self))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, return_emb=False):
        x = (x / 128).permute(0, 2, 1)          # (B, T, C)
        x = self.inp(x) + self.pos[:, :x.size(1)]
        x = self.enc(x)
        x = x.mean(dim=1)                       # global avg pool over time
        emb = self.mlp(x)
        if return_emb:
            return emb
        return self.head(emb)


# ======== MODEL 3: LSTM ========
class EMGLSTM(nn.Module):
    def __init__(self, ch=CH, seq=SEQ, 
                 classes=CLASSES, dropout=DROPOUT,
                 hidden=48, layers=2):
        super().__init__()
        self.seq = seq
        self.lstm = nn.LSTM(
            input_size=ch, hidden_size=hidden, num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=False, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(32, classes)

        self.apply(self._init)
        print(count_params(self))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, return_emb=False):
        x = (x / 128).permute(0, 2, 1)          # (B, T, C)
        out, _ = self.lstm(x)                    # (B, T, 2H)
        x = out[:, -1]                           # last timestep
        emb = self.mlp(x)
        if return_emb:
            return emb
        return self.head(emb)



class RestLoss(nn.Module):
    def __init__(self, alpha1=0.25, alpha2=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def forward(self, logits, targets):
        loss = self.ce(logits, targets)
        pred = logits.argmax(1)

        p1 = (pred != targets) & (pred != 0)
        loss = loss * (1 + self.alpha1 * p1.float())

        p2 = (targets == 0) & (pred != 0)
        loss = loss * (1 + self.alpha2 * p2.float())

        return loss.mean()
    

class EqLoss(nn.Module):
    def __init__(self, std_scale=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.std_scale = std_scale

    def forward(self, logits, targets):
        l = self.ce(logits, targets)
        return l.mean() + self.std_scale * l.std()
    
    
