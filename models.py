import torch
import torch.nn as nn; import torch.nn.functional as F

from utils import *


# ======== MODELS ========

class CNN(nn.Module):
    def __init__(self, ch=CH, seq=SEQ, emb_dim=128, 
                 num_classes=CLASSES, dropout=DROPOUT):
        super().__init__()
        self.emb_dim = emb_dim
        self.dropout = dropout

        self.conv1 = nn.Conv1d(ch, 64, 8, padding="same")
        self.conv2 = nn.Conv1d(64, 128, 4, padding="same")
        self.conv3 = nn.Conv1d(128, 256, 4, padding="same")

        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(256, 128)
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
