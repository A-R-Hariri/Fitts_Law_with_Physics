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

        self.conv1 = nn.Conv1d(ch, 32, 8, dilation=1, padding="same")
        self.conv2 = nn.Conv1d(ch, 32, 8, dilation=2, padding="same")
        self.conv3 = nn.Conv1d(ch, 32, 8, dilation=4, padding="same")
        self.conv4 = nn.Conv1d(96, 128, 4, dilation=1, padding="same")

        self.pool = nn.AdaptiveAvgPool1d(1)
        # with torch.no_grad():
        #     dummy = torch.zeros(1, ch, seq)
        #     x1 = self.conv1(dummy)
        #     x2 = self.conv2(dummy)
        #     x3 = self.conv3(dummy)
        #     x = torch.cat((x1, x2, x3), 1)
        #     x = self.conv4(x)
        #     fc_in = x.flatten(1).shape[1]
        
        self.fc1 = nn.Linear(128, 128)
        self.fc_emb = nn.Linear(128, emb_dim)

        self.drop = nn.Dropout(self.dropout)
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

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

        x1 = self.relu(self.conv1(x))
        x1 = self.drop(x1)
        x2 = self.relu(self.conv2(x))
        x2 = self.drop(x2)
        x3 = self.relu(self.conv3(x))
        x3 = self.drop(x3)
        x = torch.cat((x1, x2, x3), 1)
        x = self.relu(self.conv4(x))
        x = self.drop(x)

        x = self.pool(x).squeeze(-1)
        # x = x.flatten(1)

        x = self.fc1(x)
        x = self.gelu(x)
        emb = self.fc_emb(x)

        logits = self.classifier(emb)

        if return_emb and return_logits:
            return emb, logits
        if return_emb:
            return emb
        return logits
