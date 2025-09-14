import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularMLP(nn.Module):
    def __init__(self, in_dim, hid=64, out_dim=32, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hid, out_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class LateFusionClassifier(nn.Module):
    def __init__(self, img_dim=2048, clin_dim=32, out_dim=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(img_dim + clin_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )
    def forward(self, img_feats, clin_embed):
        z = F.normalize(img_feats, dim=1)
        return self.classifier(torch.cat([z, clin_embed], dim=1))  