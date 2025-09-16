import torch
import torch.nn as nn
import torch.nn.functional as F

class MetadataOnlyMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=[128, 64, 32], dropout=0.3, out_dim=2):
        super().__init__()
        layers = []
        prev_dim = in_dim
        
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, out_dim)
    
    def forward(self, x):
        z = self.net(x)
        return self.head(z)

class MetadataOnlyResidualMLP(nn.Module):
    """Alternative with residual connections for better gradient flow"""
    
    class ResBlock(nn.Module):
        def __init__(self, dim, dropout=0.3):
            super().__init__()
            self.block = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim)
            )
        
        def forward(self, x):
            return F.relu(x + self.block(x))
    
    def __init__(self, in_dim, hidden_dim=128, n_blocks=3, dropout=0.3, out_dim=2):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList([
            self.ResBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.head = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        x = self.project(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
