import torch
import torch.nn as nn
import torch.nn.functional as F

class MetadataFusionMLP(nn.Module):
    def __init__(self, tabular_dim, text_dim, hidden_dims=[128, 64, 32], dropout=0.3, out_dim=2):
        super().__init__()
        
        # Tabular feature processing
        self.tabular_net = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Text feature processing
        self.text_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Fusion layers
        fusion_layers = []
        prev_dim = hidden_dims[0] * 2  # Concatenated tabular + text
        
        for h in hidden_dims[1:]:
            fusion_layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        
        self.fusion_net = nn.Sequential(*fusion_layers)
        self.head = nn.Linear(prev_dim, out_dim)
    
    def forward(self, tabular_x, text_x):
        # Process each modality
        tab_feats = self.tabular_net(tabular_x)
        text_feats = self.text_net(text_x)
        
        # Concatenate and fuse
        fused = torch.cat([tab_feats, text_feats], dim=1)
        fused = self.fusion_net(fused)
        return self.head(fused)

class MetadataResidualMLP(nn.Module):
    """Residual version with separate processing for tabular and text"""
    
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
    
    def __init__(self, tabular_dim, text_dim, hidden_dim=128, n_blocks=3, dropout=0.3, out_dim=2):
        super().__init__()
        
        # Tabular branch
        self.tabular_project = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.tabular_blocks = nn.ModuleList([
            self.ResBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        
        # Text branch
        self.text_project = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.text_blocks = nn.ModuleList([
            self.ResBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fusion_blocks = nn.ModuleList([
            self.ResBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        
        self.head = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, tabular_x, text_x):
        # Process tabular features
        tab = self.tabular_project(tabular_x)
        for block in self.tabular_blocks:
            tab = block(tab)
        
        # Process text features
        txt = self.text_project(text_x)
        for block in self.text_blocks:
            txt = block(txt)
        
        # Fuse and process
        fused = self.fusion(torch.cat([tab, txt], dim=1))
        for block in self.fusion_blocks:
            fused = block(fused)
        
        return self.head(fused)