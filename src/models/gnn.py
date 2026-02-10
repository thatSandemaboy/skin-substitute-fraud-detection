"""
Graph Neural Network models for fraud detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool


class FraudGNN(nn.Module):
    """
    Graph Neural Network for provider fraud classification.
    
    Uses GraphSAGE convolutions to aggregate neighborhood information.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        return self.classifier(x)
    
    def get_embeddings(self, x, edge_index):
        """Get node embeddings without classification head."""
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


class FraudGAT(nn.Module):
    """
    Graph Attention Network for fraud detection.
    
    Uses attention mechanism to weight neighbor importance.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return self.classifier(x)


class AnomalyDetector(nn.Module):
    """
    Graph Autoencoder for unsupervised anomaly detection.
    
    Learns normal patterns; high reconstruction error = anomaly.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        latent_channels: int = 16,
    ):
        super().__init__()
        
        # Encoder
        self.encoder_conv1 = SAGEConv(in_channels, hidden_channels)
        self.encoder_conv2 = SAGEConv(hidden_channels, latent_channels)
        
        # Decoder
        self.decoder = nn.Linear(latent_channels, in_channels)
    
    def encode(self, x, edge_index):
        x = self.encoder_conv1(x, edge_index)
        x = F.relu(x)
        x = self.encoder_conv2(x, edge_index)
        return x
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_recon = self.decode(z)
        return x_recon, z
    
    def reconstruction_error(self, x, edge_index):
        """Compute per-node reconstruction error (anomaly score)."""
        x_recon, _ = self.forward(x, edge_index)
        error = torch.mean((x - x_recon) ** 2, dim=1)
        return error
