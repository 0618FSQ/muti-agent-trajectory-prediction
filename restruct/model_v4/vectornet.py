import torch
from torch import nn
from .layers import TrajectoryEncoder
from .layers import GlobalGraph
from .layers import Dense
from .layers import CrossEncoder
import numpy as np


class SpatioTemporalNet(nn.Module):
    def __init__(self,
                 in_channels=8,
                 hidden_size=64,
                 sub_layers=3):
        super().__init__()

        self.traj_embedding = nn.Sequential(
            Dense(in_channels, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.dense = nn.Sequential(
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.spatio_traj_encoder = TrajectoryEncoder(
            hidden_size, sub_layers
        )

        self.temporal_traj_encoder = TrajectoryEncoder(
            hidden_size, sub_layers
        )

        self.spation_temporal_cross_encoder = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )

        self.global_graph = GlobalGraph(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )

        self.dense2 = nn.Sequential(
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
    @staticmethod
    def max_pooling(x, agt_mask=None):
        if agt_mask is not None:
            x_masked = torch.max(torch.masked_fill(x, ~agt_mask.unsqueeze(dim=-1), 0), dim=-2)[0] 
            return x_masked
        else:
            return torch.max(x, dim=-2)[0]

    def forward(self, padding, mask=None, valid_mask=None):
        traj_encoding = self.traj_embedding(padding)  # [bs, elements_amount, points, hidden_size]
        spatio_traj_encoding = self.spatio_traj_encoder(traj_encoding, mask)  # [bs, elements_amount, points, hidden_size]
        temporal_traj_encoding = self.temporal_traj_encoder(traj_encoding.transpose(1, 2), mask.transpose(1, 2))
        spatio_traj_encoding = self.dense(spatio_traj_encoding)
        temporal_traj_encoding = self.dense(temporal_traj_encoding)

        spatio_traj_feature = self.max_pooling(spatio_traj_encoding, mask)  # [bs, elements_amount, hidden_size]
        temporal_traj_feature = self.max_pooling(temporal_traj_encoding, mask.transpose(1, 2))  # [bs, elements_amount, hidden_size]

        traj_feature = self.spation_temporal_cross_encoder(spatio_traj_feature, temporal_traj_feature, mask)
        traj_feature = self.dense2(traj_feature)
        traj_feature = self.global_graph(traj_feature, valid_mask)
        return traj_feature

class VectorNet(nn.Module):
    def __init__(self,
                 in_channels=8,
                 hidden_size=64,
                 sub_layers=3
    ):
        super().__init__()
        self.traj_embedding = nn.Sequential(
            Dense(in_channels, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.dense = nn.Sequential(
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.traj_encoder = TrajectoryEncoder(
            hidden_size, sub_layers
        )
        self.global_graph = GlobalGraph(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )
        
        self.dense2 = nn.Sequential(
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

    @staticmethod
    def max_pooling(x, agt_mask=None):
        if agt_mask is not None:
            x_masked = torch.max(torch.masked_fill(x, ~agt_mask.unsqueeze(dim=-1), 0), dim=-2)[0] 
            return x_masked
        else:
            return torch.max(x, dim=-2)[0]

    def forward(self, padding, mask=None, valid_mask=None):
        traj_encoding = self.traj_embedding(padding)  # [bs, elements_amount, points, hidden_size]
        traj_encoding = self.traj_encoder(traj_encoding, mask)  # [bs, elements_amount, points, hidden_size]
        traj_encoding = self.dense(traj_encoding)
        traj_feature = self.max_pooling(traj_encoding, mask)  # [bs, elements_amount, hidden_size]
        traj_feature = self.dense2(traj_feature)
        traj_feature = self.global_graph(traj_feature, valid_mask)
        return traj_feature