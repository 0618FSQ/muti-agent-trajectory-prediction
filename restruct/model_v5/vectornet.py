import math
from typing import List, Optional
from model_v3.layers import TrajectoryEncoder
from model_v3.layers import GlobalGraph
from model_v3.layers import Dense
from model_v3.layers import CrossEncoder
import torch
import numpy as np
import torch.nn as nn


class FourierEmbedding(nn.Module):
    def __init__(self, input_size, hidden_size, freq_bands):
        super().__init__()
        self.dense = nn.Parameter(torch.randn(input_size, freq_bands))
        self.multy_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(freq_bands * 2 + 1, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size)
                )
                for _ in range(input_size)

            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        """

        Args:
            x: [b, polygons, input_size]

        Returns:

        """
        output = torch.einsum("ijk,kl->ijkl", x, self.dense) * 2 * math.pi
        output = torch.cat([output.cos(), output.sin(), x.unsqueeze(-1)], dim=-1)
        output = torch.stack([layer(output[:, :, i]) for i, layer in enumerate(self.multy_mlp)])
        output = output.sum(0)
        output = self.out(output)
        return output


class FourierEmbedding2(nn.Module):
    def __init__(self, input_size, hidden_size, freq_bands):
        super().__init__()
        self.dense = nn.Parameter(torch.randn(input_size, freq_bands))
        self.multy_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(freq_bands * 2 + 1, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size)
                )
                for _ in range(input_size)

            ]
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, x):
        """

        Args:
            x: [b, polygons, points, input_size]

        Returns:

        """
        output = torch.einsum("ijkl,lh->ijklh", x, self.dense) * 2 * math.pi
        output = torch.cat([output.cos(), output.sin(), x.unsqueeze(-1)], dim=-1)
        output = torch.stack([layer(output[:, :, :, i]) for i, layer in enumerate(self.multy_mlp)])
        output = output.sum(0)
        output = self.out(output)
        return output



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


if __name__ == '__main__':
    inputs = torch.randn(64, 20, 10, 15)
    net = FourierEmbedding2(15, 128, 256)
    print(net(inputs))

    inputs = torch.randn(64, 20, 15)
    net = FourierEmbedding(15, 128, 256)
    print(net(inputs))