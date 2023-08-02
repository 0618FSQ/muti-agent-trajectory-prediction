import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import CrossEncoder
from .layers import MLP
from .layers import MultyHeadAttn
from .vectornet import VectorNet


# class MLP(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()

#         self.lin1 = nn.Linear(hidden_size, hidden_size)
#         self.acti = nn.ReLU()
#         self.lin2 = nn.Linear(hidden_size, hidden_size)

#     def forward(self, x):
#         out = self.lin2(self.acti(self.lin1(x)))
#         return x + out


class Decoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_modes=6,
                 obs_horizon=20,
                 future_horizon=30,
                 sublayers=3

                 ):
        super().__init__()
        self.future_horizon = future_horizon
        self.num_modes = num_modes

        self.position_encoder = nn.Parameter(
            torch.randn([1, future_horizon, hidden_size], dtype=torch.float32)
        )

        self.classification = nn.Sequential(
            nn.Linear(hidden_size, num_modes),
            nn.Softmax()
        )
        
        pred_traj_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.pred_dense = nn.Sequential(
            MLP(hidden_size, hidden_size),
            MLP(hidden_size, hidden_size),
            MLP(hidden_size, hidden_size),
            nn.Linear(hidden_size, 2*num_modes)
        )
        
        self.pred_traj_encoder =  nn.TransformerEncoder(pred_traj_encoder, sublayers)

    def forward(self, agt_emb):
        # target_agent_feature = self.multi_layers(agt_emb)
        
        # target_agent_feature = self.pred_traj_encoder(target_agent_feature)
        classification = torch.max(agt_emb, dim=1)[0]
        future_trajectory = classification.unsqueeze(dim=1) + self.position_encoder
        pred_traj = self.pred_traj_encoder(future_trajectory)   
        
        
        
        classification = self.classification(classification)
        pred_traj = self.pred_dense(pred_traj)
        pred_traj = pred_traj.view([-1, self.future_horizon, self.num_modes, 2]).permute(0, 2, 1, 3)
        return pred_traj, classification
    
    

class FutureCrossEncoder(nn.Module):
    def __init__(self, hidden_size, sublayers, num_heads, obs_horizon):
        super().__init__()

        self.cross_encoder = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "future_vectornet": VectorNet(
                            in_channels=hidden_size,
                            hidden_size=hidden_size,
                            sub_layers=sublayers,
                            num_heads=num_heads
                        ),
                        "f2h_cross_encoder": MultyHeadAttn(d_model=hidden_size,
                                                           num_heads=num_heads,
                                                           dropout=0.2),
                        "f2a_cross_encoder": MultyHeadAttn(d_model=hidden_size,
                                                           num_heads=num_heads,
                                                           dropout=0.2),
                        "f2m_cross_encoder": MultyHeadAttn(d_model=hidden_size,
                                                           num_heads=num_heads,
                                                           dropout=0.2)
                    }
                )
                for _ in range(sublayers)
            ]
        )

        self.position_embedding = nn.Parameter(torch.randn(1, 1, obs_horizon, hidden_size))
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self,
                global_traj_padding,
                local_traj_padding,
                mask,
                valid_mask,
                map_feature,
                agent_feature,
                f2h_cross_mask,
                f2a_cross_mask,
                f2m_cross_mask):
        traj_feature = global_traj_padding

        for cross_encoder in self.cross_encoder:
            traj_feature = traj_feature.unsqueeze(-2) + self.position_embedding + local_traj_padding.unsqueeze(-2)
            traj_feature = cross_encoder["future_vectornet"](traj_feature, local_traj_padding, mask, valid_mask)
            traj_feature = cross_encoder["f2m_cross_encoder"](
                traj_feature,
                map_feature,
                map_feature,
                attn_mask=f2m_cross_mask
            )
            traj_feature = cross_encoder["f2a_cross_encoder"](
                traj_feature,
                agent_feature,
                agent_feature,
                attn_mask=f2a_cross_mask
            )
            traj_feature = cross_encoder["f2h_cross_encoder"](
                traj_feature,
                global_traj_padding,
                global_traj_padding,
                attn_mask=f2h_cross_mask
            )

        traj_feature = self.layernorm(traj_feature) + global_traj_padding

        return traj_feature