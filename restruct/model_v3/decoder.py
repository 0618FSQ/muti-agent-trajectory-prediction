import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_v3.layers import CrossEncoder
from model_v3.layers import MLP

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