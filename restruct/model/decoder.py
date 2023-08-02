import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import CrossEncoder

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.acti = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self,x):
        out = self.lin2(self.acti(self.lin1(x)))
        return x + out
    
class GoalAtt(nn.Module):
    def __init__(
            self,
            in_channels,
            global_graph_width,
            need_scale=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.graph_width = global_graph_width
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(2, global_graph_width)
        self.v_lin = nn.Linear(2, global_graph_width)
       
        self.layernorm = nn.LayerNorm(global_graph_width)
        self.scale_factor_d = 1 + \
                              int(np.sqrt(self.in_channels)) if need_scale else 1
    def forward(self, agt_emb, goals, valid_mask=None):
        query = self.q_lin(agt_emb)
        key = self.k_lin(goals)
        value = self.v_lin(goals)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.graph_width)
        attention_weights = self.masked_softmax(scores, valid_mask)

        return attention_weights

    def masked_softmax(self, X, valid_mask=None):
        if valid_mask is not None:
            X_masked = torch.masked_fill(X, valid_mask, -1e12)
            return nn.functional.softmax(X_masked, dim=-1) * (1 - valid_mask.float())
        else:
            return nn.functional.softmax(X, dim=-1)

class Decoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_modes=6,
                 pred_horizon=30):
        super().__init__()
        
        self.pred_horizon = pred_horizon

        self.pred = nn.ModuleList(
            [
                nn.Sequential(MLP(hidden_size), nn.Linear(hidden_size, 2*pred_horizon))
                for _ in range(num_modes)
            ]
        )

        self.probs = GoalAtt(in_channels=hidden_size,
                                global_graph_width=hidden_size) 


    def forward(self, agt_emb):
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](agt_emb).reshape(-1, self.pred_horizon, 2))
        reg = torch.stack(preds)

        goals = reg[:, :, -1].permute(1, 0, 2).detach()
        probs = self.probs(agt_emb, goals)
        
        return reg, probs.squeeze(1)