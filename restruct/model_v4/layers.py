import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEncoder4FutureTrajectory(nn.Module):
    def __init__(self, hidden_size, sub_layers):
        super().__init__()
        self.cross_encoder_between_history_and_future = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )

        self.cross_encoder_between_future_and_graph = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )

    def forward(self,
                future_trajectory_feature, 
                histroy_trajectory, 
                graph_feature,
                history_and_future_mask,
                future_cross_graph_mask
                ):
        output = self.cross_encoder_between_history_and_future(future_trajectory_feature, 
                                                               histroy_trajectory,
                                                               history_and_future_mask)
        output = self.cross_encoder_between_future_and_graph(output, 
                                                             graph_feature, 
                                                             future_cross_graph_mask)
        output = future_trajectory_feature + output
        return output



class Dense(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(0.05)
        )
    def forward(self, x):
        return self.dense(x)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, input_size),
            nn.LayerNorm(input_size),
            nn.Dropout(0.05)
        )
    def forward(self, x):
        output = self.dense(x)
        output = output + x
        return output


class TrajectoryEncoder(nn.Module):
    def __init__(self, hidden_size, sub_layers):
        super().__init__()
        self.trajectory_encoder = nn.ModuleList(
            [
                TrajectoryEncoderSubLayer(hidden_size=hidden_size)
                for _ in range(sub_layers)
            ]
        )

    def forward(self, x, mask=None):
        for trajectory_encoder_layer in self.trajectory_encoder:
            x = trajectory_encoder_layer(x, mask)
        return x


class TrajectoryEncoderSubLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Sequential(
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.layernorm = nn.LayerNorm(hidden_size)

    @staticmethod
    def masked_max(x, mask):
        if mask is not None:
            x_masked = torch.max(torch.masked_fill(x, ~mask.bool().unsqueeze(dim=-1), -np.inf), dim=-2)[0]
            output = x + x_masked.unsqueeze(dim=-2)
            output = torch.masked_fill(output, ~mask.bool().unsqueeze(dim=-1), 0)
            return output
        else:
            x_masked = torch.max(x, dim=-2)[0]
            output = x + x_masked.unsqueeze(dim=-2)
            return output

    def forward(self, x, mask=None):
        output = self.dense(x)

        output = self.masked_max(output, mask)
        output = self.layernorm(output)
        return output


class CrossEncoder(nn.Module):

    def __init__(
            self,
            in_channels,
            global_graph_width,
            num_global_layers=1,
            need_scale=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.global_graph_width = global_graph_width

        self.layers = nn.ModuleList(
            [
                SelfAttentionWithKeyFCLayer(
                    in_channels,
                    self.global_graph_width,
                    need_scale
                )
                for _ in range(num_global_layers)
            ]
        )

    def forward(self, x, y, valid_mask=None):
        for layer in self.layers:
            x = layer(x, y, valid_mask)

        return x


class GlobalGraph(nn.Module):

    def __init__(
            self,
            in_channels,
            global_graph_width,
            num_global_layers=1,
            need_scale=False
    ):
        super(GlobalGraph, self).__init__()
        self.in_channels = in_channels
        self.global_graph_width = global_graph_width

        self.layers = nn.ModuleList(
            [
                SelfAttentionFCLayer(in_channels, global_graph_width, need_scale) if i == 0 else SelfAttentionFCLayer(global_graph_width, global_graph_width, need_scale)
                for i in range(num_global_layers)
            ]
        )

    def forward(self, x, valid_mask):
        for layer in self.layers:
            x = layer(x, valid_mask)
        return x


class SelfAttentionWithKeyFCLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            global_graph_width,
            need_scale=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.graph_width = global_graph_width
        self.q_lin = nn.Sequential(
            Dense(in_channels, global_graph_width),
            Dense(global_graph_width, global_graph_width),
            Dense(global_graph_width, global_graph_width)
        )
        self.k_lin = nn.Sequential(
            Dense(in_channels, global_graph_width),
            Dense(global_graph_width, global_graph_width),
            Dense(global_graph_width, global_graph_width)
        )
        self.v_lin = nn.Sequential(
            Dense(in_channels, global_graph_width),
            Dense(global_graph_width, global_graph_width),
            Dense(global_graph_width, global_graph_width)
        )
        self.layernorm = nn.LayerNorm(global_graph_width)
        self.scale_factor_d = 1 + \
                              int(np.sqrt(self.in_channels)) if need_scale else 1
    def forward(self, x1, x2, valid_mask=None):
        query = self.q_lin(x1)
        key = self.k_lin(x2)
        value = self.v_lin(x2)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.graph_width)
        attention_weights = self.masked_softmax(scores, valid_mask)
        x = torch.matmul(attention_weights, value)
        output = self.layernorm(x) + x1
        return output

    def masked_softmax(self, X, valid_mask=None):
        if valid_mask is not None:
            X_masked = torch.masked_fill(X, valid_mask, -1e12)
            return nn.functional.softmax(X_masked, dim=-1) * (1 - valid_mask.float())
        else:
            return nn.functional.softmax(X, dim=-1)


class SelfAttentionFCLayer(nn.Module):

    def __init__(self, in_channels, global_graph_width, need_scale=False):
        super(SelfAttentionFCLayer, self).__init__()
        self.in_channels = in_channels
        self.graph_width = global_graph_width
        
        self.q_lin = nn.Sequential(
            Dense(in_channels, global_graph_width),
            Dense(global_graph_width, global_graph_width),
            Dense(global_graph_width, global_graph_width)
        )
        self.k_lin = nn.Sequential(
            Dense(in_channels, global_graph_width),
            Dense(global_graph_width, global_graph_width),
            Dense(global_graph_width, global_graph_width)
        )
        self.v_lin = nn.Sequential(
            Dense(in_channels, global_graph_width),
            Dense(global_graph_width, global_graph_width),
            Dense(global_graph_width, global_graph_width)
        )
        self.layernorm = nn.LayerNorm(global_graph_width)
        self.scale_factor_d = 1 + \
                              int(np.sqrt(self.in_channels)) if need_scale else 1

    def forward(self, x, valid_mask=None):
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.graph_width)
        attention_weights = self.masked_softmax(scores, valid_mask)
        output = torch.matmul(attention_weights, value)
        output = self.layernorm(output) + x
        return output

    def masked_softmax(self, X, valid_mask=None):
        if valid_mask is not None:
            X_masked = torch.masked_fill(X, valid_mask, -1e12)
            return nn.functional.softmax(X_masked, dim=-1) * (1-valid_mask.float())
        else:
            return nn.functional.softmax(X, dim=-1)
        
        

class FusionLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            global_graph_width,
            need_scale=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.global_graph_width = global_graph_width

        self.t2a = SelfAttentionWithKeyFCLayer(
                    in_channels,
                    self.global_graph_width,
                    need_scale
                    )
        
        self.t2m = SelfAttentionWithKeyFCLayer(
                    in_channels,
                    self.global_graph_width,
                    need_scale
                    )
        
        self.m2a = SelfAttentionWithKeyFCLayer(
                    in_channels,
                    self.global_graph_width,
                    need_scale
                    )

        self.a2m = SelfAttentionWithKeyFCLayer(
                    in_channels,
                    self.global_graph_width,
                    need_scale
                    )      

        self.ln = nn.Sequential(
            nn.Dropout(0.2),
            nn.LayerNorm(self.global_graph_width)
        )
        
    def forward(self, 
                target_emb, 
                agt_emb, 
                map_emb, 
                t2m_mask, 
                t2a_mask,
                a2m_mask,
                m2a_mask
                ):
        
        agt_emb2 = self.a2m(agt_emb, map_emb, a2m_mask)
        map_emb2 = self.m2a(map_emb, agt_emb, m2a_mask)
        target_emb2 = self.t2a(target_emb, agt_emb2, t2a_mask)
        target_emb2 = self.t2m(target_emb2, map_emb2, t2m_mask)
        
        target_emb = target_emb + target_emb2
        agt_emb = agt_emb + agt_emb2
        map_emb = map_emb + map_emb2
        
        target_emb = self.ln(target_emb)
        agt_emb = self.ln(agt_emb)
        map_emb = self.ln(map_emb)
        
        return target_emb , agt_emb, map_emb
    
class Fusion(nn.Module):
    
    def __init__(
            self,
            in_channels,
            global_graph_width,
            num_global_layers=1,
            need_scale=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.global_graph_width = global_graph_width

        self.layers = nn.ModuleList(
            [
                FusionLayer(
                    in_channels,
                    self.global_graph_width,
                    need_scale
                )
                for _ in range(num_global_layers)
            ]
        )

    def forward(self, 
                target_emb, 
                agt_emb, 
                map_emb, 
                t2m_mask=None, 
                t2a_mask=None,
                a2m_mask=None,
                m2a_mask=None
                ):
        
        for layer in self.layers:
            target_emb, agt_emb, map_emb = layer(target_emb, agt_emb, map_emb, t2m_mask, t2a_mask, a2m_mask, m2a_mask)

        return target_emb, agt_emb, map_emb