import math
import torch
from torch import nn
from torch.nn import functional as F

from .layers import CrossEncoder
from .vectornet import VectorNet
from .vectornet import SpatioTemporalNet
from .layers import Dense

class FutureCrossEncoder(nn.Module):
    def __init__(self, hidden_size, sub_layers):
        super().__init__()
        self.future_vector_net = SpatioTemporalNet(
            in_channels=hidden_size,
            hidden_size=hidden_size,
            sub_layers=sub_layers
        )
        self.f2h = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )
        self.f2a = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )
        self.f2m = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )
    
        self.layernorm = nn.LayerNorm(hidden_size)
        
    def forward(self,
                displacement_feature,
                target_agent_orig_embedding,
                target_agent_history_feature, 
                map_features,
                agent_features,
                
                f2h_cross_mask,
                f2a_cross_mask,
                f2m_cross_mask,
                future_mask,
                future_cross_mask
                ):
        # future_traj_feature = torch.cumsum(displacement_feature, dim=-2) + target_agent_orig_embedding.unsqueeze(dim=-2)
        future_traj_feature = displacement_feature + target_agent_orig_embedding.unsqueeze(dim=-2)
        
        future_traj_feature = self.future_vector_net(future_traj_feature, future_mask, future_cross_mask)
        future_traj_feature = self.f2h(future_traj_feature, target_agent_history_feature, f2h_cross_mask)
        future_traj_feature = self.f2a(future_traj_feature, agent_features, f2a_cross_mask)
        future_traj_feature = self.f2m(future_traj_feature, map_features, f2m_cross_mask)
        output = future_traj_feature.unsqueeze(dim=-2) + displacement_feature
        output = self.layernorm(output)
        return output
        

class SceneModel(nn.Module):
    def __init__(
            self,
            agent_input_size=7,
            map_input_size=11,
            orig_size=2,
            hidden_size=64,
            obs_horizon=20,
            future_horizon=30,
            sub_layers=1
    ):
        super().__init__()
        
        self.orig_embedding = nn.Sequential(
            Dense(orig_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size)
        )
        
        self.target_agent_trajectory_embedding = nn.Sequential(
            Dense(agent_input_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size)
        )
        
        self.agents_trajectory_embedding = nn.Sequential(
            Dense(agent_input_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size)
        )

        self.map_feature_embedding = nn.Sequential(
            Dense(map_input_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size)
        )

        self.position_encoder = nn.Parameter(
            torch.randn(1, 1, obs_horizon, hidden_size)
        )

        self.future_position_encoder = nn.Parameter(
            torch.randn(1, 1, future_horizon, hidden_size)
        )


        self.pred_dense = nn.Sequential(
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            nn.Linear(in_features=hidden_size, out_features=2)
        )

        self.map_feature_vectornet = VectorNet(
            in_channels=hidden_size,
            hidden_size=hidden_size,
            sub_layers=sub_layers
        )
        
        self.traget_agent_trajectory_vectornet = SpatioTemporalNet(
            in_channels=hidden_size,
            hidden_size=hidden_size,
            sub_layers=sub_layers
        )
        
        self.agent_feature_trajectory_vectornet = SpatioTemporalNet(
            in_channels=hidden_size,
            hidden_size=hidden_size,
            sub_layers=sub_layers
        )

        
        self.t2a = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )
        
        self.t2m = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )
        
        self.a2m = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )
        
        self.m2a = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )
        
        self.future_cross_encoder = nn.ModuleList(
            
            [
                FutureCrossEncoder(hidden_size, sub_layers)
                for _ in range(sub_layers)
            ]
        )


    def forward(
            self,
            target_agent_history_trajectory,
            target_agent_history_mask,
            target_agent_history_cross_mask,
            
            agents_history_trajectory,
            agents_history_mask,
            agents_history_cross_mask,
            
            t2a_cross_mask,
            t2m_cross_mask,
            a2m_cross_mask,
            m2a_cross_mask,
            
            map_feature,
            map_feature_mask,
            map_feature_cross_mask,
            
            target_agent_orig,
            
            future_mask,
            future_cross_mask,
            
            f2h_cross_mask,
            f2a_cross_mask,
            f2m_cross_mask
    ):
        # [b,m, line, hidden_size]
        map_embedding = self.map_feature_embedding(map_feature)
        # [b,m, hidden_size]
        map_embedding = self.map_feature_vectornet(map_embedding, map_feature_mask, map_feature_cross_mask)
        # [b,t, line, hidden_size]
        target_agent_feature_embedding = self.target_agent_trajectory_embedding(target_agent_history_trajectory) + self.position_encoder
        # [b,t, hidden_size]
        target_agent_feature_embedding = self.traget_agent_trajectory_vectornet(target_agent_feature_embedding, target_agent_history_mask, target_agent_history_cross_mask)
        # [b,a, line, hidden_size]
        agents_feature_embedding = self.agents_trajectory_embedding(agents_history_trajectory) + self.position_encoder
        # [b,t, hidden_size]
        agents_feature_embedding = self.agent_feature_trajectory_vectornet(agents_feature_embedding, agents_history_mask, agents_history_cross_mask)
        
        agents_feature_embedding = self.a2m(agents_feature_embedding, map_embedding, a2m_cross_mask)
        map_embedding = self.m2a(map_embedding, agents_feature_embedding, m2a_cross_mask)
        
        target_agent_feature_embedding = self.t2a(target_agent_feature_embedding, agents_feature_embedding, t2a_cross_mask)
        target_agent_feature_embedding = self.t2m(target_agent_feature_embedding, map_embedding, t2m_cross_mask)
        
        displacement_feature = target_agent_feature_embedding.unsqueeze(dim=-2) + self.future_position_encoder# [bs, agents, future_horizon, hidden_size]
        
        target_agent_orig_embedding = self.orig_embedding(target_agent_orig)
        
        for layer in self.future_cross_encoder:
            displacement_feature = layer(
                displacement_feature,
                target_agent_orig_embedding,
                target_agent_history_feature=target_agent_feature_embedding, 
                map_features=map_embedding,
                agent_features=agents_feature_embedding,
                
                f2h_cross_mask=f2h_cross_mask,
                f2a_cross_mask=f2a_cross_mask,
                f2m_cross_mask=f2m_cross_mask,
                future_mask=future_mask,
                future_cross_mask=future_cross_mask
            )
        future_traj = self.pred_dense(displacement_feature)
        return future_traj
