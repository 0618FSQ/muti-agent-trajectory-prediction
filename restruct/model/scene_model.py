
import torch
from torch import nn
from torch.nn import functional as F

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))


from .layers import CrossEncoder
from .vectornet import VectorNet
from .layers import Dense
from .decoder import Decoder
import numpy as np

class SceneModel(nn.Module):
    def __init__(
            self,
            agent_input_size=7,
            map_input_size=11,
            orig_size=2,
            hidden_size=64,
            obs_horizon=20,
            future_horizon=30,
            num_heads=8,
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

        self.map_feature_vectornet = VectorNet(
            in_channels=hidden_size,
            hidden_size=hidden_size,
            sub_layers=sub_layers
        )
        
        self.traget_agent_trajectory_vectornet = VectorNet(
            in_channels=hidden_size,
            hidden_size=hidden_size,
            sub_layers=sub_layers
        )
        
        self.agent_feature_trajectory_vectornet = VectorNet(
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
        
        self.a2m = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_size, num_heads=8, dropout=0.2, batch_first=True
                )
                for _ in range(sub_layers)

            ]
        )
        
        
        self.m2a = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_size, num_heads=8, dropout=0.2, batch_first=True
                )
                for _ in range(sub_layers)

            ]
        )
        self.num_heads = num_heads
        
        # self.a2m = CrossEncoder(
        #     in_channels=hidden_size,
        #     global_graph_width=hidden_size,
        #     num_global_layers=sub_layers,
        #     need_scale=True
        # )
        
        # self.m2a = CrossEncoder(
        #     in_channels=hidden_size,
        #     global_graph_width=hidden_size,
        #     num_global_layers=sub_layers,
        #     need_scale=True
        # )
        
        self.decoder = Decoder(hidden_size=hidden_size,
                               num_modes=6)
        
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
            position_emb
    ):
        # [b,m, line, hidden_size]
        map_embedding = self.map_feature_embedding(map_feature)
        # [b,m, hidden_size]
        map_embedding = self.map_feature_vectornet(map_embedding, map_feature_mask, map_feature_cross_mask)
        # [b,t, line, hidden_size]
        target_agent_feature_embedding = self.target_agent_trajectory_embedding(target_agent_history_trajectory) + position_emb
        # [b,t, hidden_size]
        target_agent_feature_embedding = self.traget_agent_trajectory_vectornet(target_agent_feature_embedding, target_agent_history_mask, target_agent_history_cross_mask)
        # [b,a, line, hidden_size]
        agents_feature_embedding = self.agents_trajectory_embedding(agents_history_trajectory) + position_emb
        # [b,t, hidden_size]
        agents_feature_embedding = self.agent_feature_trajectory_vectornet(agents_feature_embedding, agents_history_mask, agents_history_cross_mask)
        
        for a2m in self.a2m:
            agents_feature_embedding,_ = a2m(agents_feature_embedding, map_embedding,  map_embedding, att_mask = torch.repeat_interleave(t2a_cross_mask, self.num_heads, 0))
        
        for m2a in self.m2a:
            map_embedding ,_= m2a(map_embedding, agents_feature_embedding,  agents_feature_embedding, att_mask = torch.repeat_interleave(m2a_cross_mask, self.num_heads, 0))
        
        # agents_feature_embedding = self.a2m(agents_feature_embedding, map_embedding, a2m_cross_mask)
        # map_embedding = self.m2a(map_embedding, agents_feature_embedding, m2a_cross_mask)
        
        target_agent_feature_embedding = self.t2a(target_agent_feature_embedding, agents_feature_embedding, t2a_cross_mask)
        target_agent_feature_embedding = self.t2m(target_agent_feature_embedding, map_embedding, t2m_cross_mask)
        
        pred_trajs, probs = self.decoder(target_agent_feature_embedding)

        return pred_trajs, probs
    
