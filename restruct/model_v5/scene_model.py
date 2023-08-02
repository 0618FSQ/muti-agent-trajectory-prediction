
import torch
from torch import nn
from torch.nn import functional as F

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))


from .layers import CrossEncoder
from .vectornet import VectorNet
from .vectornet import FourierEmbedding, FourierEmbedding2
from .layers import Dense
from .decoder import Decoder
from .layers import Fusion
from .layers import MultyHeadAttn
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
        
        self.target_agent_trajectory_embedding = FourierEmbedding(
            input_size=agent_input_size,
            hidden_size=hidden_size,
            freq_bands=hidden_size * 2
        )
        
        self.agents_trajectory_embedding = FourierEmbedding2(
            input_size=agent_input_size,
            hidden_size=hidden_size,
            freq_bands=hidden_size * 2
        )

        self.map_feature_embedding = FourierEmbedding2(
            input_size=map_input_size,
            hidden_size=hidden_size,
            freq_bands=hidden_size * 2
        )

        self.map_feature_vectornet = VectorNet(
            in_channels=hidden_size,
            hidden_size=hidden_size,
            sub_layers=sub_layers
        )

        self.target_agent_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dropout=0.1,
            batch_first=True
        ), sub_layers
        )
        
        self.agent_feature_trajectory_vectornet = VectorNet(
            in_channels=hidden_size,
            hidden_size=hidden_size,
            sub_layers=sub_layers
        )
   
        self.t2a = nn.ModuleList(
            [
                MultyHeadAttn(
                    d_model=hidden_size, num_heads=8,
                )
                for _ in range(sub_layers)

            ]
        )
        
        self.t2m = nn.ModuleList(
            [
                MultyHeadAttn(
                    d_model=hidden_size, num_heads=8,
                )
                for _ in range(sub_layers)

            ]
        )
        
        self.a2m = nn.ModuleList(
            [
                MultyHeadAttn(
                    d_model=hidden_size, num_heads=8,
                )
                for _ in range(sub_layers)

            ]
        )
        
        
        self.m2a = nn.ModuleList(
            [
                MultyHeadAttn(
                    d_model=hidden_size, num_heads=num_heads,
                )
                for _ in range(sub_layers)

            ]
        )
        
        
        # self.num_heads = num_heads
        
        self.fusion = Fusion(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )
        
        self.decoder = Decoder(hidden_size=hidden_size,
                               num_modes=6)
        
    def forward(
            self,
            target_agent_history_trajectory,
            
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
            
            position_emb
    ):
        # [b,m, line, hidden_size]
        map_embedding = self.map_feature_embedding(map_feature)
        # [b,m, hidden_size]
        map_embedding = self.map_feature_vectornet(map_embedding, map_feature_mask, map_feature_cross_mask)
        # [b,t, line, hidden_size]
        target_agent_feature_embedding = self.target_agent_trajectory_embedding(target_agent_history_trajectory) + position_emb.squeeze(0)
        # [b,t, hidden_size]
        target_agent_feature_embedding = self.target_agent_encoder(target_agent_feature_embedding)
        # [b,a, line, hidden_size]
        agents_feature_embedding = self.agents_trajectory_embedding(agents_history_trajectory) + position_emb
        # [b,t, hidden_size]
        agents_feature_embedding = self.agent_feature_trajectory_vectornet(agents_feature_embedding, agents_history_mask, agents_history_cross_mask)
        
        # for a2m in self.a2m:
        #     agents_feature_embedding = a2m(agents_feature_embedding, map_embedding,  map_embedding, a2m_cross_mask)

        # for m2a in self.m2a:
        #     map_embedding = m2a(map_embedding, agents_feature_embedding,  agents_feature_embedding, m2a_cross_mask)
        
        # for t2a in self.t2a:
        #     target_agent_feature_embedding = t2a(target_agent_feature_embedding, agents_feature_embedding,  agents_feature_embedding, t2a_cross_mask)
        
        # for t2m in self.t2m:
        #     target_agent_feature_embedding = t2m(target_agent_feature_embedding, map_embedding,  map_embedding, t2m_cross_mask)
        
        for i in range(len(self.a2m)):
            agents_feature_embedding = self.a2m[i](agents_feature_embedding, map_embedding,  map_embedding, a2m_cross_mask)
            map_embedding = self.m2a[i](map_embedding, agents_feature_embedding,  agents_feature_embedding, m2a_cross_mask)
            target_agent_feature_embedding = self.t2a[i](target_agent_feature_embedding, agents_feature_embedding,  agents_feature_embedding, t2a_cross_mask)
            target_agent_feature_embedding = self.t2m[i](target_agent_feature_embedding, map_embedding,  map_embedding, t2m_cross_mask)
        
        # target_agent_feature_embedding, agents_feature_embedding, map_embedding = \
        #     self.fusion(
        #         target_agent_feature_embedding,
        #         agents_feature_embedding,
        #         map_embedding,
        #         t2m_cross_mask,
        #         t2a_cross_mask,
        #         a2m_cross_mask,
        #         m2a_cross_mask
        #     )
        
        pred_trajs, probs = self.decoder(target_agent_feature_embedding)

        return pred_trajs, probs
    
