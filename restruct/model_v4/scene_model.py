import torch
from torch import nn
from torch.nn import functional as F

from .layers import CrossEncoder
from .vectornet import VectorNet
from .layers import Dense
from .layers import Fusion
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

        self.fusion = Fusion(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sub_layers,
            need_scale=True
        )

        self.position_emb = nn.Parameter(
            torch.randn([1, obs_horizon, hidden_size], dtype=torch.float32)
        )

        self.decoder = Decoder(hidden_size,
                 num_modes=6,
                 obs_horizon=obs_horizon,
                 future_horizon=future_horizon,
                 sublayers=3)

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
            map_feature_cross_mask
    ):
        # [b,m, line, hidden_size]
        map_embedding = self.map_feature_embedding(map_feature)
        # [b,m, hidden_size]
        map_embedding = self.map_feature_vectornet(map_embedding, map_feature_mask, map_feature_cross_mask)
        # [b,t, line, hidden_size]
        target_agent_feature_embedding = self.target_agent_trajectory_embedding(
            target_agent_history_trajectory) + self.position_emb
        # [b,t, hidden_size]
        target_agent_feature_embedding = self.target_agent_encoder(target_agent_feature_embedding)
        # [b,a, line, hidden_size]
        agents_feature_embedding = self.agents_trajectory_embedding(agents_history_trajectory) + self.position_emb.unsqueeze(dim=0)
        # [b,t, hidden_size]
        agents_feature_embedding = self.agent_feature_trajectory_vectornet(agents_feature_embedding,
                                                                           agents_history_mask,
                                                                           agents_history_cross_mask)

        target_agent_feature_embedding, agents_feature_embedding, map_embedding = \
            self.fusion(
                target_agent_feature_embedding,
                agents_feature_embedding,
                map_embedding,
                t2m_cross_mask,
                t2a_cross_mask,
                a2m_cross_mask,
                m2a_cross_mask
            )

        pred_trajs, probs = self.decoder(target_agent_feature_embedding)

        return pred_trajs, probs

if __name__ == '__main__':
    def randbool(size):
        return torch.randint(2, size) == torch.randint(2, size)


    target_agent_history_trajectory = torch.randn([32, 20, 7], dtype=torch.float32)

    agents_history_trajectory = torch.randn([32, 18, 20, 7], dtype=torch.float32)
    agents_history_mask = randbool([32, 18, 20])
    agents_history_cross_mask = randbool([32, 18, 18])

    t2a_cross_mask = randbool([32, 20, 18])
    t2m_cross_mask = randbool([32, 20, 80])
    a2m_cross_mask = randbool([32, 18, 80])
    m2a_cross_mask = randbool([32, 80, 18])

    map_feature = torch.randn([32, 80, 10, 7], dtype=torch.float32)
    map_feature_mask = randbool([32, 80, 10])
    map_feature_cross_mask = randbool([32, 80, 80])

    model = SceneModel(
        agent_input_size=7,
        map_input_size=7,
        orig_size=2,
        hidden_size=64,
        obs_horizon=20,
        future_horizon=30,
        sub_layers=1
    )
    print(
        model(
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
            map_feature_cross_mask
        )
    )