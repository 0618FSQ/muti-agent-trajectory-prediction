import math
import torch
from torch import nn
from torch.nn import functional as F

from .layers import CrossEncoder
from .vectornet import VectorNet
from .layers import Dense
from .vectornet import SpatioTemporalNet


class EncoderLayer(nn.Module):
    def __init__(
            self,
            agent_size,
            map_size,
            hidden_size,
            sublayer,
            obs_horizon

    ):
        super().__init__()

        self.agents_dense = nn.Sequential(
            Dense(agent_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.map_dense = nn.Sequential(
            Dense(map_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.agents_encoder_net = SpatioTemporalNet(
            in_channels=hidden_size,
            hidden_size=hidden_size,
            sub_layers=sublayer
        )

        self.map_encoder_net = VectorNet(
            in_channels=hidden_size,
            hidden_size=hidden_size,
            sub_layers=sublayer
        )

        self.trajectory_embedding = nn.Sequential(
            Dense(agent_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.trajectory_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dropout=0.1, batch_first=True, norm_first=False
        )

        self.t2a = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sublayer
        )

        self.t2m = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sublayer
        )

        self.a2m = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sublayer
        )

        self.m2a = CrossEncoder(
            in_channels=hidden_size,
            global_graph_width=hidden_size,
            num_global_layers=sublayer
        )

        self.position_encoder = nn.Parameter(
            torch.randn(1, obs_horizon, hidden_size)
        )

    def forward(
            self,
            target_agent_trajectory,

            env_agent_trajectory,
            env_agent_mask,
            env_agent_att_mask,

            map_feature,
            map_feature_mask,
            map_feature_att_mask,

            t2m_mask,
            t2a_mask,

            a2m_mask,
            m2a_mask

    ):
        target_agent_feature = self.trajectory_embedding(target_agent_trajectory) + self.position_encoder
        target_agent_feature = self.trajectory_encoder(target_agent_feature)

        env_agent_feature = self.agents_dense(env_agent_trajectory) + self.position_encoder.unsqueeze(dim=0)
        env_agent_feature = self.agents_encoder_net(env_agent_feature, env_agent_mask, env_agent_att_mask)

        map_feature = self.map_dense(map_feature)
        map_feature = self.map_encoder_net(map_feature, map_feature_mask, map_feature_att_mask)

        map_feature = self.m2a(map_feature, env_agent_trajectory, m2a_mask)
        env_agent_feature = self.a2m(env_agent_feature, map_feature, a2m_mask)

        target_agent_feature = self.t2a(target_agent_feature, env_agent_feature, t2m_mask)
        target_agent_feature = self.t2m(target_agent_feature, map_feature, t2a_mask)
        return target_agent_feature


class DecoderLayer(nn.Module):
    def __init__(self, agent_size, hidden_size, sublayer, pred_horizon):
        super().__init__()
        self.dense = nn.Sequential(
            Dense(agent_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=pred_horizon + 1, embedding_dim=hidden_size, padding_idx=0
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=8, dropout=0.1, batch_first=True,
                                                   norm_first=True)
        self.transformer_decoder_layer = nn.TransformerDecoder(
            decoder_layer, sublayer
        )
        self.pred_dense = nn.Sequential(
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, hidden_size),
            Dense(hidden_size, 2),
            nn.LeakyReLU()
        )

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
            self,
            target_agent_feature,

            tgt,
            position,
            tgt_mask=None,
            memory=None
    ):
        tgt = self.dense(tgt) + self.position_embedding(position)

        if self.training:
            if tgt_mask is None:
                device = tgt.device
                tgt_mask = self._generate_square_subsequent_mask(tgt).to(device)
            output1 = self.transformer_decoder(tgt, target_agent_feature, tgt_mask)
            output2 = self.pred_dense(output1)
            return output1, output2
        else:
            if memory:
                tgt = torch.cat([memory, tgt], dim=-1)
            device = tgt.device
            tgt_mask = self._generate_square_subsequent_mask(tgt).to(device)
            output1 = self.transformer_decoder(tgt, target_agent_feature, tgt_mask)
            output2 = self.pred_dense(output1)
            return output1, output2[:, :, -1]


class Model(nn.Module):
    def __init__(self,
                 agent_size,
                 map_size,
                 hidden_size,
                 sublayer,
                 obs_horizon,
                 pred_horizon
                 ):
        super(Model, self).__init__()
        self.encoder = EncoderLayer(
            agent_size,
            map_size,
            hidden_size,
            sublayer,
            obs_horizon
        )
        self.decoder = DecoderLayer(
            agent_size, hidden_size, sublayer, pred_horizon
        )
        self.pred_horizon = pred_horizon
        self.agent_size = agent_size

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,

                target_agent_trajectory,

                env_agent_trajectory,
                env_agent_mask,
                env_agent_att_mask,

                map_feature,
                map_feature_mask,
                map_feature_att_mask,

                t2m_mask,
                t2a_mask,

                a2m_mask,
                m2a_mask,
                tgt=None,
                position=None
                ):
        target_agent_feature = self.encoder(target_agent_trajectory,

                                            env_agent_trajectory,
                                            env_agent_mask,
                                            env_agent_att_mask,

                                            map_feature,
                                            map_feature_mask,
                                            map_feature_att_mask,

                                            t2m_mask,
                                            t2a_mask,

                                            a2m_mask,
                                            m2a_mask)
        if self.training:
            _, output = self.decoder(target_agent_feature, tgt, position)
            return output
        else:
            bs = target_agent_trajectory.shape[0]
            tgt = torch.zeros([bs, 1, self.agent_size])
            output = torch.ones([bs, self.pred_horizon, 2])
            o1 = None
            for i in range(self.pred_horizon):
                position = torch.zeros([bs, 1]) + i
                o1, tgt = self.decoder(target_agent_feature, tgt, position, o1)
                output[:, i, :] = tgt

            return output