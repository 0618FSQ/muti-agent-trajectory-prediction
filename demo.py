import math
import torch
import torch.nn as nn
import numpy as np


class MultyHeadAttn(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2):
        super().__init__()
        assert d_model % num_heads == 0
        self.q_lin = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.k_lin = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.v_lin = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.layernorm = nn.LayerNorm(d_model // num_heads)
        self.scaled = d_model // num_heads
        self.num_heads = num_heads

    @classmethod
    def masked_softmax(cls, x, mask):
        if mask is not None:
            x_masked = torch.masked_fill(x, mask, -1e12)
            return nn.functional.softmax(x_masked, dim=-1) * (1 - mask.float())
        else:
            return nn.functional.softmax(x, dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        q: [b, q, h]
        k: [b, k, h]
        v: [b. k, h]
        attn_mask: [b, q, k]
        """
        query = self.q_lin(q)
        key = self.k_lin(k)
        value = self.v_lin(v)

        k_size = k.size(1)
        q_size = q.size(1)

        query = query.view(-1, q_size, self.scaled)
        key = key.view(-1, k_size, self.scaled)
        value = value.view(-1, k_size, self.scaled)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.scaled)
        attention_weights = self.masked_softmax(scores, torch.repeat_interleave(mask, self.num_heads, 0))
        x = torch.matmul(attention_weights, value)
        x = self.layernorm(x)
        x = x.view(-1, q_size, self.scaled * self.num_heads)
        output = q + x
        return output



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


class TrajectoryEncoder(nn.Module):
    def __init__(self, hidden_size, sub_layers):
        super().__init__()
        self.trajectory_encoder = nn.ModuleList(
            [
                TrajectoryEncoderSubLayer(hidden_size=hidden_size)
                for _ in range(sub_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        output = x
        for trajectory_encoder_layer in self.trajectory_encoder:
            output = trajectory_encoder_layer(output, mask)
        output = self.layer_norm(output) + x
        return output


class TrajectoryEncoderSubLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
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
        output = self.layernorm(output) + x
        return output


class VectorNet(nn.Module):
    
    def __init__(self,
                 in_channels=8,
                 hidden_size=64,
                 sub_layers=3,
                 num_heads=4,
                 freq_bands=12,
                 return_local=False
                 ):
        super().__init__()
        self.local_embedding = FourierEmbedding(
            input_size=in_channels, hidden_size=hidden_size, freq_bands=freq_bands
        )
        self.global_embedding = FourierEmbedding2(
            input_size=in_channels, hidden_size=hidden_size, freq_bands=freq_bands
        )

        self.traj_encoder = TrajectoryEncoder(
            hidden_size, sub_layers
        )
        self.traj_dense = nn.Sequential(
            FourierEmbedding2(input_size=hidden_size, hidden_size=hidden_size, freq_bands=freq_bands),
            nn.ReLU()
        )

        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "global_graph": MultyHeadAttn(d_model=hidden_size, num_heads=num_heads, dropout=0.2),
                    "local_graph": MultyHeadAttn(d_model=hidden_size, num_heads=num_heads, dropout=0.2),
                    "x_2_y_cross_graph": MultyHeadAttn(d_model=hidden_size, num_heads=num_heads, dropout=0.2),
                    "y_2_x_cross_graph": MultyHeadAttn(d_model=hidden_size, num_heads=num_heads, dropout=0.2)
                }
            )
            for _ in range(sub_layers)
        )

        self.return_local = return_local
        self.num_heads = num_heads

    @staticmethod
    def max_pooling(x, agt_mask=None):
        if agt_mask is not None:
            x_masked = torch.max(torch.masked_fill(x, ~agt_mask.unsqueeze(dim=-1), 0), dim=-2)[0]
            return x_masked
        else:
            return torch.max(x, dim=-2)[0]

    def forward(self, global_padding, local_padding, mask=None, valid_mask=None):

        global_embedding = self.global_embedding(global_padding)
        local_embedding = self.local_embedding(local_padding)

        global_embedding = self.traj_encoder(x=global_embedding, mask=mask)
        global_embedding = self.traj_dense(global_embedding)

        global_embedding = self.max_pooling(global_embedding, mask)

        for layer in self.layers:
            global_embedding = layer["global_graph"](global_embedding, global_embedding, global_embedding, valid_mask)
            local_embedding = layer["local_graph"](local_embedding, local_embedding, local_embedding, valid_mask)
            local_embedding = layer["x_2_y_cross_graph"](local_embedding, global_embedding, global_embedding,
                                                         valid_mask)
            global_embedding = layer["y_2_x_cross_graph"](global_embedding, local_embedding, local_embedding,
                                                          valid_mask)

        if self.return_local:
            return global_embedding, local_embedding

        return global_embedding


class VectorNet2(nn.Module):
    def __init__(self,
                 hidden_size=64,
                 sub_layers=3,
                 num_heads=4,
                 return_local=False
                 ):
        super().__init__()
        self.local_embedding = nn.Sequential(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            ),
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            ),
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        )
        self.global_embedding = nn.Sequential(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            ),
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            ),
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
        )

        self.traj_encoder = TrajectoryEncoder(
            hidden_size, sub_layers
        )
        self.traj_dense = nn.Sequential(
            nn.Sequential(
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                ),
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                ),
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            )
        )

        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "global_graph": MultyHeadAttn(d_model=hidden_size, num_heads=num_heads, dropout=0.2),
                    "local_graph": MultyHeadAttn(d_model=hidden_size, num_heads=num_heads, dropout=0.2),
                    "x_2_y_cross_graph": MultyHeadAttn(d_model=hidden_size, num_heads=num_heads, dropout=0.2),
                    "y_2_x_cross_graph": MultyHeadAttn(d_model=hidden_size, num_heads=num_heads, dropout=0.2)
                }
            )
            for _ in range(sub_layers)
        )

        self.return_local = return_local
        self.num_heads = num_heads

    @staticmethod
    def max_pooling(x, agt_mask=None):
        if agt_mask is not None:
            x_masked = torch.max(torch.masked_fill(x, ~agt_mask.unsqueeze(dim=-1), 0), dim=-2)[0]
            return x_masked
        else:
            return torch.max(x, dim=-2)[0]

    def forward(self, global_padding, local_padding, mask=None, valid_mask=None):

        global_embedding = self.global_embedding(global_padding)
        local_embedding = self.local_embedding(local_padding)

        global_embedding = self.traj_encoder(x=global_embedding, mask=mask)
        global_embedding = self.traj_dense(global_embedding)

        global_embedding = self.max_pooling(global_embedding, mask)

        for layer in self.layers:
            global_embedding = layer["global_graph"](global_embedding, global_embedding, global_embedding, valid_mask)
            local_embedding = layer["local_graph"](local_embedding, local_embedding, local_embedding, valid_mask)
            local_embedding = layer["x_2_y_cross_graph"](local_embedding, global_embedding, global_embedding,
                                                         valid_mask)
            global_embedding = layer["y_2_x_cross_graph"](global_embedding, local_embedding, local_embedding,
                                                          valid_mask)

        if self.return_local:
            return global_embedding, local_embedding

        return global_embedding



class FutureCrossEncoder(nn.Module):
    def __init__(self, hidden_size, sublayers, num_heads, obs_horizon):
        super().__init__()

        self.cross_encoder = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "future_vectornet": VectorNet2(
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
            traj_feature = cross_encoder["future_vectornet"](traj_feature,
                                                             local_traj_padding,
                                                             mask,
                                                             valid_mask)
            traj_feature = cross_encoder["f2m_cross_encoder"](
                traj_feature,
                map_feature,
                map_feature,
                mask=f2m_cross_mask
            )
            traj_feature = cross_encoder["f2a_cross_encoder"](
                traj_feature,
                agent_feature,
                agent_feature,
                mask=f2a_cross_mask
            )
            traj_feature = cross_encoder["f2h_cross_encoder"](
                traj_feature,
                global_traj_padding,
                global_traj_padding,
                mask=f2h_cross_mask
            )

        traj_feature = self.layernorm(traj_feature) + global_traj_padding

        return traj_feature


class Predictor(nn.Module):
    def __init__(self, hidden_size, sublayers, num_heads, obs_horizon):
        super().__init__()
        self.future_cross_encoder = FutureCrossEncoder(
            hidden_size, sublayers, num_heads, obs_horizon
        )
        self.position_embedding = nn.Parameter(torch.randn(1, 1, obs_horizon, hidden_size))
        self.predict_dense = nn.Sequential(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            ),
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            ),
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            ),
            nn.Linear(hidden_size, 2)
        )

    def forward(self,
                global_traj_padding,
                local_traj_padding,
                mask,
                cross_mask,
                map_feature,
                agent_feature,
                f2h_cross_mask,
                f2a_cross_mask,
                f2m_cross_mask):
        traj_feature = self.future_cross_encoder(global_traj_padding,
                                                                     local_traj_padding,
                                                                     mask,
                                                                     cross_mask,
                                                                     map_feature,
                                                                     agent_feature,
                                                                     f2h_cross_mask,
                                                                     f2a_cross_mask,
                                                                     f2m_cross_mask)
        traj_feature = traj_feature.unsqueeze(-2) + self.position_embedding
        predict_traj = self.predict_dense(traj_feature)
        return predict_traj


class Encoder(nn.Module):
    def __init__(self,
                 agent_feature_size,
                 map_feature_size,
                 sublayer,

                 hidden_size=64,
                 num_heads=4

                 ):
        super().__init__()
        self.env_agents_vectornet = VectorNet(
            in_channels=agent_feature_size,
            hidden_size=hidden_size,
            sub_layers=sublayer,
            num_heads=num_heads
        )
        self.target_agent_vectornet = VectorNet(
            in_channels=agent_feature_size,
            hidden_size=hidden_size,
            sub_layers=sublayer,
            num_heads=num_heads,
            return_local=True
        )
        self.map_vectornet = VectorNet(
            in_channels=map_feature_size,
            hidden_size=hidden_size,
            sub_layers=sublayer,
            num_heads=num_heads
        )

        self.cross_encoder = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "a2m": MultyHeadAttn(d_model=hidden_size, num_heads=num_heads),
                        "m2a": MultyHeadAttn(d_model=hidden_size, num_heads=num_heads),
                        "t2m": MultyHeadAttn(d_model=hidden_size, num_heads=num_heads),
                        "t2a": MultyHeadAttn(d_model=hidden_size, num_heads=num_heads)
                    }
                )
                for _ in range(sublayer)
            ]
        )

    def forward(self,

                map_feature,
                map_local_feature,
                map_feature_mask,
                map_feature_cross_mask,

                agent_feature,
                agent_local_feature,
                agent_feature_mask,
                agent_feature_cross_mask,

                target_agent_feature,
                target_agent_local_feature,
                target_agent_feature_mask,
                target_agent_feature_cross_mask,

                a2m_mask,
                m2a_mask,
                t2a_mask,
                t2m_mask

                ):
        map_embedding = self.map_vectornet(map_feature,
                                           map_local_feature,
                                           map_feature_mask,
                                           map_feature_cross_mask)
        agent_embedding = self.env_agents_vectornet(agent_feature,
                                                    agent_local_feature,
                                                    agent_feature_mask,
                                                    agent_feature_cross_mask)

        target_agent_embedding, target_local_embedding = self.target_agent_vectornet(target_agent_feature,
                                                                                     target_agent_local_feature,
                                                                                     target_agent_feature_mask,
                                                                                     target_agent_feature_cross_mask)
        for cross_encoder in self.cross_encoder:
            agent_embedding = cross_encoder["a2m"](agent_embedding,
                                                   map_embedding,
                                                   map_embedding,
                                                   a2m_mask)
            map_embedding = cross_encoder["m2a"](map_embedding,
                                                 agent_embedding,
                                                 agent_embedding,
                                                 m2a_mask)

            target_agent_embedding = cross_encoder["t2m"](target_agent_embedding,
                                                          map_embedding,
                                                          map_embedding,
                                                          t2m_mask)
            target_agent_embedding = cross_encoder["t2a"](target_agent_embedding,
                                                          agent_embedding,
                                                          agent_embedding,
                                                          t2a_mask)
        return target_agent_embedding, target_local_embedding, map_embedding, agent_embedding


class Model(nn.Module):
    def __init__(self,
                 agent_feature_size,
                 map_feature_size,
                 sublayer,
                 obs_horizon=30,
                 hidden_size=64,
                 num_heads=4
                 ):
        super(Model, self).__init__()
        self.encoder = Encoder(agent_feature_size,
                               map_feature_size,
                               sublayer,

                               hidden_size,
                               num_heads)
        self.predictor = Predictor(hidden_size,
                                   sublayer,
                                   num_heads,
                                   obs_horizon
                                   )

    def forward(self,
                map_feature,
                map_local_feature,
                map_feature_mask,
                map_feature_cross_mask,

                agent_feature,
                agent_local_feature,
                agent_feature_mask,
                agent_feature_cross_mask,

                target_agent_feature,
                target_agent_local_feature,
                target_agent_feature_mask,
                target_agent_feature_cross_mask,

                a2m_mask,
                m2a_mask,
                t2a_mask,
                t2m_mask,
                future_mask,
                future_cross_mask,
                f2h_cross_mask,
                f2a_cross_mask,
                f2m_cross_mask
                ):
        target_agent_embedding, target_local_embedding, map_embedding, agent_embedding = self.encoder(
            map_feature,
            map_local_feature,
            map_feature_mask,
            map_feature_cross_mask,

            agent_feature,
            agent_local_feature,
            agent_feature_mask,
            agent_feature_cross_mask,

            target_agent_feature,
            target_agent_local_feature,
            target_agent_feature_mask,
            target_agent_feature_cross_mask,

            a2m_mask,
            m2a_mask,
            t2a_mask,
            t2m_mask
        )
        future_traj = self.predictor(
            target_agent_embedding,
            target_local_embedding,
            future_mask,
            future_cross_mask,
            map_embedding,
            agent_embedding,
            f2h_cross_mask,
            f2a_cross_mask,
            f2m_cross_mask
        )
        return future_traj


if __name__ == '__main__':
    inputs = torch.randn(64, 20, 10, 15)
    net = FourierEmbedding2(15, 128, 256)
    print(net(inputs))

    # inputs = torch.randn(64, 20, 15)
    # net = FourierEmbedding(15, 128, 256)
    # print(net(inputs))

    vector_net = VectorNet(
        in_channels=15,
        hidden_size=64,
        sub_layers=3,
        num_heads=4,
        return_local=False
    )
    vector_net(
        global_padding=inputs,
        local_padding=torch.randn(64, 20, 15),
        mask=torch.randint(0, 2, size=[64, 20, 10]).bool(),
        valid_mask=torch.randint(0, 2, size=[64, 20, 20]).bool()
    )
