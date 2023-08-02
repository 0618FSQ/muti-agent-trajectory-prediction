from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from model_v3.scene_model import SceneModel





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
        pred_horizon=30,
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