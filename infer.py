import torch
import os

import numpy as np
import argparse
import json
from tqdm import tqdm

# from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).resolve().parents[1]))

from demo import Model as SceneModel
from dataset import RoadDataset
from torch.utils.data import DataLoader
from argoverse_data.smooth import Kalman_traj_smooth

def infer(model, test_loader, use_cuda=True):
    
    model.eval()
    res = {}
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            target_agent_history_trajectory=data["target_agent_history_trajectory"]
            target_agent_history_mask=data["target_agent_history_mask"]
            target_agent_history_cross_mask=data["target_agent_history_cross_mask"]

            agents_history_trajectory=data["agents_history_trajectory"]
            agents_history_mask=data["agents_history_mask"]
            agents_history_cross_mask=data["agents_history_cross_mask"]

            t2a_cross_mask=data["t2a_cross_mask"]
            t2m_cross_mask=data["t2m_cross_mask"]
            a2m_cross_mask=data["a2m_cross_mask"]
            m2a_cross_mask=data["m2a_cross_mask"]

            map_feature=data["map_feature"]
            map_feature_mask=data["map_feature_mask"]
            map_feature_cross_mask=data["map_feature_cross_mask"]

            target_agent_local_feature=data["target_agent_local_feature"]
            agent_local_feature=data['agent_local_feature']
            map_local_feature=data['new_map_local_feature']

            future_mask=data["future_mask"]
            future_cross_mask=data["future_cross_mask"]

            f2h_cross_mask=data["f2h_cross_mask"]
            f2a_cross_mask=data["f2a_cross_mask"]
            f2m_cross_mask=data["f2m_cross_mask"]
            
            rot = data["rot"]
            orig = data["orig"]
            id = data["id"]
            if use_cuda:
                target_agent_history_trajectory=target_agent_history_trajectory.cuda()
                target_agent_history_mask=target_agent_history_mask.cuda()
                target_agent_history_cross_mask=target_agent_history_cross_mask.cuda()

                agents_history_trajectory=agents_history_trajectory.cuda()
                agents_history_mask=agents_history_mask.cuda()
                agents_history_cross_mask=agents_history_cross_mask.cuda()

                t2a_cross_mask=t2a_cross_mask.cuda()
                t2m_cross_mask=t2m_cross_mask.cuda()
                a2m_cross_mask=a2m_cross_mask.cuda()
                m2a_cross_mask=m2a_cross_mask.cuda()

                map_feature=map_feature.cuda()
                map_feature_mask=map_feature_mask.cuda()
                map_feature_cross_mask=map_feature_cross_mask.cuda()

                target_agent_local_feature=target_agent_local_feature.cuda()
                agent_local_feature=agent_local_feature.cuda()
                map_local_feature=map_local_feature.cuda()

                future_mask=future_mask.cuda()
                future_cross_mask=future_cross_mask.cuda()

                f2h_cross_mask=f2h_cross_mask.cuda()
                f2a_cross_mask=f2a_cross_mask.cuda()
                f2m_cross_mask=f2m_cross_mask.cuda()
                
                rot = rot.cuda()
                orig = orig.cuda()
                
            out = model(
                map_feature=map_feature,
                map_local_feature=map_local_feature,
                map_feature_mask=map_feature_mask,
                map_feature_cross_mask=map_feature_cross_mask,

                agent_feature=agents_history_trajectory,
                agent_local_feature=agent_local_feature,
                agent_feature_mask=agents_history_mask,
                agent_feature_cross_mask=agents_history_cross_mask,

                target_agent_feature=target_agent_history_trajectory,
                target_agent_local_feature=target_agent_local_feature,
                target_agent_feature_mask=target_agent_history_mask,
                target_agent_feature_cross_mask=target_agent_history_cross_mask,

                a2m_mask=a2m_cross_mask,
                m2a_mask=m2a_cross_mask,
                t2a_mask=t2a_cross_mask,
                t2m_mask=t2m_cross_mask,
                future_mask=future_mask,
                future_cross_mask=future_cross_mask,
                f2h_cross_mask=f2h_cross_mask,
                f2a_cross_mask=f2a_cross_mask,
                f2m_cross_mask=f2m_cross_mask
            )
            
            pred_traj = out[:, 0, :, :]
            real_traj = torch.matmul(pred_traj, rot)
            real_traj = torch.add(orig.unsqueeze(dim=1), real_traj)
            if use_cuda:
                real_traj = real_traj.cpu().detach().numpy()
                # traj_smoothed = Kalman_traj_smooth(group, 
                #         process_noise_std = 0.01, 
                #         measurement_noise_std = 0.7)

                for idx, data_id in enumerate(id):
                    res[int(data_id)] = np.stack([real_traj[idx]] * 6)
        torch.cuda.empty_cache()
    return  res 


@torch.no_grad()
def eval_infer(model, eval_loader, use_cuda=True):

        model.eval()
        
        for i, data in enumerate(eval_loader):
            target_agent_history_trajectory=data["target_agent_history_trajectory"]
            target_agent_history_mask=data["target_agent_history_mask"]
            target_agent_history_cross_mask=data["target_agent_history_cross_mask"]

            agents_history_trajectory=data["agents_history_trajectory"]
            agents_history_mask=data["agents_history_mask"]
            agents_history_cross_mask=data["agents_history_cross_mask"]

            t2a_cross_mask=data["t2a_cross_mask"]
            t2m_cross_mask=data["t2m_cross_mask"]
            a2m_cross_mask=data["a2m_cross_mask"]
            m2a_cross_mask=data["m2a_cross_mask"]

            map_feature=data["map_feature"]
            map_feature_mask=data["map_feature_mask"]
            map_feature_cross_mask=data["map_feature_cross_mask"]

            target_agent_local_feature=data["target_agent_local_feature"]
            agent_local_feature=data['agent_local_feature']
            map_local_feature=data['new_map_local_feature']

            future_mask=data["future_mask"]
            future_cross_mask=data["future_cross_mask"]

            f2h_cross_mask=data["f2h_cross_mask"]
            f2a_cross_mask=data["f2a_cross_mask"]
            f2m_cross_mask=data["f2m_cross_mask"]
            
            y = data["y"]
            location = data["location"]
            y_mask = data["y_mask"]
            
            if use_cuda :
                target_agent_history_trajectory=target_agent_history_trajectory.cuda()
                target_agent_history_mask=target_agent_history_mask.cuda()
                target_agent_history_cross_mask=target_agent_history_cross_mask.cuda()

                agents_history_trajectory=agents_history_trajectory.cuda()
                agents_history_mask=agents_history_mask.cuda()
                agents_history_cross_mask=agents_history_cross_mask.cuda()

                t2a_cross_mask=t2a_cross_mask.cuda()
                t2m_cross_mask=t2m_cross_mask.cuda()
                a2m_cross_mask=a2m_cross_mask.cuda()
                m2a_cross_mask=m2a_cross_mask.cuda()

                map_feature=map_feature.cuda()
                map_feature_mask=map_feature_mask.cuda()
                map_feature_cross_mask=map_feature_cross_mask.cuda()

                target_agent_local_feature=target_agent_local_feature.cuda()
                agent_local_feature=agent_local_feature.cuda()
                map_local_feature=map_local_feature.cuda()

                y = y.cuda()
                location = location.cuda()
                y_mask = y_mask.cuda()
            
                future_mask=future_mask.cuda()
                future_cross_mask=future_cross_mask.cuda()

                f2h_cross_mask=f2h_cross_mask.cuda()
                f2a_cross_mask=f2a_cross_mask.cuda()
                f2m_cross_mask=f2m_cross_mask.cuda()
                
            out = model(
                map_feature=map_feature,
                map_local_feature=map_local_feature,
                map_feature_mask=map_feature_mask,
                map_feature_cross_mask=map_feature_cross_mask,

                agent_feature=agents_history_trajectory,
                agent_local_feature=agent_local_feature,
                agent_feature_mask=agents_history_mask,
                agent_feature_cross_mask=agents_history_cross_mask,

                target_agent_feature=target_agent_history_trajectory,
                target_agent_local_feature=target_agent_local_feature,
                target_agent_feature_mask=target_agent_history_mask,
                target_agent_feature_cross_mask=target_agent_history_cross_mask,

                a2m_mask=a2m_cross_mask,
                m2a_mask=m2a_cross_mask,
                t2a_mask=t2a_cross_mask,
                t2m_mask=t2m_cross_mask,
                future_mask=future_mask,
                future_cross_mask=future_cross_mask,
                f2h_cross_mask=f2h_cross_mask,
                f2a_cross_mask=f2a_cross_mask,
                f2m_cross_mask=f2m_cross_mask
            )

        torch.cuda.empty_cache()
        return   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1,5')
    parser.add_argument('--train_config_path', type=str, default="scene_model_v2.json")
    config = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    train_config = json.load(open(config.train_config_path, "r"))

    use_cuda = True
    # device = torch.device("cuda:3") if use_cuda else torch.device("cpu")

    # 步骤三：模型分布式处理
    # model = SceneModel(**train_config['model'])
    model = SceneModel(
        agent_feature_size=6,
        map_feature_size=6,
        sublayer=3
    )
    model.load_state_dict(torch.load(train_config['load_path']))
    model.cuda()
    
    # eval_datasets = RoadDataset([os.path.join(train_config['valid_data_directory'], file_name) for file_name in os.listdir(train_config['valid_data_directory'])])
    # eval_loader = DataLoader(
    # eval_datasets, 
    # batch_size=20,
    # drop_last=False) 
    
    test_datasets = RoadDataset([os.path.join(train_config['test_data_directory'], file_name) for file_name in os.listdir(train_config['test_data_directory'])])
    test_loader = DataLoader(
    test_datasets, 
    batch_size=24,
    drop_last=False)       
    
    # eval_res = eval_infer(model, eval_loader)
    res = infer(model, test_loader)
    
    from argoverse.evaluation.competition_util import generate_forecasting_h5
    generate_forecasting_h5(res, "submit_v3/")