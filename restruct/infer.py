import torch
import os

import numpy as np
import argparse
import json
from tqdm import tqdm

# from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).resolve().parents[1]))

from model_v3.scene_model import SceneModel
from dataset.get_data import ArgoData
from torch.utils.data import DataLoader

def getPositionEncoding(seq_len,dim,n=10000):
    PE = np.zeros(shape=(seq_len, dim))
    for pos in range(seq_len):
        for i in range(int(dim/2)):
            denominator = np.power(n, 2*i/dim)
            PE[pos,2*i] = np.sin(pos/denominator)
            PE[pos,2*i+1] = np.cos(pos/denominator)

    return torch.from_numpy(PE).to(torch.float32)


def infer(model, test_loader, use_cuda=True):
    
    model.eval()
    trajs = {}
    probabilities = {}
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            target_agent_history_trajectory=data["target_agent_history_trajectory"]
            # target_agent_history_mask=data["target_agent_history_mask"]
            # target_agent_history_cross_mask=data["target_agent_history_cross_mask"]

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

            # target_agent_orig=data["target_agent_orig"]
            
            rot = data["rot"]
            orig = data["orig"]
            id = data["id"]
            
            if use_cuda:
                target_agent_history_trajectory=target_agent_history_trajectory.cuda()
                # target_agent_history_mask=target_agent_history_mask.cuda()
                # target_agent_history_cross_mask=target_agent_history_cross_mask.cuda()

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

                # target_agent_orig=target_agent_orig.cuda()
                
                rot = rot.cuda()
                orig = orig.cuda()
                
                # position_emb = getPositionEncoding(19, dim=128).unsqueeze(0).unsqueeze(0)
                # position_emb = position_emb.cuda()
                
            pred_trajs, probs = model(
                target_agent_history_trajectory=target_agent_history_trajectory,
                # target_agent_history_mask=target_agent_history_mask,
                # target_agent_history_cross_mask=target_agent_history_cross_mask,

                agents_history_trajectory=agents_history_trajectory,
                agents_history_mask=agents_history_mask,
                agents_history_cross_mask=agents_history_cross_mask,

                t2a_cross_mask=t2a_cross_mask,
                t2m_cross_mask=t2m_cross_mask,
                a2m_cross_mask=a2m_cross_mask,
                m2a_cross_mask=m2a_cross_mask,

                map_feature=map_feature,
                map_feature_mask=map_feature_mask,
                map_feature_cross_mask=map_feature_cross_mask,

                # target_agent_orig=target_agent_orig,
                # position_emb = position_emb
            )
            
            real_traj = torch.matmul(pred_trajs.permute(1, 0, 2, 3), rot)
            real_traj = torch.add(orig.unsqueeze(dim=1).unsqueeze(0), real_traj)

            real_traj = real_traj.cpu().detach().numpy()
            probs = probs.cpu().detach().numpy()
            for idx, data_id in enumerate(id):
                trajs[data_id.item()] = real_traj[:, idx, :, :]
                probabilities[data_id.item()] = probs[idx].tolist()
        torch.cuda.empty_cache()
    return  trajs, probabilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,5,6')
    parser.add_argument('--train_config_path', type=str, default="scene_model_v2.json")
    config = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    train_config = json.load(open(config.train_config_path, "r"))

    use_cuda = True
    # device = torch.device("cuda:3") if use_cuda else torch.device("cpu")

    # 步骤三：模型分布式处理
    model = SceneModel(**train_config['model'])
    model.load_state_dict(torch.load(train_config['load_path']))
    model.cuda()
    
    test_datasets = ArgoData([os.path.join(train_config['test_data_directory'], file_name) for file_name in os.listdir(train_config['test_data_directory'])])
    test_loader = DataLoader(
    test_datasets, 
    batch_size=16,
    drop_last=False)       
    
    # eval_res = eval_infer(model, eval_loader)
    trajs, probabilities = infer(model, test_loader)
    
    from argoverse.evaluation.competition_util import generate_forecasting_h5
    generate_forecasting_h5(trajs,
                            "submit_v3/",
                            filename='submit0621',
                            probabilities=probabilities)