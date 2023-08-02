from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch.utils.data import Dataset
from argoverse_data.argo_test_processor import ArgoverseInputDataProcessor
import pandas as pd
import numpy as np
import torch
import joblib
import os

class RoadDatasetProcessor(Dataset):
    def __init__(self, 
                 base, 
                 save_path,
                 obs_horizon=20, 
                 future_horizon=30,
                 max_map_num=160,
                 max_target_agt_num=60,
                 max_agt_num=140,
                 map_features_amount = 10,
                ):
        self.file_list = [os.path.join(base, file_name) for file_name in os.listdir(base)]
        self.obs_horizon = obs_horizon
        self.future_horizon = future_horizon
        self.max_map_num = max_map_num
        self.max_target_agt_num = max_target_agt_num
        self.max_agt_num = max_agt_num
        self.map_features_amount = map_features_amount
        self.processor = ArgoverseInputDataProcessor(
            file_path_list=self.file_list,
            mem_location='',
            mode = None
        )
        self.save_path = save_path
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        data = self.processor[idx]
        # df = pd.read_pickle(self.file_list[idx])

        target_agent_history_trajectory = data['target_agent_history_trajectory']
        target_agent_history_mask = data['target_agent_history_mask']
        
        target_agent_local_feature = data['target_agent_local_feature']
        agent_local_feature = data['agent_local_feature']
        map_local_feature = data['map_local_feature']
        
        agents_history_trajectory = data['agents_history_trajectory']
        agents_history_mask = data['agents_history_mask']
        map_feature = data['map_feature']
        map_feature_mask = data['map_feature_mask']
        
        agent_length = data['agent_length']
        target_agent_length = data['target_agent_length']
        map_length = map_feature.shape[0]
        
        rot = data['rot']
        orig = data['orig']
        id = data['data_id']
        # get padding
        # target agent
        
        new_target_agt_hist_traj = np.zeros([self.max_target_agt_num, self.obs_horizon - 1, 6], dtype=np.float32)
        new_target_agt_traj_mask = np.zeros([self.max_target_agt_num, self.obs_horizon - 1], dtype=np.bool_)
        new_target_agent_local_feature = np.zeros([self.max_target_agt_num, 6], dtype=np.float32)
        future_mask = np.zeros([self.max_target_agt_num, self.future_horizon], dtype=np.bool_)
        if target_agent_length <= self.max_target_agt_num:

            new_target_agt_hist_traj[:target_agent_length, :, :] = target_agent_history_trajectory 
            new_target_agt_traj_mask[:target_agent_length, :] = target_agent_history_mask
            new_target_agent_local_feature[:target_agent_length, :] = target_agent_local_feature
            future_mask[:target_agent_length] = True
        else:
            new_target_agt_hist_traj = target_agent_history_trajectory[:self.max_target_agt_num, :, :]
            new_target_agt_traj_mask = target_agent_history_mask[:self.max_target_agt_num, :]
            new_target_agent_local_feature = target_agent_local_feature[:self.max_target_agt_num, :]
            future_mask[:] = True
            target_agent_length = self.max_target_agt_num
            
        # map feature
        new_map_feature = np.zeros([self.max_map_num, self.obs_horizon - 1, 6], dtype=np.float32)
        new_map_feature_mask = np.zeros([self.max_map_num, self.obs_horizon - 1], dtype=np.bool_)
        new_map_local_feature = np.zeros([self.max_map_num, 6], dtype=np.float32)
        if map_length <= self.max_map_num:
            new_map_feature[:map_length, :, :] = map_feature
            new_map_feature_mask[:map_length, :] = map_feature_mask
            new_map_local_feature[:map_length, :] = map_local_feature
        else:
            new_map_feature = map_feature[:self.max_map_num, :]
            new_map_feature_mask = map_feature_mask[:self.max_map_num, :]
            new_map_local_feature = map_local_feature[:self.max_map_num, :]
            map_length = self.max_map_num
            
        # agent feature
        new_agent_local_feature = np.zeros([self.max_agt_num, 6], dtype=np.float32)
        new_agt_hist_traj = np.zeros([self.max_agt_num, self.obs_horizon - 1, 6], dtype=np.float32)
        new_agt_hist_traj_mask = np.zeros([self.max_agt_num, self.obs_horizon - 1], dtype=np.bool_)
        if agent_length <= self.max_agt_num:
            new_agent_local_feature[:agent_length, :] = agent_local_feature
            new_agt_hist_traj[:agent_length, :, :] = agents_history_trajectory
            new_agt_hist_traj_mask[:agent_length, :] = agents_history_mask
        else:
            new_agent_local_feature = agent_local_feature[:self.max_agt_num]
            new_agt_hist_traj = agents_history_trajectory[:self.max_agt_num, :, :]
            new_agt_hist_traj_mask = agents_history_mask[:self.max_agt_num, :]
            agent_length = self.max_agt_num

        # get mask
        t2m_cross_mask = self.cross_mask(new_target_agt_hist_traj, new_map_feature, target_agent_length, map_length)
        t2a_cross_mask = self.cross_mask(new_target_agt_hist_traj, new_agt_hist_traj, target_agent_length, agent_length)
        a2m_cross_mask = self.cross_mask(new_agt_hist_traj, new_map_feature, agent_length, map_length)
        m2a_cross_mask = self.cross_mask(new_map_feature, new_agt_hist_traj, agent_length, map_length)
        f2h_cross_mask = self.cross_mask(future_mask, new_target_agt_hist_traj, target_agent_length, target_agent_length)
        f2a_cross_mask = self.cross_mask(future_mask, new_agt_hist_traj, target_agent_length, agent_length)
        f2m_cross_mask = self.cross_mask(future_mask, new_map_feature, target_agent_length, map_length)
        
        target_agt_hist_cross_mask = self.self_attention_mask(new_target_agt_hist_traj, target_agent_length)
        agt_hist_cross_mask = self.self_attention_mask(new_agt_hist_traj, agent_length)
        map_feature_cross_mask = self.self_attention_mask(new_map_feature, map_length)
        future_cross_mask = self.self_attention_mask(future_mask, target_agent_length)
        
        res = {
            "target_agent_history_trajectory": torch.from_numpy(new_target_agt_hist_traj).type(torch.float32),
            "target_agent_history_mask": torch.from_numpy(new_target_agt_traj_mask).type(torch.bool),
            "target_agent_history_cross_mask": target_agt_hist_cross_mask,
            
            "agents_history_trajectory": torch.from_numpy(new_agt_hist_traj).type(torch.float32),
            "agents_history_mask": torch.from_numpy(new_agt_hist_traj_mask).type(torch.bool),
            "agents_history_cross_mask": agt_hist_cross_mask,
            
            "t2a_cross_mask": t2a_cross_mask,
            "t2m_cross_mask": t2m_cross_mask,
            "a2m_cross_mask": a2m_cross_mask,
            "m2a_cross_mask": m2a_cross_mask,
            
            "map_feature": torch.from_numpy(new_map_feature).type(torch.float32),
            "map_feature_mask": torch.from_numpy(new_map_feature_mask).type(torch.bool),
            "map_feature_cross_mask": map_feature_cross_mask,
            
            "target_agent_local_feature": torch.from_numpy(new_target_agent_local_feature).type(torch.float32),
            "agent_local_feature": torch.from_numpy(new_agent_local_feature).type(torch.float32),
            "new_map_local_feature": torch.from_numpy(new_map_local_feature).type(torch.float32),
            
            "future_mask": future_mask,
            "future_cross_mask": future_cross_mask,
            
            "f2h_cross_mask": f2h_cross_mask,
            "f2a_cross_mask": f2a_cross_mask,
            "f2m_cross_mask": f2m_cross_mask,
            
            "rot": torch.from_numpy(rot).type(torch.float32),
            "orig": torch.from_numpy(orig).type(torch.float32),
            "id": id
        }
        
        file_path = self.file_list[idx]
        directories, file_name = os.path.splitext(file_path)
        directories = directories.split("/")
        sub, file_name = directories[-2:]
        
        f_name = f"{sub}_{file_name}"
        
        joblib.dump(res, os.path.join(self.save_path, f"{f_name}.bin"))
        
        return []

    def cross_mask(self, x, y, x_len, y_len):
        mask = torch.zeros(size=[x.shape[0], y.shape[0]], dtype=torch.bool)
        mask[:, y_len:] = True
        mask[x_len:] = True
        return mask
    
    def self_attention_mask(self, x, valid_len):
        shape = [x.shape[0], x.shape[0]]
        mask = torch.zeros(size=shape, dtype=torch.bool)
        
        mask[:, valid_len:] = True
        mask[valid_len:] = True
        return mask
    
if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    
    pipeline = sys.argv[1] if len(sys.argv) >= 2 else 'test'
    base = f"/home/data1/prediction/dataset/argoverse/csv/{pipeline}"
    
    road_dataset = RoadDatasetProcessor(base=base,
                               save_path=f"/home/caros/data16t/tmp/argo_processed_0706/{pipeline}"
                               )
    # road_dataset[0]
    data_loader = DataLoader(
        road_dataset,
        batch_size=256,
        num_workers=32
    )
    for i, data in enumerate(data_loader):
        print(i)