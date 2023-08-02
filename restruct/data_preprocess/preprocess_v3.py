from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import torch
import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List
from torch.utils.data import Dataset
from utils.calculator import Calculator
from argoverse.map_representation.map_api import ArgoverseMap


class Preprocess(Dataset):
    def __init__(
            self,
            data_path: str,
            save_path: str,
            normalized= True,
            obs_horizon= 20,
            pred_horizon= 30,
            obs_range=150,
            max_map_num=160,
            max_target_agt_num=1,
            max_agt_num=50,

    ):
        super().__init__()
        self.save_path = save_path
        self.data_path = data_path
        self.files_list = os.listdir(data_path)
        self.normalized = normalized
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.obs_range = obs_range
        self.map_api = ArgoverseMap()

        self.max_map_num = max_map_num
        self.max_target_agt_num = max_target_agt_num
        self.max_agt_num = max_agt_num
        
    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.files_list[idx])
        df = pd.read_csv(file_path)
        
        data = self.read_argo_data(df)
        ans = self.process(data)
        graph = self.get_map_features(data)
        
        processed_data = ans.copy()
        processed_data.update(graph)
        
        padded_data = self.padding(processed_data)
        self.save(padded_data, file_path)
        return []

    def save(self, data, file_path):
        
        directories, file_name = os.path.splitext(file_path)
        directories = directories.split("/")
        sub, file_name = directories[-2:]
        f_name = f"{sub}_{file_name}"
        os.makedirs(self.save_path, exist_ok=True)
        joblib.dump(data, os.path.join(self.save_path, f"{f_name}.bin"))

    def read_argo_data(self, df: DataFrame):
        
        df = df.sort_values(by="TIMESTAMP")
        df.reset_index(drop=True, inplace=True)
        agt_times = np.sort(np.unique(df['TIMESTAMP'].values))
        mapping = dict([(ts, i) for i, ts in enumerate(agt_times)])

        df["index"] = df['TIMESTAMP'].apply(lambda x: mapping[x]).values

        groups = df.groupby(['TRACK_ID', 'OBJECT_TYPE'])

        keys = list(groups.groups.keys())
        obj_type = [x[1] for x in keys]
        agt_idx = obj_type.index('AGENT')

        ctx_trajs, ctx_steps = [], []

        orig = None
        rot = None

        for (track_id, object_type), group in groups:
            traj = group[["X", "Y"]].values
            steps = group["index"].values
            ctx_trajs.append(traj)
            ctx_steps.append(steps)
            if object_type == "AGENT":
                orig = traj[self.obs_horizon - 1]
                pre = orig - traj[0]
                if self.normalized:
                    theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2
                    rot = np.asarray(
                        [
                            [np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]
                        ],
                        np.float32
                    )
                else:
                    theta = None
                    rot = np.asarray(
                        [
                            [1.0, 0.0],
                            [0.0, 1.0]
                        ],
                        np.float32
                    )
        ctx_trajs[0], ctx_trajs[agt_idx] = ctx_trajs[agt_idx], ctx_trajs[0]
        ctx_steps[0], ctx_steps[agt_idx] = ctx_steps[agt_idx], ctx_steps[0]

        data = dict()
        data['city'] = df["CITY_NAME"].values[0]
        data['trajs'] = ctx_trajs
        data['steps'] = ctx_steps
        data["orig"] = orig
        data["rot"] = rot
        return data

    def get_future_trajectory_and_mask(self, trajectory, index):
        target = np.zeros([self.pred_horizon, 2], dtype=np.float32)
        target_mask = np.zeros([self.pred_horizon], dtype=np.bool_)
        for i in range(self.pred_horizon):
            if ((i + self.obs_horizon - 1) in index) and ((i + self.obs_horizon) in index):
                target[i, :] = trajectory[(i + self.obs_horizon) == index, :] - trajectory[(i + self.obs_horizon - 1) == index, :]
                target_mask[i] = True
        return target, target_mask

    def get_traj_feature_and_mask(self, trajectory, index):
        real_trajectory = torch.FloatTensor(
            trajectory
        )
        padding_trajectory = torch.zeros(
            [self.obs_horizon, 2], dtype=torch.float32
        )

        padding_trajectory[index] = real_trajectory

        feature = torch.zeros(
            [self.obs_horizon - 1, 6], dtype=torch.float32
        )
        mask = torch.zeros(
            [self.obs_horizon - 1], dtype=torch.bool
        )
        mask[index[:-1]] = True

        feature[:, :2] = (padding_trajectory[:-1, :] + padding_trajectory[1:, :]) / 2
        feature[:, 2:4] = padding_trajectory[:-1, :] - padding_trajectory[1:, :]
        feature[:, -1] = 1
        new_feature = torch.masked_fill(feature, ~mask.unsqueeze(dim=-1), 0)
        return new_feature.detach().numpy(), mask.detach().numpy()

    def get_agent_trajectory(self, trajectory, index):
        
        assert trajectory.shape[0] == index.shape[0]
        
        real_trajectory = np.zeros([self.obs_horizon, 2], dtype=np.float32)
        real_trajectory[index, :] = trajectory

        mask = np.zeros([self.obs_horizon], dtype=np.bool_)
        mask[index] = True

        location_feature = (real_trajectory[:-1] + real_trajectory[1:]) / 2
        vel = real_trajectory[1:] - real_trajectory[:-1]
        heading = np.arctan2(vel[:, 1], vel[:, 0]).reshape(-1, 1)

        feature = np.hstack([location_feature, vel, heading])
        feature_mask = mask[1:] & mask[:-1]
        
        feature = torch.from_numpy(feature).type(torch.float32)
        mask = torch.from_numpy(feature_mask).type(torch.bool)
        feature = torch.masked_fill(feature, ~mask.unsqueeze(-1), 0)
        return feature.detach().numpy(), feature_mask

    def process(self, data):
        rot = data["rot"]
        orig = data["orig"]

        agents_history_trajectory_feature = []
        agents_history_trajectory_feature_mask = []
        agents_history_trajectory_feature_length = 0

        target_agent_history_trajectory_feature = []
        target_agent_history_feature_mask = []
        target_agent_history_length = 0
        target_agent_current_orig = []

        target_list = []
        target_mask_list = []
        location_list = []

        for i, [traj, step] in enumerate(zip(data['trajs'], data['steps'])):
            
            if np.all(step >= self.obs_horizon):
                continue
            
            if (self.obs_horizon - 1) not in step:
                continue
            
            future_step = step[step >= self.obs_horizon]
            obs_step = step[step < self.obs_horizon]
            if len(obs_step) <= 5 or len(future_step) <= 5:
                continue

            # 过滤静止的agent
            calculator = Calculator(x=traj[:, 0], y=traj[:, 1], time=step * 0.1)
            speed = calculator.get_v()
            if np.all(speed <= 0.5):
                continue
            
            trajectory = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            agent_history = trajectory[step < self.obs_horizon]
            agent_feature, agent_mask = self.get_agent_trajectory(agent_history, obs_step)
            
            if i == 0:
                target_agent_history_length += 1
                target_orig = trajectory[step == self.obs_horizon - 1]
                target, target_mask = self.get_future_trajectory_and_mask(trajectory, step)
                location = np.cumsum(target, axis=0)

                target_agent_current_orig.append(target_orig)
                target_agent_history_trajectory_feature.append(agent_feature)
                target_agent_history_feature_mask.append(agent_mask)
                target_list.append(target)
                target_mask_list.append(target_mask)
                location_list.append(location)
            
            else:
                agents_history_trajectory_feature_length += 1
                agents_history_trajectory_feature.append(agent_feature)
                agents_history_trajectory_feature_mask.append(agent_mask)
        
        if not agents_history_trajectory_feature:
            agent_feature = np.zeros(shape=(19, 5), dtype=np.float32)
            agent_mask = np.zeros(19, dtype=np.bool_)
            agents_history_trajectory_feature.append(agent_feature)
            agents_history_trajectory_feature_mask.append(agent_mask)
            
        new_data = {
            "target_agent_history_trajectory": np.stack(target_agent_history_trajectory_feature),
            "target_agent_history_mask": np.stack(target_agent_history_feature_mask),

            "agent_length": agents_history_trajectory_feature_length,
            "target_agent_length": target_agent_history_length,

            "agents_history_trajectory": np.stack(agents_history_trajectory_feature),
            "agents_history_mask": np.stack(agents_history_trajectory_feature_mask),

            "target": np.stack(target_list),
            "target_mask": np.stack(target_mask_list),
            "location_seq": np.stack(location_list),

            "target_agent_orig": np.stack(target_agent_current_orig).reshape(-1, 2)
        }
        return new_data

    def get_future_location_relative_with_current_position(self, future_trajectory, future_steps, current_orig):
        ans = np.zeros([self.pred_horizon, 2], dtype=np.float32)
        future_location_relative_with_current_position = future_trajectory - current_orig
        ans[future_steps - self.obs_horizon, :] = future_location_relative_with_current_position
        return ans
    
    def get_line_feature(self, ctrln, rot, interst_point):
        
        ctrln = np.array(ctrln)
        
        # assert ctrln.shape == (10,2), f"ctrln shape is not (10,2) current shape: ({ctrln.shape})"
        
        ctrln = np.matmul(rot, (ctrln - interst_point.reshape(-1, 2)).T).T
        
        ctr = (ctrln[:-1] + ctrln[1:]) / 2
        feat = ctrln[1:] - ctrln[:-1]
        # 改sin 
        heading = np.arctan2(feat[:, 1], feat[:, 0]).reshape(-1, 1)
        
        tmp = np.hstack([ctr, feat, heading])  # [9, 5]
        ans = np.zeros(
            [9, 5], dtype=np.float32
        )
        ans[:tmp.shape[0], :] = tmp
        mask = np.zeros(
            [9], dtype=np.bool_
        )
        mask[:tmp.shape[0]] = True

        return ans, mask

    def get_map_features(self, data):
        interst_point = data["orig"]
        city = data["city"]

        lane_ids = self.map_api.get_lane_ids_in_xy_bbox(interst_point[0],
                                                        interst_point[1],
                                                        city_name=city,
                                                        query_search_range_manhattan=self.obs_range)

        map_features = []
        map_feature_mask = []
        for lane_id in lane_ids:
            ctrln = self.map_api.get_lane_segment_centerline(lane_id, city)[:, :2]

            if len(ctrln) > 2:
                ctr_map_feature = self.get_line_feature(
                    ctrln, rot=data["rot"], interst_point=interst_point
                )
                map_features.append(ctr_map_feature[0])
                map_feature_mask.append(ctr_map_feature[1])

        map_features = np.stack(map_features)
        map_feature_mask = np.stack(map_feature_mask)
        map_feature = {
            "map_feature": map_features,
            "map_feature_mask": map_feature_mask
        }
        return map_feature

    def padding(self, data):

        target_agent_history_trajectory = data['target_agent_history_trajectory']
        target_agent_history_mask = data['target_agent_history_mask']
        target = data['target']
        target_mask = data['target_mask']
        location_seq = data['location_seq']
        target_agent_orig = data['target_agent_orig']
        agents_history_trajectory = data['agents_history_trajectory']
        agents_history_mask = data['agents_history_mask']
        map_feature = data['map_feature']
        map_feature_mask = data['map_feature_mask']
        
        agent_length = data['agent_length']
        target_agent_length = data['target_agent_length']
        map_length = map_feature.shape[0]
        
        # get padding
        # target agent
        new_target_agt_hist_traj = np.zeros([self.max_target_agt_num, self.obs_horizon - 1, 5], dtype=np.float32)
        new_target_agt_hist_traj[:target_agent_length, :, :] = target_agent_history_trajectory 
        new_target_agt_traj_mask = np.zeros([self.max_target_agt_num, self.obs_horizon - 1], dtype=np.bool_)
        new_target_agt_traj_mask[:target_agent_length, :] = target_agent_history_mask
        new_target_orig = np.zeros([self.max_target_agt_num, 2], dtype=np.float32)
        new_target_orig[:target_agent_length, :] = target_agent_orig
        
        # y
        new_y = np.zeros([self.max_target_agt_num, self.pred_horizon, 2], dtype=np.float32)
        new_y[:target_agent_length, :, :] = target
        new_location = np.zeros([self.max_target_agt_num, self.pred_horizon, 2], dtype=np.float32)
        new_location[:target_agent_length, :, :] = location_seq
        new_y_mask = np.zeros([self.max_target_agt_num, self.pred_horizon], dtype=np.bool_)
        new_y_mask[:target_agent_length, :] = target_mask
        
        # map feature
        new_map_feature = np.zeros([self.max_map_num, 9, 5], dtype=np.float32)
        new_map_feature_mask = np.zeros([self.max_map_num, 9], dtype=np.bool_)
        if map_length <= self.max_map_num:
            new_map_feature[:map_length, :, :] = map_feature
            new_map_feature_mask[:map_length, :] = map_feature_mask
        else:
            new_map_feature = map_feature[:self.max_map_num, :]
            new_map_feature_mask = map_feature_mask[:self.max_map_num, :]
        
        # agent feature
        new_agt_hist_traj = np.zeros([self.max_agt_num, self.obs_horizon - 1, 5], dtype=np.float32)
        new_agt_hist_traj_mask = np.zeros([self.max_agt_num, self.obs_horizon - 1], dtype=np.bool_)
        if agent_length <= self.max_agt_num:
            new_agt_hist_traj[:agent_length, :, :] = agents_history_trajectory
            new_agt_hist_traj_mask[:agent_length, :] = agents_history_mask
        else:
            new_agt_hist_traj = agents_history_trajectory[:self.max_agt_num, :, :]
            new_agt_hist_traj_mask = agents_history_mask[:self.max_agt_num, :]

        # get mask
        t2m_cross_mask = self.cross_mask(new_target_agt_hist_traj, new_map_feature, 19, map_length)
        t2a_cross_mask = self.cross_mask(new_target_agt_hist_traj, new_agt_hist_traj, 19, agent_length)
        a2m_cross_mask = self.cross_mask(new_agt_hist_traj, new_map_feature, agent_length, map_length)
        m2a_cross_mask = self.cross_mask(new_map_feature, new_agt_hist_traj, agent_length, map_length)

        
        target_agt_hist_cross_mask = self.self_attention_mask(new_target_agt_hist_traj, target_agent_length)
        agt_hist_cross_mask = self.self_attention_mask(new_agt_hist_traj, agent_length)
        map_feature_cross_mask = self.self_attention_mask(new_map_feature, map_length)

        
        res = {
            "target_agent_history_trajectory": torch.from_numpy(new_target_agt_hist_traj).squeeze().type(torch.float32),
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
            
            "target_agent_orig": torch.from_numpy(new_target_orig).type(torch.float32),
            
            "y": torch.from_numpy(new_y).type(torch.float32),
            "location": torch.from_numpy(new_location).type(torch.float32),
            "y_mask": torch.from_numpy(new_y_mask).type(torch.bool)
        }
        

        
        return res

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
    
    pipeline = sys.argv[1] if len(sys.argv) >= 2 else 'train'
    data_path = f"/home/data1/prediction/dataset/argoverse/csv/{pipeline}"
    save_path = f"/home/caros/home/fushuaiqi/data/argo_processed/{pipeline}"
    road_dataset = Preprocess(data_path=data_path, save_path=save_path)
    
    # road_dataset[0]
    data_loader = DataLoader(
        road_dataset,
        batch_size=256,
        num_workers=32
    )
    
    total = len(os.listdir(data_path)) / 256
    for i, data in enumerate(data_loader):
        print(f"{i} / {total}")