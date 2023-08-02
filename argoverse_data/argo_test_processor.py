from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import List
import pandas as pd
from pandas import DataFrame
import os
from torch.utils.data import Dataset
import numpy as np
from collections import deque
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from utils.calculator import Calculator


class ArgoverseInputDataProcessor(Dataset):
    def __init__(
            self,
            file_path_list: List[str],
            mem_location: str,
            normalized: bool = True,
            obs_horizon: int = 20,
            pred_horizon: int = 30,
            obs_range=150,
            mode = 'compute_state'
    ):
        super().__init__()
        self.file_path_list = file_path_list
        self.mem_location = mem_location
        self.normalized = normalized
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.obs_range = obs_range
        self.map_api = ArgoverseMap()
        self.mode = mode
        
    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        file_path = self.file_path_list[idx]
        directories, file_name = os.path.splitext(file_path)
        directories = directories.split("/")
        sub, file_name = directories[-2:]
        df = pd.read_csv(file_path)
        data = self.read_argo_data(df)
        ans = self.process(data)
        graph = self.get_map_features(data)
            
        processed_data = ans.copy()
        processed_data.update(graph)
        processed_data["data_id"] = file_name
        if self.mode == 'compute_state':
            return self.get_num(processed_data)
        else:
            return processed_data

        # if len(graph) == 0:
        #     return []
        # self.save(sub, file_name, data)
        # return []
        

    def save(self, sub, file_name, data):
        base = self.mem_location
        df = pd.DataFrame(
            [list(data.values())],
            columns=list(data.keys())
        )
        f_name = f"{sub}_{file_name}.pkl"
        df.to_pickle(os.path.join(base, f_name))

    def get_num(self, data):
        return data['target_agent_length'],data['agent_length'], data['map_feature'].shape[0]


    def read_argo_data(self, df: DataFrame):

        """TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y"""
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
        speed_feature = real_trajectory[:-1] - real_trajectory[1:]
        padding = np.zeros([self.obs_horizon - 1, 2], dtype=np.float32)

        feature = np.hstack([location_feature, speed_feature, padding])
        feature_mask = mask[:-1] & mask[1:]
        feature = torch.from_numpy(feature).type(torch.float32)
        mask = torch.from_numpy(feature_mask).type(torch.bool)
        feature = torch.masked_fill(feature, ~mask.unsqueeze(-1), 0)
        
        agent_local_feature = None
        if mask.any():
            agent_local_feature = feature[mask][-1]
        
        return feature.detach().numpy(), feature_mask, agent_local_feature


    def process(self, data):
        rot = data["rot"]
        orig = data["orig"]
        """
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
            
        future_mask,
        future_cross_mask,
            
        f2h_cross_mask,
        f2a_cross_mask,
        f2m_cross_mask,
        
        target,
        target_mask
        这块把相对于最后一帧的距离加上，最后看是不是要加入到对应的误差中
        
        """

        agents_history_trajectory_feature = []
        agents_history_trajectory_feature_mask = []
        agents_history_trajectory_feature_length = 0

        target_agent_history_trajectory_feature = []
        target_agent_history_feature_mask = []
        target_agent_history_length = 0
        
        
        target_agent_current_orig = []
        agent_local_feature_list = []
        target_agent_local_feature_list = []
        
        for traj, step in zip(data['trajs'], data['steps']):
            if traj.shape[0] == 1:
                continue
            # 环境所有的物体，都放进对应的agents_history_trajectory
            trajectory = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            agent_history = trajectory[step < self.obs_horizon]
            agent_index = step[step < self.obs_horizon]

            agent_feature, agent_mask, agent_local_feature = self.get_agent_trajectory(agent_history, agent_index)
            
            if agent_local_feature is not None:
                agents_history_trajectory_feature_length += 1
                agent_local_feature_list.append(agent_local_feature)
                agents_history_trajectory_feature.append(agent_feature)
                agents_history_trajectory_feature_mask.append(agent_mask)

            if (self.obs_horizon - 1) not in step:
                # 过滤掉不在预测场景中的数据
                continue
            obs_step = step[step < self.obs_horizon]
            if len(obs_step) <= 5:
                continue
            calculator = Calculator(x=traj[:, 0], y=traj[:, 1], time=step * 0.1)
            speed = calculator.get_v()
            if np.all(speed <= 0.5):
                continue
            target_agent_local_feature_list.append(agent_local_feature)
            target_agent_history_trajectory_feature.append(agent_feature)
            target_agent_history_feature_mask.append(agent_mask)
            target_agent_history_length += 1

            target_orig = trajectory[step == self.obs_horizon - 1]
            target_agent_current_orig.append(target_orig)

        new_data = {
            "target_agent_history_trajectory": np.stack(target_agent_history_trajectory_feature),
            "target_agent_history_mask": np.stack(target_agent_history_feature_mask),

            "agent_length": agents_history_trajectory_feature_length,
            "target_agent_length": target_agent_history_length,

            "agents_history_trajectory": np.stack(agents_history_trajectory_feature),
            "agents_history_mask": np.stack(agents_history_trajectory_feature_mask),

            "target_agent_orig": np.stack(target_agent_current_orig).reshape(-1, 2),
            "target_agent_local_feature": np.stack(target_agent_local_feature_list),
            "agent_local_feature": np.stack(agent_local_feature_list),
            "rot": rot, 
            "orig": orig
        }
        return new_data
 
    def get_line_feature(self, ctrln, rot, interst_point):
        num_segs = len(ctrln) - 1
        ctrln = np.array(ctrln)
        ctrln = np.matmul(rot, (ctrln - interst_point.reshape(-1, 2)).T).T
        
        ctr = (ctrln[:-1] + ctrln[1:]) / 2

        feat = ctrln[:-1] - ctrln[1:]

        control = np.ones([num_segs, 1], np.float32)
        intersect = np.zeros([num_segs, 1], np.float32)
        tmp = np.hstack([ctr, feat, control, intersect])  # [9, 6]
        map_local_feature = tmp[-1]
        ans = np.zeros(
            [self.obs_horizon - 1, 6], dtype=np.float32
        )
        ans[:tmp.shape[0], :] = tmp
        mask = np.zeros(
            [self.obs_horizon - 1], dtype=np.bool_
        )
        mask[:tmp.shape[0]] = True

        return ans, mask, map_local_feature

    def get_map_features(self, data):
        interst_point = data["orig"]
        city = data["city"]

        lane_ids = self.map_api.get_lane_ids_in_xy_bbox(interst_point[0],
                                                        interst_point[1],
                                                        city_name=city,
                                                        query_search_range_manhattan=self.obs_range)

        map_features = []
        map_feature_mask = []
        map_local_feature = []
        for lane_id in lane_ids:
            ctrln = self.map_api.get_lane_segment_centerline(lane_id, city)[:, :2]

            if len(ctrln) > 2:
                ctr_map_feature = self.get_line_feature(
                    ctrln, rot=data["rot"], interst_point=interst_point
                )
                map_features.append(ctr_map_feature[0])
                map_feature_mask.append(ctr_map_feature[1])
                map_local_feature.append(ctr_map_feature[2])

        map_features = np.stack(map_features)
        map_feature_mask = np.stack(map_feature_mask)
        map_local_feature = np.stack(map_local_feature)
        map_feature = {
            "map_feature": map_features,
            "map_feature_mask": map_feature_mask,
            "map_local_feature": map_local_feature
        }
        return map_feature


if __name__ == "__main__":
    import sys
    from tqdm import tqdm
    
    pipeline = sys.argv[1] if len(sys.argv) >= 2 else 'train'
    base = f"/home/data1/prediction/dataset/argoverse/csv/{pipeline}"
    file_list = [os.path.join(base, file_name) for file_name in os.listdir(base)]
    processor = ArgoverseInputDataProcessor(
    file_path_list=file_list,
    mem_location='',
    )
    max_target_agent_num = 0
    max_agent_num = 0
    max_map_num = 0
    for i in tqdm(range(len(file_list))):
        target_agent_num, agent_num, map_num = processor[i]
        max_target_agent_num = max(max_target_agent_num, target_agent_num)
        max_agent_num = max(max_agent_num, agent_num)
        max_map_num = max(max_map_num, map_num)
        
    print('max_target_agent_num',max_target_agent_num)
    print('max_agent_num',max_agent_num)
    print('max_map_num',max_map_num)
    
    