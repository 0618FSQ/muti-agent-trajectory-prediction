from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from typing import List
import pandas as pd
from pandas import DataFrame
import os
import numpy as np
import json

class ArgoverseInputDataProcessor:
    def __init__(
            self,
            file_path_list: List[str],
            save_path: str,
            obs_horizon: int = 20,
            pred_horizon: int = 30,

    ):
        super().__init__()
        self.file_path_list = file_path_list
        self.save_path = save_path
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

    def porcess(self):
        
        res = []
        for file_path in tqdm(self.file_path_list[:1000]):
            df = pd.read_csv(file_path)
            data = self.read_argo_data(df)
            end_points = self.extract_end_points(data)
            res.extend(end_points)
        
        with open(os.path.join(self.save_path, 'end_point.json'), 'w') as f:
            json.dump(res, f, indent=4)
        return res
      
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

        for (track_id, object_type), group in groups:
            traj = group[["X", "Y"]].values
            steps = group["index"].values
            ctx_trajs.append(traj)
            ctx_steps.append(steps)
                    
        ctx_trajs[0], ctx_trajs[agt_idx] = ctx_trajs[agt_idx], ctx_trajs[0]
        ctx_steps[0], ctx_steps[agt_idx] = ctx_steps[agt_idx], ctx_steps[0]

        data = dict()
        data['city'] = df["CITY_NAME"].values[0]
        data['trajs'] = ctx_trajs
        data['steps'] = ctx_steps

        return data

    def extract_end_points(self, data):
        
        end_points = []

        for traj, step in zip(data['trajs'], data['steps']):
            if self.obs_horizon - 1 not in step or 49 not in step or self.obs_horizon not in step:
                continue
            
            orig = traj[step == (self.obs_horizon - 1)][0]
            
            pre = traj[step == self.obs_horizon][0] - orig

            theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2
            rot = np.asarray(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ],
                np.float32
            )

            trajectory = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T
            end_point = trajectory[step==49][0].tolist()
            end_points.append(end_point)
            
        return end_points

  

if __name__ == "__main__":
    import sys
    from tqdm import tqdm
    
    pipeline = sys.argv[1] if len(sys.argv) >= 2 else 'valid'
    base = f"/home/data1/prediction/dataset/argoverse/csv/{pipeline}"
    file_list = [os.path.join(base, file_name) for file_name in os.listdir(base)]
    processor = ArgoverseInputDataProcessor(
    file_path_list=file_list,
    save_path='')

    processor.porcess()

    