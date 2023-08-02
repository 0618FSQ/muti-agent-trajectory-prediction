from torch.utils.data import Dataset
import json
import numpy as np
import os
import joblib

class RoadDataset(Dataset):
    def __init__(self, 
                 files, 
                 obs_horizon=20, 
                 pred_horizon=30, 
                 max_map_feature_length=200
    ):
        super().__init__()
        self.files = files
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.max_map_feature_length = max_map_feature_length
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        data = joblib.load(file_path)
        return data
