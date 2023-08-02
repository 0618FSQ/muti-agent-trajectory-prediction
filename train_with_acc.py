import os
import torch
from torch.optim import Adam, AdamW
import numpy as np
import argparse
import json

# from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).resolve().parents[1]))

from model.model_v2.scene_model_v3 import SceneModel
from dataset import RoadDataset
from trainer.optim_schedule import ScheduledOptim
from loss import SceneLoss
from trainer.scene_trainer_v2 import SceneTrainer
from torch.utils.data import DataLoader
from accelerate import Accelerator


def train(epoch, model, loader):
    
    total_loss = 0.0
    num_points = 0
    model.train()
    for i, data in enumerate(loader):
        optm_schedule.zero_grad()

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

        target_agent_orig=data["target_agent_orig"]

        future_mask=data["future_mask"]
        future_cross_mask=data["future_cross_mask"]

        f2h_cross_mask=data["f2h_cross_mask"]
        f2a_cross_mask=data["f2a_cross_mask"]
        f2m_cross_mask=data["f2m_cross_mask"]
        
        y = data["y"]
        location = data["location"]
        y_mask = data["y_mask"]
        

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

        target_agent_orig=target_agent_orig.cuda()

        future_mask=future_mask.cuda()
        future_cross_mask=future_cross_mask.cuda()

        f2h_cross_mask=f2h_cross_mask.cuda()
        f2a_cross_mask=f2a_cross_mask.cuda()
        f2m_cross_mask=f2m_cross_mask.cuda()
        y = y.cuda()
        location = location.cuda()
        y_mask = y_mask.cuda()
        
        out = model(
            target_agent_history_trajectory=target_agent_history_trajectory,
            target_agent_history_mask=target_agent_history_mask,
            target_agent_history_cross_mask=target_agent_history_cross_mask,

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

            target_agent_orig=target_agent_orig,

            future_mask=future_mask,
            future_cross_mask=future_cross_mask,

            f2h_cross_mask=f2h_cross_mask,
            f2a_cross_mask=f2a_cross_mask,
            f2m_cross_mask=f2m_cross_mask
        )
        
        diff_loss, ade_loss, fde_loss = loss_fun(out, location, y, y_mask)
        loss = diff_loss + ade_loss + fde_loss
        accelerator.backward(loss)
        optim.step()
        # self.optm_schedule.step()
        points = torch.sum(y_mask.float()).item()
        num_points += points
        total_loss += loss.item() * points
        print("[Info:train_Ep_{}_iter_{}: loss: {:.5e}; diff_loss: {:.5e}; ade_loss: {:.5e}; fde_loss: {:.5e}; avg_loss: {:.5e}]".format(epoch, 
                                                                                                i, 
                                                                                                loss.item(),
                                                                                                diff_loss.item(),
                                                                                                ade_loss.item(),
                                                                                                fde_loss.item(),
                                                                                                total_loss / num_points))
        return loss.item()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1,2,3,4,5,6,7')
    parser.add_argument('--train_config_path', type=str, default="argo_pred0526/scene_model_v2.json")
    config = parser.parse_args()
    train_config = json.load(open(config.train_config_path, "r"))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    
    accelerator = Accelerator()
    device = accelerator.device
    
    loss_fun = SceneLoss(reduction="none")
    min_loss = np.inf

    # 步骤三：模型分布式处理
    model = SceneModel(**train_config['model'])
    
    optim = AdamW(model.parameters(), 
                  lr=train_config['optimizer']['lr'], 
                  betas=train_config['optimizer']['betas'], 
                  weight_decay=train_config['optimizer']['weight_decay']
                  )
    
    optm_schedule = ScheduledOptim(optim, 
                                   train_config['optim_schedule']['lr'], 
                                   n_warmup_epoch=train_config['optim_schedule']['n_warmup_epoch'], 
                                   update_rate=train_config['optim_schedule']['update_rate'],
                                   decay_rate=train_config['optim_schedule']['decay_rate'])
    train_datasets = RoadDataset([os.path.join(train_config['train_data_directory'], file_name) for file_name in os.listdir(train_config['train_data_directory'])])
    eval_datasets = RoadDataset([os.path.join(train_config['valid_data_directory'], file_name) for file_name in os.listdir(train_config['valid_data_directory'])])

    train_dataloader = DataLoader(
        train_datasets, 
        batch_size=train_config['batch_size'],
    )

    model, optimizer, train_dataloader = accelerator.prepare(model, optim, train_dataloader)


    for epoch in range(train_config['epochs']):
        loss = train(epoch, model, train_dataloader)
        