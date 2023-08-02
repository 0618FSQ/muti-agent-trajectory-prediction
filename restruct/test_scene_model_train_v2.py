import os
from torch.optim import Adam, AdamW
import numpy as np
import argparse
import json

# from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataset.get_data import ArgoData
from model_v5.scene_model import SceneModel
from trainer.optim_schedule import ScheduledOptim
from loss import SceneLoss
from trainer.scene_trainer_v2 import SceneTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0,1,2,3,4')
    parser.add_argument('--train_config_path', type=str, default="restruct/scene_model_v2.json")
    config = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    loss_fun = SceneLoss(reduction="none")

    min_loss = np.inf
    train_config = json.load(open(config.train_config_path, "r"))

    model = SceneModel(**train_config['model'])
    
    model_params = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % ((model_params)/1e6))
    
    
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
    train_datasets = ArgoData([os.path.join(train_config['train_data_directory'], file_name) for file_name in os.listdir(train_config['train_data_directory'])])
    eval_datasets = ArgoData([os.path.join(train_config['valid_data_directory'], file_name) for file_name in os.listdir(train_config['valid_data_directory'])])
    scene_trainer = SceneTrainer(
        model=model,  
        optimizer=optim, 
        loss_fun=loss_fun, 
        train_dataset=train_datasets, 
        eval_dataset=eval_datasets, 
        test_dataset=None, 
        optm_schedule=optm_schedule, 
        use_cuda=train_config['use_cuda'], 
        multy_gpu_type=train_config['multy_gpu_type'], 
        checkpoint_saving_dir=train_config['checkpoint_saving_dir'], 
        saving_dir=train_config['saving_dir'], 
        epochs=train_config['epochs'],
        batch_size=train_config['batch_size'],
        load_path=train_config['load_path'],
        # checkpoint=train_config['checkpoint']
    )
    
    scene_trainer.do_train()