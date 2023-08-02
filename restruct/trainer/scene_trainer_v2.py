from trainer.trainer import Trainer
import torch
import gc
import numpy as np

class SceneTrainer(Trainer):
    def __init__(self, model,  optimizer, loss_fun, train_dataset, eval_dataset, test_dataset, optm_schedule, use_cuda:bool, multy_gpu_type:str, checkpoint:str='', checkpoint_saving_dir:str='', saving_dir:str='', epochs:int = 20, load_path:str = "", batch_size:int=128, collate_fn=None):
        super().__init__(model, 
                    optimizer,
                    loss_fun,
                    train_dataset,
                    eval_dataset,
                    test_dataset,
                    optm_schedule,
                    use_cuda,
                    multy_gpu_type,
                    checkpoint,
                    checkpoint_saving_dir,
                    saving_dir,
                    epochs,
                    load_path,
                    batch_size, 
                    collate_fn)
    
    @torch.no_grad()
    def eval(self, epoch):
        total_loss = 0.0
        num_points = 0
        self.model.eval()
        
        for i, data in enumerate(self.eval_loader):
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
            
            y = data["y"]
            location = data["location"]
            y_mask = data["y_mask"]
            position_emb = self.getPositionEncoding(19, dim=128).unsqueeze(0).unsqueeze(0)
            
            if self.use_cuda and self.device.type == "cuda":
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

                y = y.cuda()
                location = location.cuda()
                y_mask = y_mask.cuda()
                
                
                position_emb = position_emb.cuda()            

            pred_trajs, probs = self.model(
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
                position_emb=position_emb
            )

            ade_loss, fde_loss, cls_loss = self.loss_fun(pred_trajs, probs, location, y, y_mask)
            loss = cls_loss + ade_loss + fde_loss
            points = torch.sum(y_mask.float()).item()
            num_points += points
            total_loss += loss.item() * points
            print("[Info:eval_Ep_{}_iter_{}: loss: {:.5e}; cls_loss: {:.5e}; ade_loss: {:.5e}; fde_loss: {:.5e}; avg_loss: {:.5e}]".format(epoch, 
                                                                                                 i, 
                                                                                                 loss.item(),
                                                                                                 cls_loss.item(),
                                                                                                 ade_loss.item(),
                                                                                                 fde_loss.item(),
                                                                                                 total_loss / num_points))
        torch.cuda.empty_cache()
        return  total_loss 
    
    def train(self, epoch):
        
        if self.multy_gpu_type == 'ddp':
            self.train_sample.set_epoch(epoch)
        total_loss = 0.0
        num_points = 0
        self.model.train()
        
        for i, data in enumerate(self.train_dataloader):
            self.optm_schedule.zero_grad()

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



            y = data["y"]
            location = data["location"]
            y_mask = data["y_mask"]
            
            position_emb = self.getPositionEncoding(19, dim=128).unsqueeze(0).unsqueeze(0)
            
            if self.use_cuda and self.device.type == "cuda":
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



                y = y.cuda()
                location = location.cuda()
                y_mask = y_mask.cuda()
            
                position_emb = position_emb.cuda()
                
            pred_trajs, probs = self.model(
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

                position_emb=position_emb
            )
            
            ade_loss, fde_loss, cls_loss = self.loss_fun(pred_trajs, probs, location, y, y_mask)
            loss = ade_loss + fde_loss + cls_loss
            loss.backward()
            self.optimizer.step()
            # self.optm_schedule.step()
            points = torch.sum(y_mask.float()).item()
            num_points += points
            total_loss += loss.item() * points
            print("[Info:train_Ep_{}_iter_{}: loss: {:.5e}; cls_loss: {:.5e}; ade_loss: {:.5e}; fde_loss: {:.5e}; avg_loss: {:.5e}]".format(epoch, 
                                                                                                 i, 
                                                                                                 loss.item(),
                                                                                                 cls_loss.item(),
                                                                                                 ade_loss.item(),
                                                                                                 fde_loss.item(),
                                                                                                 total_loss / num_points))
        self.optm_schedule.step() 
        return  total_loss

    def getPositionEncoding(self, seq_len,dim,n=10000):
        PE = np.zeros(shape=(seq_len, dim))
        for pos in range(seq_len):
            for i in range(int(dim/2)):
                denominator = np.power(n, 2*i/dim)
                PE[pos,2*i] = np.sin(pos/denominator)
                PE[pos,2*i+1] = np.cos(pos/denominator)

        return torch.from_numpy(PE).to(torch.float32)
    
     
    def do_train(self):
        super().do_train()