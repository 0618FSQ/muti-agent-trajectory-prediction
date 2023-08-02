from trainer.trainer import Trainer
import torch
import gc


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
    
    def cross_mask(self, x, y, x_valid_lens, y_valid_lens):
        assert x.shape[0] == y.shape[0]
        mask = torch.zeros(size=[x.shape[0], x.shape[1], y.shape[1]], dtype=torch.bool)
        for batch_id, (cnt1, cnt2) in enumerate(zip(x_valid_lens, y_valid_lens)):
            cnt1 = int(cnt1.detach().cpu().numpy())
            cnt2 = int(cnt2.detach().cpu().numpy())
            mask[batch_id, :, cnt2:] = True
            mask[batch_id, cnt1:] = True
        return mask
    
    def self_attention_mask(self, x, valid_lens):
        shape = [*x.shape[:2], x.shape[1]]
        mask = torch.zeros(size=shape, dtype=torch.bool)
        for batch_id, cnt in enumerate(valid_lens):
            cnt = int(cnt.detach().cpu().numpy())
            mask[batch_id, :, cnt:] = True
            mask[batch_id, cnt:] = True
        return mask
    
    def target_self_mask(self, y, valid_lens):
        shape = [*y.shape[:3]]
        mask = torch.ones(size=shape, dtype=torch.bool)
        for batch_id, cnt in enumerate(valid_lens):
            cnt = int(cnt.detach().cpu().numpy())
            mask[batch_id, cnt:, :] = False
        return mask
    
    @torch.no_grad()
    def eval(self, epoch):
        total_loss = 0.0
        num_points = 0
        self.model.eval()
        
        for i, data in enumerate(self.train_dataloader):
            self.optm_schedule.zero_grad()
            target_agent_history_trajectory = data['target_agent_history_trajectory']
            target_agent_orig = data['target_agent_orig']
            target_agent_history_trajectory_mask = data['target_agent_history_trajectory_mask']
            graph_padding = data['graph_padding']
            graph_mask = data['graph_mask']
            graph_length = data['graph_length']
            agent_length = data['agent_length']
            y = data['y']
            y_mask = data['y_mask']
            
            cross_agents_graph_mask = self.cross_mask(target_agent_history_trajectory, graph_padding, agent_length,graph_length)
            future_traj_cross_graph_mask = self.cross_mask(y, graph_padding, agent_length, graph_length)
            history_and_future_mask = self.self_attention_mask(target_agent_history_trajectory, agent_length)

            agents_cross_mask = self.self_attention_mask(target_agent_history_trajectory, agent_length)
            graph_cross_mask = self.self_attention_mask(graph_padding, graph_length)
            
            future_mask = self.target_self_mask(y, agent_length)
            if self.use_cuda and self.device.type == "cuda":
                target_agent_history_trajectory = target_agent_history_trajectory.cuda()
                target_agent_orig = target_agent_orig.cuda()
                target_agent_history_trajectory_mask = target_agent_history_trajectory_mask.cuda()
                graph_padding = graph_padding.cuda()
                graph_mask = graph_mask.cuda()
                agents_cross_mask = agents_cross_mask.cuda()
                graph_cross_mask = graph_cross_mask.cuda()
                cross_agents_graph_mask = cross_agents_graph_mask.cuda()
                future_traj_cross_graph_mask = future_traj_cross_graph_mask.cuda()
                history_and_future_mask = history_and_future_mask.cuda()
                future_mask = future_mask.cuda()
                y = y.cuda()
                y_mask = y_mask.cuda()
            
            out = self.model(
                target_agent_history_trajectory,
                target_agent_orig,
                target_agent_history_trajectory_mask,
                graph_padding,
                graph_mask,
                agents_cross_mask,
                graph_cross_mask,
                cross_agents_graph_mask,
                future_traj_cross_graph_mask,
                history_and_future_mask,
                future_mask
            )
            loss = self.loss_fun(out, y, y_mask)
            points = torch.sum(y_mask.float()).item()
            num_points += points
            total_loss += loss.item() * points
            print("[Info: Device_0: eval_Ep_{}_iter_{}: loss: {:.5e}; avg_loss: {:.5e}]".format(epoch, 
                                                                                                 i, 
                                                                                                 loss.item(),
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
            target_agent_history_trajectory = data['target_agent_history_trajectory']
            target_agent_orig = data['target_agent_orig']
            target_agent_history_trajectory_mask = data['target_agent_history_trajectory_mask']
            graph_padding = data['graph_padding']
            graph_mask = data['graph_mask']
            graph_length = data['graph_length']
            agent_length = data['agent_length']
            y = data['y']
            y_mask = data['y_mask']
            
            cross_agents_graph_mask = self.cross_mask(target_agent_history_trajectory, graph_padding, agent_length, graph_length)
            future_traj_cross_graph_mask = self.cross_mask(y, graph_padding, agent_length, graph_length)
            history_and_future_mask = self.self_attention_mask(target_agent_history_trajectory, agent_length)

            agents_cross_mask = self.self_attention_mask(target_agent_history_trajectory, agent_length)
            graph_cross_mask = self.self_attention_mask(graph_padding, graph_length)
            
            future_mask = self.target_self_mask(y, agent_length)
            if self.use_cuda and self.device.type == "cuda":
                target_agent_history_trajectory = target_agent_history_trajectory.cuda()
                target_agent_orig = target_agent_orig.cuda()
                target_agent_history_trajectory_mask = target_agent_history_trajectory_mask.cuda()
                graph_padding = graph_padding.cuda()
                graph_mask = graph_mask.cuda()
                agents_cross_mask = agents_cross_mask.cuda()
                graph_cross_mask = graph_cross_mask.cuda()
                cross_agents_graph_mask = cross_agents_graph_mask.cuda()
                future_traj_cross_graph_mask = future_traj_cross_graph_mask.cuda()
                history_and_future_mask = history_and_future_mask.cuda()
                future_mask = future_mask.cuda()
                y = y.cuda()
                y_mask = y_mask.cuda()
            
            out = self.model(
                target_agent_history_trajectory,
                target_agent_orig,
                target_agent_history_trajectory_mask,
                graph_padding,
                graph_mask,
                agents_cross_mask,
                graph_cross_mask,
                cross_agents_graph_mask,
                future_traj_cross_graph_mask,
                history_and_future_mask,
                future_mask
            )
            loss = self.loss_fun(out, y, y_mask)
            loss.backward()
            self.optimizer.step()
            # self.optm_schedule.step()
            points = torch.sum(y_mask.float()).item()
            num_points += points
            total_loss += loss.item() * points
            print("[Info: Device_0: train_Ep_{}_iter_{}: loss: {:.5e}; avg_loss: {:.5e}]".format(epoch, 
                                                                                                 i, 
                                                                                                 loss.item(),
                                                                                                 total_loss / num_points))
        return  total_loss
            
    def do_train(self):
        super().do_train()