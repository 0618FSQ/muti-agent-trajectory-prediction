import os
import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, 
                 model, 
                 optimizer,
                 loss_fun,
                 train_dataset,
                 eval_dataset,
                 test_dataset,
                 optm_schedule,
                 use_cuda:bool,
                 multy_gpu_type:str,
                 checkpoint:str=None,
                 checkpoint_saving_dir=None,
                 saving_dir=None,
                 epochs = 20,
                 load_path:str = "",
                 batch_size:int=128,
                 collate_fn=None
                 ):
        """_summary_

        Args:
            model (_type_): 模型类
            optimizer (_type_): 优化器
            loss_fun (_type_): 损失函数
            train_dataset (_type_): 预测集
            eval_dataset (_type_): 验证集
            test_dataset (_type_): 测试集
            optm_schedule (_type_): _description_
            use_cuda (bool): 使用cuda
            multy_gpu_type (str): gpu使用类型： no_use, single_gpu, dp, ddp之一
            checkpoint (str, optional): 断电
            checkpoint_saving_dir (_type_, optional): 存放断点位置
            saving_dir (_type_, optional): 存放模型文件位置
            epochs (int, optional): 迭代轮数
            load_path (str, optional): 加载模型文件位置
            batch_size (int, optional): batch size
        """
        assert multy_gpu_type in ["no_use", "single_gpu", "dp", "ddp"]
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.optm_schedule = optm_schedule
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.use_cuda = use_cuda
        self.multy_gpu_type = multy_gpu_type
        self.checkpoint = checkpoint
        self.checkpoint_saving_dir = checkpoint_saving_dir
        self.saving_dir=saving_dir
        self.epochs = epochs
        self.load_path = load_path
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.__init()
        
    def __init(self):
        # if self.checkpoint:
        #     if not os.path.exists(self.checkpoint):
        #         raise RuntimeWarning("The checkpoint does not exist")
        #     else:
        #         self.load_checkpoint()
        # if self.load_path:
        #     if not os.path.exists(self.load_path):
        #         raise RuntimeWarning("The model path does not exist")
        #     else:
        #         self.load()
        if self.use_cuda:
            if torch.cuda.is_available():
                if self.multy_gpu_type == "no_use":
                    self.device = torch.device("cpu")
                    self.train_dataloader = DataLoader(
                        self.train_dataset, 
                        batch_size=self.batch_size,
                        collate_fn=self.collate_fn
                    )
                    self.eval_loader = DataLoader(
                        self.eval_dataset, 
                        batch_size=self.batch_size
                        
                    )
                    if self.checkpoint:
                        if not os.path.exists(self.checkpoint):
                            raise RuntimeWarning("The checkpoint does not exist")
                        else:
                            self.load_checkpoint()
                    if self.load_path:
                        if not os.path.exists(self.load_path):
                            raise RuntimeWarning("The model path does not exist")
                        else:
                            self.load()
                            
                elif self.multy_gpu_type == "single_gpu":
                    self.device = torch.device("cuda", 0)
                    self.train_dataloader = DataLoader(
                        self.train_dataset, 
                        batch_size=self.batch_size,
                        collate_fn=self.collate_fn
                        
                    )
                    self.eval_loader = DataLoader(
                        self.eval_dataset, 
                        batch_size=self.batch_size,
                        collate_fn=self.collate_fn
                        
                    )
                    self.model = self.model.cuda()
                    if self.checkpoint:
                        if not os.path.exists(self.checkpoint):
                            raise RuntimeWarning("The checkpoint does not exist")
                        else:
                            self.load_checkpoint()
                    if self.load_path:
                        if not os.path.exists(self.load_path):
                            raise RuntimeWarning("The model path does not exist")
                        else:
                            self.load()                    

                elif self.multy_gpu_type == "dp":
                    self.device = torch.device("cuda", 0)
                    self.train_dataloader = DataLoader(
                        self.train_dataset, 
                        batch_size=self.batch_size,
                        collate_fn=self.collate_fn,
                        shuffle=True
                    )
                    self.eval_loader = DataLoader(
                        self.eval_dataset, 
                        batch_size=self.batch_size,
                        collate_fn=self.collate_fn
                        
                    )
                    self.model = torch.nn.DataParallel(self.model)
                    if self.checkpoint:
                        if not os.path.exists(self.checkpoint):
                            raise RuntimeWarning("The checkpoint does not exist")
                        else:
                            self.load_checkpoint()
                    if self.load_path:
                        if not os.path.exists(self.load_path):
                            raise RuntimeWarning("The model path does not exist")
                        else:
                            self.load()     
                            
                elif self.multy_gpu_type == "ddp":
                    torch.distributed.init_process_group(backend="nccl")
                    local_rank = torch.distributed.get_rank()
                    torch.cuda.set_device(local_rank)
                    self.device = torch.device("cuda", local_rank)
                    self.model = self.model.cuda()
                    self.model = torch.nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=True)

                    self.train_sample = DistributedSampler(self.train_dataset, shuffle=True)
                    self.train_dataloader = DataLoader(
                        self.train_dataset, 
                        sampler=self.train_sample, 
                        batch_size=self.batch_size,
                        collate_fn=self.collate_fn
                        
                    )
                    self.eval_sample = DistributedSampler(self.eval_dataset)
                    self.eval_loader = DataLoader(
                        self.eval_dataset, 
                        sampler=self.eval_sample, 
                        batch_size=self.batch_size,
                        collate_fn=self.collate_fn
                        
                    )
                    if self.checkpoint:
                        if not os.path.exists(self.checkpoint):
                            raise RuntimeWarning("The checkpoint does not exist")
                        else:
                            self.load_checkpoint()
                    if self.load_path:
                        if not os.path.exists(self.load_path):
                            raise RuntimeWarning("The model path does not exist")
                        else:
                            self.load()     
                            
                else:
                    raise RuntimeError(f"There is no {self.multy_gpu_type}")
    
            else:
                self.train_dataloader = DataLoader(
                        self.train_dataset, 
                        batch_size=self.batch_size,
                        collate_fn=self.collate_fn
                        
                )
                self.eval_loader = DataLoader(
                        self.eval_dataset, 
                        batch_size=self.batch_size,
                        collate_fn=self.collate_fn
                )
                self.device = torch.device("cpu")
        else:
            self.train_dataloader = DataLoader(self.train_dataset, 
                                               batch_size=self.batch_size, 
                                               collate_fn=self.collate_fn,
                                               shuffle=True)
            self.eval_loader = DataLoader(self.eval_dataset,
                                          batch_size=self.batch_size,
                                          
                                          collate_fn=self.collate_fn
                                          )
            self.device = torch.device("cpu")



    def save_checkpoint(self, epoch):
        if not os.path.exists(self.checkpoint_saving_dir):
            os.makedirs(self.checkpoint_saving_dir, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict() if self.multy_gpu_type not in ["dp", "ddp"] else self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, os.path.join(self.checkpoint_saving_dir, "checkpoint_iter_{}.ckpt".format(epoch)))
    
    
    def train(self, epoch):
        raise NotImplementedError("Please implements the train function")
    def eval(self, epoch):
        raise NotImplementedError("Please implements the eval function")

    def save(self, prefix):
        if not os.path.exists(self.saving_dir):
            os.makedirs(self.saving_dir, exist_ok=True)
        torch.save(
            self.model.state_dict() if self.multy_gpu_type not in ["dp", "ddp"] else self.model.module.state_dict(),
            os.path.join(self.saving_dir, "{}_{}.pth".format(prefix, type(self.model).__name__))
        )
    def do_train(self):
        min_loss = np.inf
        for epoch in range(self.epochs):
            train_loss = self.train(epoch)
            eval_loss = self.eval(epoch)
            self.save_checkpoint(epoch)
            self.save(epoch)
            if min_loss >= eval_loss:
                self.save('best')
                min_loss = eval_loss

    def load_checkpoint(self) -> bool:
        try:
            ckpt = torch.load(self.checkpoint, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except:
            raise RuntimeWarning("Loading checkpoint fails, and We do not load the file")
        finally:
            return True
        
    def load(self) -> bool:
        try:
            self.model.load_state_dict(torch.load(self.load_path, map_location=self.device))
        except:
            raise RuntimeWarning("Loading model file fails,  and We do not load the file")
        finally:
            return True