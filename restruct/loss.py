# loss function for train the model
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorLoss(nn.Module):
    """
        The loss function for train vectornet, Loss = L_traj + alpha * L_node
        where L_traj is the negative Gaussian log-likelihood loss, L_node is the huber loss
    """

    def __init__(self, reduction='mean'):
        super(VectorLoss, self).__init__()

        if reduction in ["mean", "sum"]:
            self.reduction = reduction
        else:
            raise NotImplementedError("[VectorLoss]: The reduction has not been implemented!")

    def forward(self, pred, y_diff):
        batch_size = pred.size()[0]
        loss = 0.0

        l_traj = F.mse_loss(pred, y_diff, reduction='sum')

        if self.reduction == 'mean':
            l_traj /= batch_size

        loss += l_traj

        return loss


class SceneLoss(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        if reduction in ["mean", "sum", "none"]:
            self.reduction = reduction
        else:
            raise NotImplementedError("[VectorLoss]: The reduction has not been implemented!")

        self.cls_loss = nn.CrossEntropyLoss()
        
    def forward(self, pred, probs, location, y_diff, mask=None):
        # pred = pred.permute(1, 0, 2, 3)
        num_modes = pred.size()[1]
        batch_size = pred.size()[0]
        pred_horizon = pred.size()[2]
        loss = 0.0
        if mask is None:
            l_traj = F.mse_loss(pred, y_diff, reduction='sum')

            if self.reduction == 'mean':
                l_traj /= batch_size

            loss += l_traj

            return loss
        else:

            # diff_loss = (pred - y_diff) ** 2
            # diff_loss = diff_loss[:, :, :, 0] + diff_loss[:, :, :, 1]
            # diff_loss = torch.sqrt(diff_loss)
            # diff_loss = torch.masked_select(diff_loss, mask)
            
            # pred_location = torch.cumsum(pred, dim=-2)
            ade_loss = (pred - location) ** 2
            ade_loss = ade_loss[:, :, :, 0] + ade_loss[:, :, :, 1]
            ade_loss = torch.sqrt(ade_loss)
            # ade_loss = torch.masked_fill(ade_loss, ~mask, 0)
            # ade_loss = torch.masked_select(ade_loss, mask)
            ade_loss = torch.sum(ade_loss, dim=-1) / pred_horizon
            ade_loss, min_idx = torch.min(ade_loss, dim=-1)
            
            # labels = torch.zeros(batch_size,num_modes).scatter_(1, min_idx.reshape(batch_size,1), 1).long()
            cls_loss = self.cls_loss(probs, min_idx.long())
            
            fde_loss = (pred[torch.arange(batch_size), min_idx,-1,:] - location[:,:,-1,:].squeeze()) ** 2
            fde_loss = torch.sum(fde_loss, dim=-1)
            fde_loss = torch.sqrt(fde_loss)
            return torch.mean(ade_loss), torch.mean(fde_loss), cls_loss