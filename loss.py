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

    def forward(self, pred, location, y_diff, mask=None):
        batch_size = pred.size()[0]
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
            # # diff_loss = diff_loss * mask.float()
            # diff_loss = torch.masked_select(diff_loss, mask)
            # pred_location = torch.cumsum(pred, dim=-2)
            
            ade_loss = torch.norm((pred - location), p=2, dim=-1)

            ade_loss = torch.masked_select(ade_loss, mask)
            
            fde_loss = (pred[:,:,-1,:] - location[:,:,-1,:]) ** 2
            fde_loss = fde_loss[:, :, 0] + fde_loss[:, :, 1]
            fde_loss = torch.sqrt(fde_loss)
            # fde_loss = fde_loss * mask[:, :, -1].float()
            fde_loss = torch.masked_select(fde_loss, mask[:, :, -1])
            return torch.mean(ade_loss), torch.mean(fde_loss)