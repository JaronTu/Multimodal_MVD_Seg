from torch_topological.nn import CubicalComplex
from torch_topological.nn import WassersteinDistance, SlicedWassersteinDistance
from torch_topological.nn.data import batch_iter
import torch.nn as nn
import torch
# import gudhi

class Topological_loss(nn.Module):
    def __init__(self, topo_loss_q=2, topo_lambda=0.1):
        super().__init__()
        self.cubical_complex = CubicalComplex(
            dim=2,
            superlevel=False
        )
        self.topo_loss = WassersteinDistance(q=topo_loss_q)
        # self.topo_loss = SlicedWassersteinDistance()
        self.topo_lambda = topo_lambda

    def forward(self, pred, gt):
        # print(pred.squeeze().shape)
        # print(gt.shape)
        # pers_pred = self.cubical_complex(pred.squeeze())
        pers_pred = self.cubical_complex(pred)
        # pers_gt = self.cubical_complex(gt.squeeze())
        pers_gt = self.cubical_complex(gt)

        # print(pers_pred)
        # print(type(pers_gt))
        # print(type(pers_pred))
        # dim = 2
        # if dim is not None:
        pers_pred = [
            x for x in batch_iter(pers_pred)
        ]

        pers_gt = [
            x for x in batch_iter(pers_gt)
        ]
        # print("len",len(pers_gt))
        # print([pred_batch, true_batch] for pred_batch, true_batch in zip(pers_pred, pers_gt))
        topo_loss = torch.stack([
            self.topo_loss(pred_batch, true_batch)
            for pred_batch, true_batch in zip(pers_pred, pers_gt)
        ])
        topo_loss = topo_loss.mean()

        return self.topo_lambda * topo_loss