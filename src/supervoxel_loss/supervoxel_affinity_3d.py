"""
Created on Fri November 17 22:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Supervoxel-based topological loss function for training affinity-based neural
networks to perform instance segmentation.

"""

import numpy as np
import torch
import torch.nn as nn
from supervoxel_loss.critical_detection_3d import detect_critical
from toolz.itertoolz import last
from waterz import agglomerate as run_watershed
from torch.autograd import Variable


class SuperVoxelAffinity(nn.Module):
    """
    Supervoxel-based topological loss function for training affinity-based
    neural networks to perform instance segmentation.

    """
    def __init__(
        self,
        edges,
        accuracy_threshold=0.5,
        alpha=10.0,
        beta=10.0,
        criterion=None,
        device=0,
        return_cnts=False,
    ):
        """
        Constructs a SuperVoxelLoss object.

        Parameters
        ----------
        edges : list[tuple[int]]
            Edge affinities learned by model (e.g. edges=[[1, 0, 0], 
            [0, 1, 0], [0, 0, 1]]).
        accuracy_threshold : float, optional
            Minimum precision obtained by model which is required to compute
            topological terms in loss. The default value is 0.5.
        alpha : float, optional
            Weight penalty applied to negatively critical components.
            The default value is 10.0.
        beta : float, optional
            Weight penalty appplied to positively critical components.
            The default value is 10.0.
        criterion : torch.nn.modules.loss
            Loss function that is used to penalize critical components. If
            provided, you must set "reduction=None" so that the 
            The default value is None. The default value is 0.
        device : int, optional
            Device (e.g. cpu or gpu id) used to train model.
        return_cnts : bool, optional
            Indicates whether to return the number of negatively and
            positively critical components.
            The default value is False.

        Returns
        -------
        None

        """
        super(SuperVoxelAffinity, self).__init__()
        self.accuracy_threshold = accuracy_threshold
        self.alpha = alpha
        self.beta = beta    
        self.decoder = SuperVoxelAffinity.Decoder(edges)
        self.device = device
        self.edges = list(edges)
        self.return_cnts = return_cnts
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, target_labels):
        loss = 0
        stats = {"Splits": [], "Merges": []}
        for i in range(preds.size(0)):
            # Check accuracy
            precision = self.get_precision(preds[i, ...], target_labels[i, ...])
            if precision < self.accuracy_threshold:
                loss += self.voxel_loss(preds[i, ...], target_labels[i, ...])
                continue

            # Detect critical components
            pred_labels_i = self.get_pred_labels(preds[i, ...])
            target_labels_i = np.array(target_labels[i, 0, ...], dtype=int)
            if self.alpha > 0:
                neg_critical_mask, num_splits = detect_critical(
                    target_labels_i, pred_labels_i
                )
                stats["Splits"].append(num_splits)

            if self.beta > 0:
                pos_critical_mask, num_merges = detect_critical(
                    pred_labels_i, target_labels_i
                )
                stats["Merges"].append(num_merges)

            # Compute supervoxel loss
            neg_critical_mask = self.toGPU(neg_critical_mask)
            pos_critical_mask = self.toGPU(pos_critical_mask)
            target_labels_i = self.toGPU(target_labels[i, ...])
            for j, edge in enumerate(self.edges):
                # Get affinities
                target_affs_j = get_aff(target_labels_i, edge)
                pred_affs_j = self.decoder(preds[i, ...], j)
                neg_aff_j = get_aff(neg_critical_mask, edge)
                pos_aff_j = get_aff(pos_critical_mask, edge)

                # Compute terms
                loss_i = self.criterion(pred_affs_j, target_affs_j)
                if self.alpha > 0:
                    neg_critical_loss = self.alpha * neg_aff_j * loss_i
                if self.beta > 0:
                    pos_critical_loss = self.beta * pos_aff_j * loss_i

                # Sum terms
                if self.alpha > 0:
                    loss_i += neg_critical_loss
                if self.beta > 0:
                    loss_i += pos_critical_loss
                loss += 100 * loss_i.mean()
        stats["Splits"] = np.mean(stats["Splits"])
        stats["Merges"] = np.mean(stats["Merges"])
        return loss, stats

    def get_precision(self, preds, target_labels):
        target = get_aff(target_labels, self.edges[0])
        pred = toCPU(self.decoder(preds, 0))
        true_positive = torch.sum((pred > 0) & (target > 0)).item()
        false_positive = torch.sum((pred > 0) & (target == 0)).item()
        return true_positive / max((true_positive + false_positive), 1e-8)

    def get_pred_labels(self, preds):
        pred_affs = binarize(toCPU(preds, return_numpy=True))
        iterator = run_watershed(pred_affs, [0])
        return next(iterator).astype(int)
        
    def get_pred_affs(self, preds):
        return [self.decoder(preds, i) for i in range(3)]

    def toGPU(self, arr):
        """
        Convert "arr" to a tensor and then move to GPU.

        Parameters
        ----------
        arr : torch.array
            Array to be converted to a tensor and moved to GPU.

        Returns
        -------
        torch.tensor
            Tensor on GPU.
        """
        if type(arr) == np.ndarray:
            arr[np.newaxis, ...] = arr
            arr = torch.from_numpy(arr)
        return Variable(arr).to(self.device, dtype=torch.float32)

    def voxel_loss(self, preds, target_labels):
        loss = 0
        target_labels = self.toGPU(target_labels)
        for j, edge in enumerate(self.edges):
            target_affs_j = get_aff(target_labels, edge)
            pred_affs_j = self.decoder(preds, j)
            loss += 100 * self.criterion(pred_affs_j, target_affs_j).mean()
        return loss

    class Decoder(nn.Module):
        def __init__(self, edges):
            super(SuperVoxelAffinity.Decoder, self).__init__()
            self.edges = list(edges)

        def forward(self, x, i):
            num_channels = x.size(-4)
            assert num_channels == len(self.edges)
            assert i < num_channels and i >= 0
            return get_pair_first(
                x[..., [i], :, :, :], self.edges[i]
            )

def toCPU(arr, return_numpy=False):
    if return_numpy:
        return np.array(arr.cpu().detach())
    else:
        return arr.detach().cpu()


def get_aff(labels, edge):
    o1, o2 = get_pair(labels, edge)
    ret = (o1 == o2) & (o1 != 0)
    return ret.type(labels.type())


def binarize(arr, threshold=0):
    return (arr > threshold).astype(np.float32)


def get_pair_first(arr, edge):
    shape = arr.size()[-3:]
    edge = np.array(edge)
    os1 = np.maximum(edge, 0)
    os2 = np.maximum(-edge, 0)
    ret = arr[..., os1[0]:shape[0]-os2[0],
                   os1[1]:shape[1]-os2[1],
                   os1[2]:shape[2]-os2[2]]
    return ret


def get_pair(arr, edge):
    shape = arr.size()[-3:]
    edge = np.array(edge)
    os1 = np.maximum(edge, 0)
    os2 = np.maximum(-edge, 0)

    arr1 = arr[..., os1[0]:shape[0]-os2[0],
                    os1[1]:shape[1]-os2[1],
                    os1[2]:shape[2]-os2[2]]
    arr2 = arr[..., os2[0]:shape[0]-os1[0],
                    os2[1]:shape[1]-os1[1],
                    os2[2]:shape[2]-os1[2]]    
    return arr1, arr2
