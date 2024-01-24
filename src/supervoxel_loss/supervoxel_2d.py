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
from scipy.ndimage import label
from supervoxel_loss.critical_detection_2d import detect_critical
from toolz.itertoolz import last
from waterz import agglomerate as run_watershed
from torch.autograd import Variable


class SuperVoxel(nn.Module):
    """
    Supervoxel-based topological loss function for training affinity-based
    neural networks to perform instance segmentation.

    """
    def __init__(
        self,
        accuracy_threshold=0.5,
        alpha=1.0,
        beta=1.0,
        criterion=None,
        device=0,
        return_cnts=False,
    ):
        """
        Constructs a SuperVoxelLoss object.

        Parameters
        ----------
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
        super(SuperVoxel, self).__init__()
        self.accuracy_threshold = accuracy_threshold
        self.alpha = alpha
        self.beta = beta    
        self.device = device
        self.return_cnts = return_cnts
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, targets):
        loss = 0
        stats = {"Splits": [], "Merges": []}
        for i in range(preds.size(0)):
            # Check accuracy
            precision = self.get_precision(preds[i, ...], targets[i, ...])
            if precision < self.accuracy_threshold:
                loss += self.voxel_loss(preds[i, ...], targets[i, ...])
                continue

            # Detect critical components
            pred_labels_i, _ = label(toCPU(preds[i, 0, ...], return_numpy=True))
            target_labels_i, _ = label(toCPU(targets[i, 0, ...], return_numpy=True))
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
            loss_i = self.criterion(preds[i, ...], targets[i, ...])
            if self.alpha > 0:
                neg_critical_loss = self.alpha * neg_critical_mask * loss_i
            if self.beta > 0:
                pos_critical_loss = self.beta * pos_critical_mask * loss_i

            # Sum terms
            if self.alpha > 0:
                loss_i += neg_critical_loss
            if self.beta > 0:
                loss_i += pos_critical_loss
            loss += loss_i.mean()
        stats["Splits"] = np.mean(stats["Splits"])
        stats["Merges"] = np.mean(stats["Merges"])
        return loss, stats

    def get_precision(self, preds, targets):
        true_positive = torch.sum((preds > 0) & (targets > 0)).item()
        false_positive = torch.sum((preds > 0) & (targets == 0)).item()
        return true_positive / max((true_positive + false_positive), 1e-8)

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


def toCPU(arr, return_numpy=False):
    if return_numpy:
        return np.array(arr.cpu().detach())
    else:
        return arr.detach().cpu()


def binarize(arr, threshold=0):
    return (arr > threshold).astype(np.float32)
