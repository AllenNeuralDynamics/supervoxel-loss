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
from time import time
from torch.autograd import Variable
from concurrent.futures import ProcessPoolExecutor, as_completed

sigmoid = nn.Sigmoid()


class SuperVoxel(nn.Module):
    """
    Supervoxel-based topological loss function for training affinity-based
    neural networks to perform instance segmentation.

    """

    def __init__(
        self,
        alpha=0.5,
        beta=0.5,
        criterion=None,
        device=0,
        pred_threshold=0.5,
        return_mask=False,
    ):
        """
        Constructs a SuperVoxelLoss object.

        Parameters
        ----------
        criterion : torch.nn.modules.loss
            Loss function that is used to penalize critical components. If
            provided, you must set "reduction=None" so that the 
            The default value is None. The default value is 0.
        device : int, optional
            Device (e.g. cpu or gpu id) used to train model.

        Returns
        -------
        None

        """
        super(SuperVoxel, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.return_mask = return_mask
        self.threshold = pred_threshold
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.BCEWithLogitsLoss(
                reduction="none", pos_weight=torch.cuda.FloatTensor([1])
            )

    def forward(self, preds, targets):
        # Initializations
        loss = self.criterion(preds, targets)
        targets = toCPU(targets[:, 0, ...], return_numpy=True)
        preds = toCPU(preds[:, 0, ...], return_numpy=True)
        preds = self.binarize(preds)

        # Compute critical components
        masks = self.get_critical_masks(preds, targets)
        for i in range(preds.shape[0]):
            term_1 = (1 - self.alpha) * loss[i, ...]
            term_2 = self.alpha * masks[i, ...] * loss[i, ...]
            loss[i, ...] = term_1 + term_2
        return loss if self.return_mask else loss.mean()

    def binarize(self, arr):
        return (arr > self.threshold).astype(int)

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

    def get_critical_masks(self, preds, targets):
        processes = []
        # preds_flip = 1 - preds
        # targets_flip = 1 - targets
        masks = np.zeros(preds.shape)
        with ProcessPoolExecutor() as executor:
            for i in range(preds.shape[0]):
                pred, _ = label(preds[i, ...])
                target, _ = label(targets[i, ...])
                processes.append(
                    executor.submit(get_mask, target, pred, i, "neg")
                )
                processes.append(
                    executor.submit(get_mask, pred, target, i, "pos")
                )

                # Islands
                # target_flip, _ = label(targets_flip[i, ...])
                # pred_flip, _ = label(preds_flip[i, ...])
                # processes.append(executor.submit(get_mask, pred_flip, target_flip, i))

            for process in as_completed(processes):
                i, mask_i, neg_or_pos = process.result()
                if neg_or_pos == "neg":
                    masks[i, ...] += self.beta * mask_i
                else:
                    masks[i, ...] += (1 - self.beta) * mask_i
        return self.toGPU(masks)


def get_mask(target_labels, pred_labels, process_id, neg_or_pos):
    mask = detect_critical(target_labels, pred_labels)
    return process_id, mask, neg_or_pos


def toCPU(arr, return_numpy=False):
    if return_numpy:
        return np.array(arr.cpu().detach())
    else:
        return arr.detach().cpu()
