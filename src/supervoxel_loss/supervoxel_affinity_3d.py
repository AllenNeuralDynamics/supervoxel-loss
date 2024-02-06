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
from concurrent.futures import ProcessPoolExecutor, as_completed
from supervoxel_loss.critical_detection_3d import detect_critical
from toolz.itertoolz import last
from tifffile import imwrite
from time import time
from torch.autograd import Variable
from waterz import agglomerate as run_watershed


class SuperVoxelAffinity(nn.Module):
    """
    Supervoxel-based topological loss function for training affinity-based
    neural networks to perform instance segmentation.

    """

    def __init__(
        self,
        edges,
        alpha=0.5,
        beta=0.5,
        criterion=None,
        device=0,
        pred_threshold=0.5,
        return_cnts=False,
    ):
        """
        Constructs a SuperVoxelLoss object.

        Parameters
        ----------
        edges : list[tuple[int]]
            Edge affinities learned by model (e.g. edges=[[1, 0, 0],
            [0, 1, 0], [0, 0, 1]]).

        alpha : float, optional

        beta : float, optional

        criterion : torch.nn.modules.loss
            Loss function that is used to penalize critical components. If
            provided, must set "reduction=None". The default is None.
        device : int, optional
            Device (e.g. cpu or gpu id) used to train model.
        pred_threshold : float, optional

        return_cnts : bool, optional
            Indicates whether to return the number of negatively and
            positively critical components. The default is False.

        Returns
        -------
        None

        """
        super(SuperVoxelAffinity, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.decoder = SuperVoxelAffinity.Decoder(edges)
        self.device = device
        self.edges = list(edges)
        self.pred_threshold = pred_threshold
        self.return_cnts = return_cnts
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred_affs, target_labels):
        loss = 0
        pred_labels = self.get_pred_labels(pred_affs)
        masks, stats = self.get_critical_masks(pred_labels, target_labels)        
        for i in range(pred_affs.size(0)):
            mask_i = self.toGPU(masks[i, ...])
            target_labels_i = self.toGPU(target_labels[i, ...])
            for j, edge in enumerate(self.edges):
                # Compute affs
                pred_affs_j = self.decoder(pred_affs[i, ...], j)
                target_affs_j = get_aff(target_labels_i, edge)
                mask_aff_j = get_aff(mask_i, edge)

                # Compute loss
                loss_j = self.criterion(pred_affs_j, target_affs_j)
                term_1 = (1 - self.alpha) * loss_j
                term_2 = self.alpha * mask_aff_j * loss_j
                loss += (term_1 + term_2).mean()
        return loss, stats

    def binarize(self, pred):
        return (pred > self.pred_threshold).astype(np.float32)

    def get_pred_labels(self, pred_affs):
        pred_affs = toCPU(pred_affs, return_numpy=True)
        pred_labels = []
        for i in range(pred_affs.shape[0]):
            pred_labels.append(self.to_labels(pred_affs[i, ...]))
        return pred_labels

    def to_labels(self, pred_affs):
        iterator = run_watershed(self.binarize(pred_affs), [0])
        return next(iterator).astype(int)

    def get_critical_masks(self, preds, targets):
        processes = []
        stats = {"Splits": 0, "Merges": 0}
        masks = np.zeros((len(preds),) + preds[0].shape)
        targets = np.array(targets, dtype=int)
        with ProcessPoolExecutor() as executor:
            for i in range(len(preds)):
                processes.append(
                    executor.submit(
                        get_mask, targets[i, 0, ...], preds[i], i, "neg"
                    )
                )
                processes.append(
                    executor.submit(
                        get_mask, preds[i], targets[i, 0, ...], i, "pos"
                    )
                )
            for process in as_completed(processes):
                i, mask_i, n_criticals, neg_or_pos = process.result()
                if neg_or_pos == "neg":
                    masks[i, ...] += self.beta * mask_i
                    stats["Splits"] += n_criticals / len(preds)
                else:
                    masks[i, ...] += (1 - self.beta) * mask_i
                    stats["Merges"] += n_criticals / len(preds)
        return self.toGPU(masks), stats

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

    class Decoder(nn.Module):
        def __init__(self, edges):
            super(SuperVoxelAffinity.Decoder, self).__init__()
            self.edges = list(edges)

        def forward(self, x, i):
            num_channels = x.size(-4)
            assert num_channels == len(self.edges)
            assert i < num_channels and i >= 0
            return get_pair_first(x[..., [i], :, :, :], self.edges[i])


def toCPU(arr, return_numpy=False):
    if return_numpy:
        return np.array(arr.cpu().detach(), np.float32)
    else:
        return arr.detach().cpu()


def get_mask(target, pred, process_id, neg_or_pos):
    mask, n_criticals = detect_critical(target, pred)
    return process_id, mask, n_criticals, neg_or_pos


def get_aff(labels, edge):
    o1, o2 = get_pair(labels, edge)
    ret = (o1 == o2) & (o1 != 0)
    return ret.type(labels.type())


def get_pair_first(arr, edge):
    shape = arr.size()[-3:]
    edge = np.array(edge)
    os1 = np.maximum(edge, 0)
    os2 = np.maximum(-edge, 0)
    ret = arr[
        ...,
        os1[0] : shape[0] - os2[0],
        os1[1] : shape[1] - os2[1],
        os1[2] : shape[2] - os2[2],
    ]
    return ret


def get_pair(arr, edge):
    shape = arr.size()[-3:]
    edge = np.array(edge)
    os1 = np.maximum(edge, 0)
    os2 = np.maximum(-edge, 0)

    arr1 = arr[
        ...,
        os1[0] : shape[0] - os2[0],
        os1[1] : shape[1] - os2[1],
        os1[2] : shape[2] - os2[2],
    ]
    arr2 = arr[
        ...,
        os2[0] : shape[0] - os1[0],
        os2[1] : shape[1] - os1[1],
        os2[2] : shape[2] - os1[2],
    ]
    return arr1, arr2
