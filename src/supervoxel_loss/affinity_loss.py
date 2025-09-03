"""
Created on Fri November 17 22:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


This module implements a supervoxel-based topological loss function designed
for training affinity-based neural networks to perform instance segmentation.
The loss function penalizes voxel- and structure-level mistakes by focusing on
"critical" regions in the prediction.

The core idea is to calculate the loss based on two key components:
    1. Voxel-Level Mistakes
        These occur when the predicted segmentation differs from the ground
        truth at the voxel-level.

    2. Structure-Level Mistakes
        These occur when the structure of the predicted segmentation differs
        from the ground truth, such as over- or under-segmenting individual
        objects or incorrectly merging objects.

The module only supports 3D segmentation tasks.

"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.autograd import Variable
from waterz import agglomerate as run_watershed

import numpy as np
import torch
import torch.nn as nn

from supervoxel_loss.critical_detection_3d import detect_critical


class SuperVoxelAffinityLoss(nn.Module):
    """
    Supervoxel-based loss function for training neural networks to perform
    affinity-based instance segmentation.
    """

    def __init__(
        self,
        edges,
        alpha=0.5,
        beta=0.5,
        criterion=nn.BCEWithLogitsLoss(reduction="none"),
        device=0,
        threshold=0.5,
    ):
        """
        Instantiates SuperVoxelAffinityLoss object with the given parameters.

        Parameters
        ----------
        edges : List[Tuple[int]]
            Edge affinities learned by model (e.g. [[1, 0, 0], [0, 1, 0],
            [0, 0, 1]]).
        alpha : float, optional
            Scaling factor between 0 and 1 that determines the relative
            importance of voxel- versus structure-level mistakes. Default
            is 0.5.
        beta : float, optional
            Scaling factor between 0 and 1 that determines the relative
            importance of split versus merge mistakes. Default is 0.5.
        criterion : torch.nn.modules.loss, optional
            Loss function used to penalize voxel- and structure-level
            mistakes. If provided, must set "reduction=None". Default is
            nn.BCEWithLogitsLoss.
        device : str, optional
            Device on which to train model. Default is "cuda".
        threshold : float, optional
            Theshold used to binarize predictions. Defulat is 0.5.
        """
        # Call parent class
        super(SuperVoxelAffinityLoss, self).__init__()

        # Instance attributes
        self.alpha = alpha
        self.beta = beta
        self.criterion = criterion
        self.decoder = SuperVoxelAffinityLoss.Decoder(edges)
        self.device = device
        self.edges = list(edges)
        self.threshold = threshold

    def forward(self, pred_affs, target_labels):
        """
        Computes the loss for a batch by comparing predictions and ground
        truth.

        Parameters
        ----------
        pred_affs : torch.Tensor
            Predicted affinities with shape (batch_size, num_edges, height,
            width, depth).
        target_labels : torch.Tensor
            Target labels with shape (batch_size, height, width, depth)
            representing the ground truth labels. Note: target labels are
            converted into affinities to compute loss.

        Returns
        -------
        torch.Tensor
            Computed loss for the given batch.
        """
        # Critical components
        pred_labels = self.affs_to_labels(pred_affs)
        critical_masks = self.get_critical_masks_for_batch(
            pred_labels, target_labels
        )

        # Loss
        loss = 0
        for i in range(pred_affs.size(0)):
            critical_masks_i = self.toGPU(critical_masks[i, ...])
            target_labels_i = self.toGPU(target_labels[i, ...])
            for j, edge in enumerate(self.edges):
                # Convert to affinities
                pred_affs_j = self.decoder(pred_affs[i, ...], j)
                target_affs_j = get_aff(target_labels_i, edge)
                critical_mask_aff_j = get_aff(critical_masks_i, edge)

                # Affinity loss
                loss_j = self.criterion(pred_affs_j, target_affs_j)
                supervoxel_loss_j = critical_mask_aff_j * loss_j
                loss += (
                    (1 - self.alpha) * loss_j + self.alpha * supervoxel_loss_j
                ).mean()
        return loss

    def get_critical_masks_for_batch(self, preds, targets):
        """
        Computes critical components for each example in the given batch.

        Parameters
        ----------
        preds : torch.Tensor
            Predictions with the shape (batch_size, 1, height, width, *depth).
            This tensor should contain raw output probabilities (logits) from
            the model.
        targets : torch.Tensor
            Ground truth segmentations with the shape (batch_size, 1, height,
            width, *depth).

        Returns
        -------
        torch.Tensor
            Binary masks that identify critical components.
        """
        with ProcessPoolExecutor() as executor:
            # Assign processes
            processes = []
            targets = np.array(targets, dtype=int)
            for i in range(len(preds)):
                processes.append(
                    executor.submit(
                        self.get_critical_mask,
                        preds[i],
                        targets[i, 0, ...],
                        i,
                        -1,
                    )
                )
                processes.append(
                    executor.submit(
                        self.get_critical_mask,
                        targets[i, 0, ...],
                        preds[i],
                        i,
                        1,
                    )
                )

            # Store results
            masks = np.zeros((len(preds),) + preds[0].shape)
            for process in as_completed(processes):
                i, mask_i = process.result()
                masks[i, ...] = mask_i
        return self.toGPU(masks)

    def get_critical_mask(self, pred, target, process_id, critical_type):
        """
        Compute the critical mask for the given examples.

        Parameters
        ----------
        pred : numpy.ndarray
            Predicted segmentation.
        target : numpy.ndarray
            Ground truth segmentation.
        process_id : int
            Index of the given example from a batch.
        critical_type : int
            Indication of whether to compute positive or negative critical
            components.

        Returns
        -------
        tuple
            A tuple containing the following:
            - "process_id": Index of the given example from a batch.
            - "mask": Binary mask that identifies the critical components.
        """
        critical_mask = detect_critical(target, pred)
        scaling_factor = (1 - self.beta) if critical_type > 0 else self.beta
        return process_id, scaling_factor * critical_mask

    # --- Miscellanenous ---
    def affs_to_labels(self, affs):
        """
        Converts predicted affinities to predicted labels by decoding the
        affinities.

        Parameters
        ----------
        affs : torch.Tensor
            Tensor containing predicted affinities from a batch.

        Returns
        -------
        List[numpy.ndarray]
            Predicted labels for each example in the batch.
        """
        affs = np.array(affs.detach().cpu(), np.float32)
        labels = []
        for i in range(affs.shape[0]):
            binary_affs = (affs[i, ...] > self.threshold).astype(np.float32)
            iterator = run_watershed(binary_affs, [0])
            labels.append(next(iterator).astype(int))
        return labels

    def toGPU(self, arr):
        """
        Converts the given array to a tensor and moves it to a GPU device.

        Parameters
        ----------
        arr : numpy.array
            Array to be converted to a tensor and moved to GPU.

        Returns
        -------
        torch.tensor
            Tensor on GPU.
        """
        arr[np.newaxis, ...] = arr
        arr = torch.from_numpy(arr)
        return Variable(arr).to(self.device, dtype=torch.float32)

    class Decoder(nn.Module):
        """
        Decoder module for processing edge affinities.
        """

        def __init__(self, edges):
            """
            Instantiates Decoder object with the given edge affinities.

            Parameters
            ----------
            edges : List[Tuple[int]]
                Edge affinities learned by model (e.g. [(1, 0, 0]), (0, 1, 0),
                (0, 0, 1)]).
            """
            super(SuperVoxelAffinityLoss.Decoder, self).__init__()
            self.edges = list(edges)

        def forward(self, affs, i):
            """
            Extracts the predicted affinity for the i-th edge from the input
            tensor.

            Parameters
            ----------
            affs : torch.Tensor
                Predicted affinities for a single example.
            i : int
                Index of specific edge in "self.edges".

            Returns
            -------
            torch.Tensor
                Affinities corresponding to the i-th edge.
            """
            n_channels = affs.size(-4)
            assert n_channels == len(self.edges)
            assert i < n_channels and i >= 0
            return get_pair_first(affs[..., [i], :, :, :], self.edges[i])


# --- Helpers ---
def get_aff(labels, edge):
    """
    Computes affinities for labels based on the given edge.

    Parameters
    ----------
    labels : torch.Tensor
        Tensor containing the segmentation labels for a single example.
    edge : Tuple[int]
        Edge affinity.

    Returns
    -------
    torch.Tensor
        Binary tensor, where each element indicates the affinity for each
        voxel based on the given edge.
    """
    o1, o2 = get_pair(labels, edge)
    ret = (o1 == o2) & (o1 != 0)
    return ret.type(labels.type())


def get_pair(labels, edge):
    """
    Extracts two subarrays from "labels" by using the given edge affinity as
    an offset.

    Parameters
    ----------
    labels : torch.Tensor
        Tensor containing the segmentation labels for a single example.
    edge : Tuple[int]
        Edge affinity.

    Returns
    -------
    tuple of torch.Tensor
        A tuple containing two tensors:
        - "arr1": Subarray extracted based on the edge affinity.
        - "arr2": Subarray extracted based on the negative of the edge
                  affinity.
    """
    shape = labels.size()[-3:]
    edge = np.array(edge)
    offset1 = np.maximum(edge, 0)
    offset2 = np.maximum(-edge, 0)

    labels1 = labels[
        ...,
        offset1[0]: shape[0] - offset2[0],
        offset1[1]: shape[1] - offset2[1],
        offset1[2]: shape[2] - offset2[2],
    ]
    labels2 = labels[
        ...,
        offset2[0]: shape[0] - offset1[0],
        offset2[1]: shape[1] - offset1[1],
        offset2[2]: shape[2] - offset1[2],
    ]
    return labels1, labels2


def get_pair_first(labels, edge):
    """
    Gets subarray of "labels" based on the given edge affinity which defines
    an offset. Note this subarray will be used to compute affinities.

    Parameters
    ----------
    labels : torch.Tensor
        Segmentation labels for a single example.
    edge : Tuple[int]
        Edge affinity that defines the offset of the subarray.

    Returns
    -------
    torch.Tensor
        Subarray of "labels" based on the given edge affinity.
    """
    shape = labels.size()[-3:]
    edge = np.array(edge)
    offset1 = np.maximum(edge, 0)
    offset2 = np.maximum(-edge, 0)
    ret = labels[
        ...,
        offset1[0]: shape[0] - offset2[0],
        offset1[1]: shape[1] - offset2[1],
        offset1[2]: shape[2] - offset2[2],
    ]
    return ret
