"""
Created on Fri November 17 22:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Implementation of a supervoxel-based loss function for training affinity-based
neural networks to perform instance segmentation.

Note: We use the term "labels" to refer to a segmentation mask.

"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.autograd import Variable
from waterz import agglomerate as run_watershed

import numpy as np
import torch
import torch.nn as nn

from supervoxel_loss.critical_detection_3d import detect_critical


class SuperVoxelAffinity(nn.Module):
    """
    Supervoxel-based loss function for training affinity-based neural networks
    to perform instance segmentation.

    """

    def __init__(
        self,
        edges,
        alpha=0.5,
        beta=0.5,
        criterion=None,
        device=0,
        threshold=0.5,
        return_cnts=False,
    ):
        """
        Constructs a SuperVoxelLoss object.

        Parameters
        ----------
        edges : List[Tuple[int]]
            Edge affinities learned by model (e.g. [[1, 0, 0], [0, 1, 0],
            [0, 0, 1]]).
        alpha : float, optional
            Scaling factor that controls the relative importance of voxel-
            level versus structure-level mistakes. The default is 0.5.
        beta : float, optional
            Scaling factor that controls the relative importance of split
            versus merge mistakes. The default is 0.5.
        criterion : torch.nn.modules.loss
            Loss function used to penalize voxel- and structure-level
            mistakes. If provided, must set "reduction=None". The default is
            None.
        device : int, optional
            Device (CPU or GPU) on which the model and loss computation will
            run. The default is 0.
        threshold : float, optional
            Theshold that is used to binarize a given prediction.
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
        self.threshold = threshold
        self.return_cnts = return_cnts
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred_affs, target_labels):
        """
        Computes the loss for a batch by comparing predicted affinities with
        target labels using edge-based affinities and critical component
        detection.

        Parameters
        ----------
        pred_affs : torch.Tensor
            Tensor of predicted affinities with shape "(batch_size, num_edges,
            height, width, depth)".
        target_labels : torch.Tensor
            Tensor of target labels with shape "(batch_size, height, width,
            depth)" representing the ground truth labels.

        Returns
        -------
        torch.Tensor
            Computed loss for the given batch.
        dict
            Stats related to the critical components for the batch, such as
            the number of positively and negatively critical components.

        """
        # Compute critical components
        pred_labels = self.get_pred_labels(pred_affs)
        masks, stats = self.get_critical_masks(pred_labels, target_labels)

        # Compute loss
        loss = 0
        for i in range(pred_affs.size(0)):
            mask_i = self.toGPU(masks[i, ...])
            target_labels_i = self.toGPU(target_labels[i, ...])
            for j, edge in enumerate(self.edges):
                # Compute affinities
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
        """
        Binarizes the given prediction by using "self.threshold".

        Parameters
        ----------
        pred : numpy.ndarray
            Prediction generated by neural network.

        Returns
        -------
        numpy.ndarray
            Binarized prediction.

        """
        return (pred > self.threshold).astype(np.float32)

    def get_pred_labels(self, pred_affs):
        """
        Converts predicted affinities to predicted labels by decoding the
        affinities.

        Parameters
        ----------
        pred_affs : torch.Tensor
            Tensor containing predicted affinities for a given batch.

        Returns
        -------
        List[numpy.ndarray]
            List of predicted labels for each example in the batch.

        """
        pred_affs = toCPU(pred_affs, return_numpy=True)
        pred_labels = []
        for i in range(pred_affs.shape[0]):
            pred_labels.append(self.to_labels(pred_affs[i, ...]))
        return pred_labels

    def to_labels(self, pred_affs_i):
        """
        Converts binary predicted affinities to label assignments using
        watershed segmentation.

        Parameters
        ----------
        pred_affs : numpy.ndarray
            Tensor containing predicted affinities for a given example.

        Returns
        -------
        numpy.ndarray
            Predicted segmentation.

        """
        iterator = run_watershed(self.binarize(pred_affs_i), [0])
        return next(iterator).astype(int)

    def get_critical_masks(self, preds, targets):
        """
        Computes critical masks for predicted and target labels.

        Parameters
        ----------
        preds : List[torch.Tensor]
            List of predicted segmentation tensors, where each tensor contains
            a predicted segmentation for an example.
        targets : List[torch.Tensor]
            List of groundtruth segmentation tensors, where each tensor
            contains the groundtruth segmentation for an example.

        Returns
        -------
        tuple
            A tuple containing the following:
            - torch.Tensor: critical component mask.
            - dict: Dictionary containing the following stats:
              - "Splits": Average number of negatively critical components.
              - "Merges": Average number of positively critical components.

        """
        processes = []
        stats = {"Splits": 0, "Merges": 0}
        masks = np.zeros((len(preds),) + preds[0].shape)
        targets = np.array(targets, dtype=int)
        with ProcessPoolExecutor() as executor:
            for i in range(len(preds)):
                processes.append(
                    executor.submit(
                        get_critical_mask, targets[i, 0, ...], preds[i], i, -1
                    )
                )
                processes.append(
                    executor.submit(
                        get_critical_mask, preds[i], targets[i, 0, ...], i, 1
                    )
                )
            for process in as_completed(processes):
                i, mask_i, n_criticals, crtitical_type = process.result()
                if crtitical_type == -1:
                    masks[i, ...] += self.beta * mask_i
                    stats["Splits"] += n_criticals / len(preds)
                else:
                    masks[i, ...] += (1 - self.beta) * mask_i
                    stats["Merges"] += n_criticals / len(preds)
        return self.toGPU(masks), stats

    def toGPU(self, arr):
        """
        Converts "arr" to a tensor and moves it to the GPU.

        Parameters
        ----------
        arr : numpy.array
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
        """
        Decoder module for processing edge affinities in the
        SuperVoxelAffinity loss function.

        """
        def __init__(self, edges):
            """
            Initializes Decoder object with the given edge affinities.

            Parameters
            ----------
            edges : list of tuples
                Edge affinities learned by model (e.g. [[1, 0, 0], [0, 1, 0],
                [0, 0, 1]]).

            Returns
            -------
            None

            """
            super(SuperVoxelAffinity.Decoder, self).__init__()
            self.edges = list(edges)

        def forward(self, x, i):
            """
            Extracts the predicted affinity for the i-th edge from the input
            tensor.

            Parameters
            ----------
            x : torch.Tensor
                A tensor of predicted affinities.
            i : int
                Index of the edge for which the affinity is to be extracted.
                This index corresponds to the specific edge in "self.edges".

            Returns
            -------
            torch.Tensor
                A tensor containing the affinity values for the i-th edge.

            """
            num_channels = x.size(-4)
            assert num_channels == len(self.edges)
            assert i < num_channels and i >= 0
            return get_pair_first(x[..., [i], :, :, :], self.edges[i])


# --- helpers ---
def toCPU(arr, return_numpy=False):
    """
    Moves a tensor from the GPU to the CPU and optionally converts it to a
    NumPy array.

    Parameters
    ----------
    arr : torch.Tensor
        Array to be moved to the CPU and (optionally) be converted to a numpy
        array.
    return_numpy : bool, optional
        Indication of whether to convert tensor to an array after moving it to
        CPU. The default is False.

    Returns
    -------
    torch.Tensor or numpy.ndarray
        If "return_numpy" is False, returns the PyTorch tensor moved to CPU.
        Otherwise, returns the tensor converted to a NumPy array.

    """
    if return_numpy:
        return np.array(arr.cpu().detach(), np.float32)
    else:
        return arr.detach().cpu()


def get_critical_mask(target, pred, process_id, critical_type):
    """
    Generates a critical mask and returns the associated metadata.

    Parameters
    ----------
    target : numpy.ndarray
        Ground truth segmentation.
    pred : numpy.ndarray
        Predicted segmentation output.
    process_id : int
        A unique identifier for the example in a given batch.
    critical_type : int
        An integer indicating the type of critical component based on its sign.

    Returns
    -------
    tuple
        A tuple containing:
        - "process_id" : A unique identifier for the example in a given batch.
        - "mask" : A binary mask indicating the critical components.
        - "n_criticals" : Number of detected critical components.
        - "critical_type" : Type of critical component computed.

    """
    mask, n_criticals = detect_critical(target, pred)
    return process_id, mask, n_criticals, critical_type


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
        Tensor containing the segmentation labels for a single example.
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
    ret = arr[
        ...,
        offset1[0]: shape[0] - offset2[0],
        offset1[1]: shape[1] - offset2[1],
        offset1[2]: shape[2] - offset2[2],
    ]
    return ret
