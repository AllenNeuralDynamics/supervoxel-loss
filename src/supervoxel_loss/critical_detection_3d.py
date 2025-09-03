"""
Created on Sun November 17 22:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Computes positively and negatively critical components for 3D images.

"""

from random import sample
from scipy.ndimage import label

import numpy as np


def detect_critical_3d(y_target, y_pred):
    """
    Detcts negatively critical components by performing BFS on the
    foreground of "y_mistakes". A root voxel is sampled from the foreground,
    and BFS is used to extract the connected component. Criticality conditions
    are checked once the BFS reaches the boundary of the component.

    Parameters
    ----------
    y_target : numpy.ndarray
        Groundtruth segmentation where each segment has a unique label.
    y_pred : numpy.ndarray
        Predicted segmentation where each segment has a unique label.

    Returns
    -------
    numpy.ndarray
        Binary mask where critical components are marked with a "1".
    """
    # Compute mistakes
    y_mistakes = false_negative_mask(y_target, y_pred)
    y_target_minus_mistakes, _ = label(y_target * (1 - y_mistakes))

    # Detect critical mistakes
    n_criticals = 0
    critical_mask = np.zeros(y_target.shape, dtype=bool)
    foreground = get_foreground(y_mistakes)
    while len(foreground) > 0:
        xyz_r = sample(foreground, 1)[0]
        component_mask, visited, is_critical = extract_component(
            y_target, y_mistakes, y_target_minus_mistakes, xyz_r
        )
        foreground = foreground.difference(visited)
        if is_critical:
            critical_mask += component_mask
            n_criticals += 1
    return critical_mask, n_criticals


def extract_component(y_target, y_mistakes, y_minus_mistakes, xyz_r):
    """
    Extracts the connected component corresponding to the given root by
    performing a BFS.

    Parameters
    ----------
    y_target : numpy.ndarray
        Groundtruth segmentation where each segment has a unique label.
    y_mistakes : numpy.ndarray
        Binary mask where incorect voxel predictions are marked with a "1".
    y_minus_mistakes : numpy.ndarray
        Connected components of the groundtruth segmentation "minus" the
        mistakes mask.
    xyz_r : Tuple[int]
        Voxel coordinate of root.

    Returns
    -------
    numpy.ndarray
        Binary mask where connected component is marked with a "1".
    Set[Tuple[int]]
        Voxels visited during the BFS
    bool
        Indication of whether the connected component is critical.
    """
    mask = np.zeros(y_target.shape, dtype=bool)
    collisions = dict()
    is_critical = False
    queue = [tuple(xyz_r)]
    visited = set()
    while len(queue) > 0:
        xyz_i = queue.pop(0)
        mask[xyz_i] = 1
        for xyz_j in get_nbs(xyz_i, y_target.shape):
            if xyz_j not in visited and y_target[xyz_r] == y_target[xyz_j]:
                visited.add(xyz_j)
                if y_mistakes[xyz_j] == 1:
                    queue.append(xyz_j)
                elif not is_critical:
                    key = y_target[xyz_j]
                    if key not in collisions.keys():
                        collisions[key] = y_minus_mistakes[xyz_j]
                    elif collisions[key] != y_minus_mistakes[xyz_j]:
                        is_critical = True
    if y_target[xyz_r] not in collisions.keys():
        is_critical = True
    return mask, visited, is_critical


def false_negative_mask(y_target, y_pred):
    """
    Computes false negative mask.

    Parameters
    ----------
    y_target : numpy.ndarray
        Groundtruth segmentation where each segment has a unique label.
    y_pred : numpy.ndarray
        Predicted segmentation where each segment has a unique label.

    Returns
    -------
    numpy.ndarray
        Binary mask where false negative mistakes are marked with a "1".
    """
    false_negatives = y_target.astype(bool) * (1 - y_pred.astype(bool))
    return false_negatives.astype(int)


def get_foreground(img):
    """
    Gets the set of foreground voxels from the given image.

    Parameters
    ----------
    img : numpy.ndarray
        A 3D image to be searched.

    Returns
    -------
    Set[Tuple[int]]
        Set of tuples such that each is the coordinate of a voxel in the
        foreground.
    """
    x, y, z = np.nonzero(img)
    return set((x[i], y[i], z[i]) for i in range(len(x)))


def get_nbs(xyz, shape):
    """
    Gets all neighbors of a voxel in a 3D image using 26-connectivity.

    Parameters
    ----------
    xyz : Tuple[int]
        Coordinates of the voxel for which neighbors are to be found.
    shape : Tuple[int]
        Shape of the image that voxel belongs to.

    Returns
    -------
    numpy.ndarray
        Voxel coordinates of neighbors of the given voxel.
    """
    x_offsets, y_offsets, z_offsets = np.meshgrid(
        [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"
    )
    nbs = np.column_stack(
        [
            (xyz[0] + y_offsets).ravel(),
            (xyz[1] + x_offsets).ravel(),
            (xyz[2] + z_offsets).ravel(),
        ]
    )
    mask = np.all((nbs >= 0) & (nbs < np.array(shape)), axis=1)
    return map(tuple, list(nbs[mask]))
