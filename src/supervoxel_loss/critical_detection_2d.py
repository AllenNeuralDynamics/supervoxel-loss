"""
Created on Sun November 17 22:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Computes positively and negatively critical components in 3d space.

"""

import numpy as np
from random import sample
from scipy.ndimage import label as label_cpu


def detect_critical(y_target, y_pred):
    """
    Detects negatively critical components.

    Parameters
    ----------
    y_target : numpy.ndarray
        Groundtruth segmentation where each segment as a unique label.
    y_pred : numpy.ndarray
        Predicted segmentation where each segment as a unique label.

    Returns
    -------
    np.ndarray
        Binary mask where critical components are marked with a "1".

    """
    y_mistakes = get_false_negatives(y_target, y_pred)
    y_target_minus_ccps, _ = get_ccps(y_target * (1 - y_mistakes))
    return run_detection(
        y_target, y_mistakes, y_target_minus_ccps
    )


def run_detection(y_target, y_mistakes, y_minus_ccps):
    """
    Computes negatively critical components by running a BFS on the
    foreground. Here, a root is sampled and then passed into a subroutine
    that performs the BFS to extract a single connected component.

    Parameters
    ----------
    y_target : numpy.ndarray
        Groundtruth segmentation where each segment as a unique label.
    y_mistakes : numpy.ndarray
        Binary mask where incorect voxel predictions are marked with a "1".
    y_minus_ccps : numpy.ndarray
        Connected components of the groundtruth segmentation "minus" the
        mistakes mask.

    Returns
    -------
    numpy.ndarray
        Binary mask where critical components are marked with a "1".

    """
    num_criticals = 0
    critical_mask = np.zeros(y_target.shape)
    foreground = get_foreground(y_mistakes)
    while len(foreground) > 0:
        xyz_r = sample(foreground, 1)[0]
        component_mask, visited, is_critical = get_component(
            y_target, y_mistakes, y_minus_ccps, xyz_r
        )
        foreground = foreground.difference(visited)
        if is_critical:
            critical_mask += component_mask
            num_criticals += 1
    return critical_mask, num_criticals


def get_component(
    y_target, y_mistakes, y_minus_ccps, xyz_r
):
    """
    Performs a BFS to extract a single connected component from a given root.

    Parameters
    ----------
    y_target : numpy.ndarray
        Groundtruth segmentation where each segment as a unique label.
    y_mistakes : numpy.ndarray
        Binary mask where incorect voxel predictions are marked with a "1".
    y_minus_ccps : numpy.ndarray
        Connected components of the groundtruth segmentation "minus" the
        mistakes mask.
    xyz_r : tuple[int]
        Indices of root node. 

    Returns
    -------
    mask : numpy.ndarray
        Binary mask where critical component is marked with a "1".
    visited : set[tuple[int]]
        Set of indices visited during the BFS
    is_critical : bool
        Indication of whether connected component is critical.

    """
    mask = np.zeros(y_target.shape)
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
                    if y_target[xyz_j] not in collisions.keys():
                        collisions[y_target[xyz_j]] = y_minus_ccps[xyz_j]
                    elif collisions[y_target[xyz_j]] != y_minus_ccps[xyz_j]:
                        is_critical = True
    if y_target[xyz_r] not in collisions.keys():
        is_critical = True
    return mask, visited, is_critical


def get_false_negatives(y_target, y_pred):
    """
    Computes false negative mask.

    Parameters
    ----------
    y_target : numpy.ndarray
        Groundtruth segmentation where each segment as a unique label.
    y_pred : numpy.ndarray
        Predicted segmentation where each segment as a unique label.

    Returns
    -------
    false_negatives : numpy.ndarray
        Binary mask where false negative mistake is marked with a "1".

    """
    false_negatives = y_target.astype(bool) * (1 - y_pred.astype(bool))
    return false_negatives.astype(int)


def get_ccps(arr):
    """
    Computes connected components using routine from SciPy library.

    Parameters
    ----------
    arr : numpy.ndarray
        Array.

    Returns
    -------
    numpy.ndarray
        Connected components labeling.

    """
    return label_cpu(arr, structure=get_kernel())


def get_kernel():
    """
    Gets the connectivity kernel used in the connected components computation.

    Parmaters
    ---------
    None

    Returns
    -------
    numpy.ndarray
        Connectivity kernel used in the connected components computation.

    """
    return np.ones((3, 3))


def get_foreground(img):
    """
    Computes the foreground of an image.

    Parameters
    ----------
    
    """
    x, y = np.nonzero(img)
    return set((x[i], y[i]) for i in range(len(x)))


def get_nbs(xyz, shape):
    """
    Get all neighbors of a voxel in a 3D image with respect to 26-connectivity.

    Parameters
    ----------
    xyz : tuple[int]
        Coordinates of the voxel.
    image_shape : tuple[int]
        Tuple representing the shape of the 3D image.

    Returns
    -------
    np.ndarray
        Numpy array of shape (k, 2) with k <= 8 representing the coordinates of all 8 neighbors.

    """
    x_offsets, y_offsets = np.meshgrid(
        [-1, 0, 1], [-1, 0, 1], indexing="ij"
    )
    nbs = np.column_stack(
        [
            (xyz[0] + y_offsets).ravel(),
            (xyz[1] + x_offsets).ravel(),
        ]
    )
    mask = np.all((nbs >= 0) & (nbs < np.array(shape)), axis=1)
    return map(tuple, list(nbs[mask]))
