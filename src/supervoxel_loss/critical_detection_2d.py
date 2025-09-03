"""
Created on Sun November 17 22:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Computes positively and negatively critical components for 2D images.

"""

from random import sample
from scipy.ndimage import label

import numpy as np


def detect_critical_2d(y_target, y_pred):
    """
    Detects negatively critical components for 2D images.

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
    y_mistakes = get_false_negative_mask(y_target, y_pred)
    y_target_minus_mistakes, _ = label(y_target * (1 - y_mistakes))

    # Detect critical mistakes
    critical_mask = np.zeros(y_target.shape)
    foreground = get_foreground(y_mistakes)
    while len(foreground) > 0:
        xy_r = sample(foreground, 1)[0]
        component_mask, visited, is_critical = extract_component(
            y_target, y_mistakes, y_target_minus_mistakes, xy_r
        )
        foreground = foreground.difference(visited)
        if is_critical:
            critical_mask += component_mask
    return critical_mask


def extract_component(y_target, y_mistakes, y_minus_mistakes, xy_r):
    """
    Extracts the connected component corresponding to the given root by
    performing a BFS.

    Parameters
    ----------
    y_target : numpy.ndarray
        Groundtruth segmentation where each segment has a unique label.
    y_mistakes : numpy.ndarray
        Binary mask where incorect pixel predictions are marked with a "1".
    y_minus_mistakes : numpy.ndarray
        Connected components of the groundtruth segmentation "minus" the
        mistakes mask.
    xy_r : tuple[int]
        Pixel coordinate of root.

    Returns
    -------
    numpy.ndarray
        Binary mask where connected component is marked with a "1".
    Set[Tuple[int]]
        Pixels visited during the BFS
    bool
        Indication of whether the connected component is critical.
    """
    mask = np.zeros(y_target.shape, dtype=bool)
    collisions = dict()
    is_critical = False
    queue = [tuple(xy_r)]
    visited = set()
    while len(queue) > 0:
        xy_i = queue.pop(0)
        mask[xy_i] = 1
        for xy_j in get_nbs(xy_i, y_target.shape):
            if xy_j not in visited and y_target[xy_r] == y_target[xy_j]:
                visited.add(xy_j)
                if y_mistakes[xy_j] == 1:
                    queue.append(xy_j)
                elif not is_critical:
                    key = y_target[xy_j]
                    if key not in collisions.keys():
                        collisions[key] = y_minus_mistakes[xy_j]
                    elif collisions[key] != y_minus_mistakes[xy_j]:
                        is_critical = True
    if y_target[xy_r] not in collisions.keys():
        is_critical = True
    return mask, visited, is_critical


def get_false_negative_mask(y_target, y_pred):
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
    fn_mask = y_target.astype(bool) * (1 - y_pred.astype(bool))
    return fn_mask.astype(int)


def get_foreground(img):
    """
    Gets the set of foreground pixels from the given image.

    Parameters
    ----------
    img : numpy.ndarray
        A 2D image to be searched.

    Returns
    -------
    Set[Tuple[int]]
        Set of tuples such that each is the coordinate of a pixel in the
        foreground.
    """
    x, y = np.nonzero(img)
    return set((x[i], y[i]) for i in range(len(x)))


def get_nbs(xy, shape):
    """
    Gets all neighbors of a pixel in a 2D image using 8-connectivity.

    Parameters
    ----------
    xy : Tuple[int]
        Coordinates of the pixel for which neighbors are to be found.
    shape : Tuple[int]
        Shape of the image that pixel belongs to.

    Returns
    -------
    Iterator[Tuple[int]]
        Pixel coordinates of neighbors of the given pixel.
    """
    x_offsets, y_offsets = np.meshgrid([-1, 0, 1], [-1, 0, 1], indexing="ij")
    nbs = np.column_stack(
        [(xy[0] + y_offsets).ravel(), (xy[1] + x_offsets).ravel()]
    )
    mask = np.all((nbs >= 0) & (nbs < np.array(shape)), axis=1)
    return map(tuple, list(nbs[mask]))
