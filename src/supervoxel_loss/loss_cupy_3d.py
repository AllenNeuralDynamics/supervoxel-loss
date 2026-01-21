# Giorgio Angelotti - 2025
import torch
import torch.nn as nn
import torch.utils.dlpack
import cupy as cp
from cupyx.scipy.ndimage import binary_dilation, label
import numpy as np

# ---------------------------
# Custom Fused Kernels for Multi-Class Operations
# ---------------------------

# Fused kernel to compute false negative masks for all classes concurrently.
# For each voxel (in each class channel), if target==1 and pred==0 then mark as 1.
multi_class_false_negative_kernel = cp.ElementwiseKernel(
    'T target, T pred',
    'int8 out',
    '''
    out = ((target == (T)1) && (pred == (T)0)) ? 1 : 0;
    ''',
    'multi_class_false_negative_kernel'
)

# Fused kernel to compute false positive masks for all classes concurrently.
# For each voxel (in each class channel), if pred==1 and target==0 then mark as 1.
multi_class_false_positive_kernel = cp.ElementwiseKernel(
    'T target, T pred',
    'int8 out',
    '''
    out = ((pred == (T)1) && (target == (T)0)) ? 1 : 0;
    ''',
    'multi_class_false_positive_kernel'
)

# Fused kernel to compute the external boundary mask.
# (This kernel operates on a single class channel.)
fused_external_boundary_kernel = cp.ElementwiseKernel(
    'bool dilated, bool region_mask, int8 mistake, T volume, T root_val',
    'bool out',
    '''
    bool boundary = dilated && (!region_mask);
    out = boundary && (mistake == 0) && (volume == root_val);
    ''',
    'fused_external_boundary_kernel'
)

# Fused combine kernel: combine two masks with weights alpha and beta.
fused_combine_kernel = cp.ElementwiseKernel(
    'T mask1, T mask2, float16 beta, float16 alpha',
    'float16 out',
    '''
    out = alpha * (((float16)1 - beta) * mask1 + beta * mask2);
    ''',
    'fused_combine_kernel'
)

# ---------------------------
# Multi-Class Critical Region Detection (FN & FP)
# ---------------------------
def detect_critical_multi_class_gpu(y_target, y_pred):
    """
    Detect critical regions for all classes concurrently and compute both
    false negative (FN) and false positive (FP) critical masks.
    
    Parameters:
       y_target, y_pred : cp.ndarray with shape (num_classes, H, W, D)
         These are binary volumes for each class.
         
    Returns:
       crit_masks_fn : cp.ndarray with shape (H, W, D)
         Critical (negative) mask computed by summing over classes.
       n_crit_fn   : list of int, number of FN critical regions per class.
       crit_masks_fp : cp.ndarray with shape (H, W, D)
         Critical (positive) mask computed by summing over classes.
       n_crit_fp   : list of int, number of FP critical regions per class.
    """
    num_classes = y_target.shape[0]
    # Compute FN and FP masks concurrently for all classes.
    fn_masks = multi_class_false_negative_kernel(y_target, y_pred)
    fp_masks = multi_class_false_positive_kernel(y_target, y_pred)
    
    # Precompute the structuring elements once.
    structure = cp.ones((3, 3, 3), dtype=cp.int8)
    dilation_structure = cp.ones((3, 3, 3), dtype=cp.bool_)
    
    def process_channel(target_c, mistakes):
        """Process one class channel and return critical mask and count."""
        # For FN: use target volume; for FP: use prediction volume.
        vol_minus_mistakes, _ = label(target_c * (1 - mistakes), structure=structure)
        mistake_labels, _ = label(mistakes, structure=structure)
        crit_mask = cp.zeros(target_c.shape, dtype=cp.bool_)
        n_regions = 0
        
        # Get unique labels directly as a CuPy array.
        unique_ids = cp.unique(mistake_labels)
        for rid in unique_ids:
            # rid is a 0-dim CuPy array; use item() to compare with 0.
            if rid.item() == 0:
                continue
            region_mask = (mistake_labels == rid)
            indices = cp.argwhere(region_mask)
            if indices.shape[0] == 0:
                continue
            # Use the first index in the region as the "root"
            root_idx = tuple(int(x.item()) for x in indices[0])
            root_val = target_c[root_idx]
            # Perform dilation using the precomputed dilation structure.
            dilated = binary_dilation(region_mask, structure=dilation_structure)
            external_boundary = fused_external_boundary_kernel(
                dilated, region_mask, mistakes, target_c, root_val
            )
            if cp.any(external_boundary):
                unique_vals = cp.unique(vol_minus_mistakes[external_boundary])
                is_critical = (unique_vals.size != 1)
            else:
                is_critical = True
            if is_critical:
                # In-place update using logical OR.
                cp.logical_or(crit_mask, region_mask, out=crit_mask)
                n_regions += 1
        return crit_mask, n_regions

    # Process each non-background class (assume class 0 is background) using list comprehensions.
    crit_masks_fn_list = []
    n_crit_fn = []
    crit_masks_fp_list = []
    n_crit_fp = []
    
    # Loop over classes starting at 1.
    for c in range(1, num_classes):
        # For false negatives.
        target_c = y_target[c]
        mistakes_fn = fn_masks[c]
        crit_mask_fn, n_fn = process_channel(target_c, mistakes_fn)
        crit_masks_fn_list.append(crit_mask_fn)
        n_crit_fn.append(n_fn)
        
        # For false positives, use prediction volume.
        pred_c = y_pred[c]
        mistakes_fp = fp_masks[c]
        crit_mask_fp, n_fp = process_channel(pred_c, mistakes_fp)
        crit_masks_fp_list.append(crit_mask_fp)
        n_crit_fp.append(n_fp)
    
    # Combine the per-class masks by stacking and summing along the class dimension.
    crit_masks_fn = cp.stack(crit_masks_fn_list, axis=0).sum(axis=0)
    crit_masks_fp = cp.stack(crit_masks_fp_list, axis=0).sum(axis=0)
    return crit_masks_fn, n_crit_fn, crit_masks_fp, n_crit_fp


# ---------------------------
# Loss
# ---------------------------
class SuperVoxelLoss(nn.Module):
    """
    SuperVoxelLoss that incorporates structure-aware penalties computed
    from both false negatives and false positives concurrently.
    """
    def __init__(self, alpha=0.5, beta=0.5, threshold=0.0, device="cuda",
                 criterion=nn.CrossEntropyLoss, num_classes=2):
        """
        Parameters:
           alpha (float): Weight for structure-aware (critical) loss.
           beta (float): Balances contributions of the two critical masks.
           device (str): Device on which tensors are located.
           num_classes (int): Number of classes.
        """
        super(SuperVoxelLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.num_classes = num_classes
        self.threshold = threshold
        self.criterion = criterion(reduction="none")
        # We'll use our multi-class detection that computes both FN and FP critical masks.
        self.detect_critical = detect_critical_multi_class_gpu

    def forward(self, preds, targets):
        """
        Compute the loss with structure-aware penalties.
        
        preds: torch.Tensor with shape (B, C, H, W, D) (or label form).
        targets: torch.Tensor with shape (B, 1, H, W, D)
        """
        # Convert logits to labels if needed.
        if preds.shape[1] > 1:
            preds_squeezed = preds.argmax(dim=1)
        else:
            preds_squeezed = preds[:, 0, ...]
        
        batch_size = preds_squeezed.shape[0]
        combined_masks = []
        for i in range(batch_size):
            # Expand each label map into binary maps per class: shape (num_classes, H, W, D).
            target_bin = torch.stack([ (targets[i] == c).float() for c in range(self.num_classes) ], dim=0)
            if self.threshold > 0:
                pred_bin = torch.stack([ (preds_squeezed[i] > self.threshold).float() for c in range(self.num_classes) ], dim=0)
            else:
                pred_bin = torch.stack([ (preds_squeezed[i] == c).float() for c in range(self.num_classes) ], dim=0)
            
            # Convert to CuPy arrays (using half precision).
            target_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(target_bin.half()))
            pred_cp = cp.fromDlpack(torch.utils.dlpack.to_dlpack(pred_bin.half()))
            
            # Detect FN and FP critical masks for all classes.
            crit_fn, _, crit_fp, _ = self.detect_critical(target_cp, pred_cp)
            # We combine the two masks by summing them.
            combined = crit_fn + crit_fp
            combined_masks.append(combined)
        
        combined_masks = cp.stack(combined_masks, axis=0)

        dlpack_capsule = combined_masks.toDlpack() # if doesn't work with dlpack, convert combined masks to a numpy array (cpu) and then bring it back to gpu as a tensor!
        combined_masks_torch = torch.utils.dlpack.from_dlpack(dlpack_capsule).to(self.device)

        
        loss = (1 - self.alpha + combined_masks_torch) * self.criterion(preds, targets)

        #print(loss.mean())
        return loss.mean()

# ---------------------------
# Test Visualization with Multiple Tubular Structures, Mergers, and Cuts
# ---------------------------
if __name__ == '__main__':
    import napari

    def prepare_two_channel_input(binary_mask):
        """
        Convert a single-channel binary map (H, W, D) into a two-channel volume.
        Channel 0 is the background (1 - binary_mask) and channel 1 is the foreground (binary_mask).
        """
        background = 1 - binary_mask
        return cp.stack([background, binary_mask], axis=0)

    def visualize_critical_detection(y_target_bin, y_pred_bin, class_id):
        """
        Visualize the critical detection for a single class.
        
        y_target_bin, y_pred_bin: cp.ndarray binary maps for a specific class (shape: H, W, D).
        class_id: int, the class being visualized.
        """
        # Convert the single-channel input to a two-channel representation.
        target_cp = prepare_two_channel_input(y_target_bin)
        pred_cp = prepare_two_channel_input(y_pred_bin)
        
        # Call our multi-class detection function.
        crit_fn, n_fn, crit_fp, n_fp = detect_critical_multi_class_gpu(target_cp, pred_cp)
        
        # For a two-channel input, the detection was performed only on channel 1 (foreground).
        print(f"Class {class_id}: {n_fn[0]} negative and {n_fp[0]} positive critical regions detected.")
        
        # Combine for visualization.
        crit_mask = crit_fn + crit_fp
        target_np = cp.asnumpy(y_target_bin)
        pred_np = cp.asnumpy(y_pred_bin)
        crit_np = cp.asnumpy(crit_mask.astype(cp.float16))
        
        viewer = napari.Viewer()
        viewer.add_image(target_np, name=f'Target (Class {class_id})')
        viewer.add_image(pred_np, name=f'Prediction (Class {class_id})', colormap='viridis', opacity=0.5)
        viewer.add_image(crit_np, name=f'Critical Regions (Class {class_id})', colormap='magenta', opacity=0.5)
        napari.run()

    # ---------------------------
    # Synthetic Data Generation
    # ---------------------------
    shape = (64, 64, 64)
    Z, Y, X = np.indices(shape)
    tube_radius = 3

    # Create several tubular structures.
    # Class 1: Two tubes.
    tube1_class1 = (np.sqrt((X - 16)**2 + (Y - 16)**2) < tube_radius)
    tube2_class1 = (np.sqrt((X - 16)**2 + (Y - 48)**2) < tube_radius)
    # Class 2: Two tubes.
    tube1_class2 = (np.sqrt((X - 48)**2 + (Y - 16)**2) < tube_radius)
    tube2_class2 = (np.sqrt((X - 48)**2 + (Y - 48)**2) < tube_radius)
    # Class 3: One tube.
    tube_class3  = (np.sqrt((X - 32)**2 + (Y - 32)**2) < tube_radius)
    
    # Create target volume (background = 0).
    target_np = np.zeros(shape, dtype=np.float32)
    target_np[tube1_class1] = 1
    target_np[tube2_class1] = 1
    target_np[tube1_class2] = 2
    target_np[tube2_class2] = 2
    target_np[tube_class3]  = 3

    # Create prediction volume as a copy and then introduce errors.
    pred_np = target_np.copy()
    
    # --- Class 1 errors ---
    # Introduce a hole error.
    z_hole = slice(30, 34)
    y_hole = slice(12, 18)
    x_hole = slice(12, 18)
    pred_np[z_hole, y_hole, x_hole] = 0

    # Introduce a merger error: create an artificial bridge connecting the two Class 1 tubes.
    z_merge_class1 = slice(40, 44)
    # The merger bridge covers a region between the two tubes.
    merger_bridge_mask_class1 = ((Y >= 16) & (Y <= 48) & (X >= 10) & (X <= 22))
    # Index the merger mask over the same z-slices as the prediction.
    pred_np[z_merge_class1, :, :] = np.where(merger_bridge_mask_class1[z_merge_class1, :, :], 1, pred_np[z_merge_class1, :, :])
    
    # --- Class 2 errors ---
    # Introduce a merger error between tubes.
    y2d, x2d = np.indices((shape[1], shape[2]))
    merger_mask_class2 = ((x2d >= 38) & (x2d <= 54) & (y2d >= 10) & (y2d <= 30))
    z_merge_class2 = slice(30, 34)
    pred_np[z_merge_class2, :, :] = np.where(merger_mask_class2[None, :, :], 2, pred_np[z_merge_class2, :, :])
    
    # --- Class 3 errors ---
    # Introduce a connectivity cut error.
    z_cut_class3 = slice(45, 48)
    cut_mask_class3 = tube_class3 & ((Y >= 28) & (Y <= 36))
    pred_np[z_cut_class3, :, :] = np.where(cut_mask_class3[z_cut_class3, :, :], 0, pred_np[z_cut_class3, :, :])
    
    # Introduce an additional cut error in a different region.
    z_cut_class3_2 = slice(10, 14)
    cut_mask_class3_2 = tube_class3 & ((X >= 25) & (X <= 35))
    pred_np[z_cut_class3_2, :, :] = np.where(cut_mask_class3_2[z_cut_class3_2, :, :], 0, pred_np[z_cut_class3_2, :, :])
    
    # Convert to CuPy arrays.
    target_cp = cp.array(target_np)
    pred_cp = cp.array(pred_np)
    
    # For demonstration, visualize the critical detection for each class.
    # Visualize Class 1.
    visualize_critical_detection(target_cp == 1, pred_cp == 1, class_id=1)
    # Visualize Class 2.
    visualize_critical_detection(target_cp == 2, pred_cp == 2, class_id=2)
    # Visualize Class 3.
    visualize_critical_detection(target_cp == 3, pred_cp == 3, class_id=3)