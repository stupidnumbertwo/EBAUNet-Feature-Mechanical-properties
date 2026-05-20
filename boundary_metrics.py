import numpy as np
from skimage import segmentation
from scipy.ndimage import distance_transform_edt

def boundary_iou(pred_mask, gt_mask, boundary_width=3):
    """
    Compute Boundary IoU between prediction and ground truth masks.
    pred_mask, gt_mask: binary numpy arrays [H, W]
    """
    pred_boundary = segmentation.find_boundaries(pred_mask, mode='thick')
    gt_boundary = segmentation.find_boundaries(gt_mask, mode='thick')

    # Dilate boundaries
    pred_dilate = distance_transform_edt(~pred_boundary) <= boundary_width
    gt_dilate = distance_transform_edt(~gt_boundary) <= boundary_width

    intersection = np.logical_and(pred_boundary, gt_dilate).sum() + \
                   np.logical_and(gt_boundary, pred_dilate).sum()
    union = pred_boundary.sum() + gt_boundary.sum()

    return intersection / union if union > 0 else np.nan

def boundary_f1(pred_mask, gt_mask, tolerance=2):
    """
    Compute Boundary F1 score between prediction and ground truth masks.
    pred_mask, gt_mask: binary numpy arrays [H, W]
    """
    pred_boundary = segmentation.find_boundaries(pred_mask, mode='thick')
    gt_boundary = segmentation.find_boundaries(gt_mask, mode='thick')

    # Distance transforms
    pred_dt = distance_transform_edt(~pred_boundary)
    gt_dt = distance_transform_edt(~gt_boundary)

    pred_match = (gt_dt <= tolerance) & pred_boundary
    gt_match = (pred_dt <= tolerance) & gt_boundary

    precision = pred_match.sum() / (pred_boundary.sum() + 1e-8)
    recall = gt_match.sum() / (gt_boundary.sum() + 1e-8)

    if (precision + recall) == 0:
        return np.nan

    return 2 * precision * recall / (precision + recall)
