from __future__ import division

import torch
import numpy as np
from utils.utils import bbox_iou, xywh2xyxy

def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Args:
        prediction.shape(batch_size, num_yolo*num_anchors*grid_size*grid_size, 85)
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # From center(xywh) to corner(xyxy)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):

        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # score = object_conf. * max_class_pred_prob.
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]

        # Sort by it
        image_pred = image_pred[np.argsort(-score)]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)

        # detections.shape(unknown, 7_vals)
        # 7_vals=(x1, y1, x2, y2, object_conf., class_score, class_pred_label)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)

        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):

            #=== Indices of boxes with large IOUs and matching labels ===
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            invalid = large_overlap & label_match

            #=== Merge overlapping bboxes weighted by their confidence ===
            weights = detections[invalid, 4:5]
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()

            keep_boxes += [detections[0]]

            #=== remove the suppression ===
            detections = detections[~invalid]

        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output
