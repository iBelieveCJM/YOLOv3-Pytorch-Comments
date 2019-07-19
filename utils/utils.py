from __future__ import division

import torch
import numpy as np

def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape

    current_dim : the demension of the padding image
    boxes       : relative to the padding image
    """
    orig_h, orig_w = original_shape

    # The amount of padding that was added (relative to padding image)
    pad_x = (max(orig_h - orig_w, 0) / max(original_shape)) * current_dim 
    pad_y = (max(orig_w - orig_h, 0) / max(original_shape)) * current_dim

    # Image height and width after padding is removed (relative to padding image)
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x

    # scale bounding boxes (relative to the original image)
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    """convert from center(xywh) to corner(xyxy)
    top-left corner_x = center_x - w/2
    top-left corner_y = center_y - h/2
    bottom-right corner_x = center_x + w/2
    bottom-right corner_y = center_y + h/2
    """
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center(xywh) to corner(xyxy)
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the corner coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """
    pred_boxes.shape(batch_size, num_anchors, grid_size, grid_size, 4)
    pred_cls.shape(batch_size, num_anchors, grid_size, grid_size, num_classes)
    """
    ByteTensor  = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0) # batch_size (num_samples)
    nA = pred_boxes.size(1) # num_anchors
    nC = pred_cls.size(-1)  # num_classes
    nG = pred_boxes.size(2) # grid_size

    # Output tensors
    # shape(batch_size, num_anchors, grid_size, grid_size)
    obj_mask   = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)  # fill with 1
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    # shape(batch_size, num_anchors, grid_size, grid_size, num_classes)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    ##=== scale the target bboxes (relative to feature map) ===
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    ##=== Get anchors with best iou ===
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)

    ##=== Compute target values from target bbox ===
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()

    # get the top-left corner coordinates of the grid cell
    # where the object(target bbox center) appears
    gi, gj = gxy.long().t()  

    # Set masks
    obj_mask[b, best_n, gj, gi]   = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Center offset
    # (gx.floor(), gy.floor()) is the top-left corner of the grid cell
    # where the object(target bbox center) appears
    # b_x = sigmod(t_x) + c_x   ==> target_sigmod(t_x) = b_x - c_x
    # b_y = sigmod(t_y) + c_y   ==> target_sigmod(t_y) = b_y - c_y
    tx[b, best_n, gj, gi] = gx - gx.floor() 
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    # b_w = anchor_w * exp(t_w) ==> target_(t_w) = log(b_w / anchor_w)
    # b_h = anchor_h * exp(t_h) ==> target_(t_h) = log(b_h / anchor_h)
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    ##=== One-hot encoding of label ===
    tcls[b, best_n, gj, gi, target_labels] = 1

    ##=== Compute label correctness and iou at best anchor ===
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
