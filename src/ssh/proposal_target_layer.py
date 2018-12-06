# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from configs import cfgs
import numpy as np
import numpy.random as npr
from utils import encode_and_decode
from utils.boxes_overlap import bbox_overlaps


def proposal_target_layer(rpn_rois, gt_boxes, name):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """
    # Proposal ROIs (x1, y1, x2, y2) coming from RPN
    # gt_boxes (x1, y1, x2, y2, label)
    all_rois = rpn_rois
    if name == 'M1' or 'm1':
        minibatch_size = cfgs.M1_MINIBATCH_SIZE
        positive_rate = cfgs.M1_POSITIVE_RATE
    elif name == 'M2' or 'm2':
        minibatch_size = cfgs.M2_MINIBATCH_SIZE
        positive_rate = cfgs.M2_POSITIVE_RATE
    elif name == 'M3' or 'm3':
        minibatch_size = cfgs.M3_MINIBATCH_SIZE
        positive_rate = cfgs.M3_POSITIVE_RATE
    else:
        raise Exception('parameters error')

    # np.inf
    rois_per_image = np.inf if minibatch_size == -1 else minibatch_size

    fg_rois_per_image = np.round(positive_rate * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    labels, rois, bbox_targets, keep_inds = _sample_rois(all_rois, gt_boxes, fg_rois_per_image,
                                                         rois_per_image, cfgs.CLASS_NUM+1, name)
    # print(labels.shape, rois.shape, bbox_targets.shape)
    rois = rois.reshape(-1, 4)
    labels = labels.reshape(-1)
    bbox_targets = bbox_targets.reshape(-1, (cfgs.CLASS_NUM+1) * 4)
    keep_inds = np.array(keep_inds, np.int32)

    return rois, labels, bbox_targets, keep_inds


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]

    return bbox_targets


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image.
    that is : [label, tx, ty, tw, th]
    """

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = encode_and_decode.encode_boxes(unencode_boxes=gt_rois,
                                             reference_boxes=ex_rois,
                                             scale_factors=cfgs.ROI_SCALE_FACTORS)
    # targets = encode_and_decode.encode_boxes(ex_rois=ex_rois,
    #                                          gt_rois=gt_rois,
    #                                          scale_factor=cfgs.ROI_SCALE_FACTORS)

    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _sample_rois(all_rois, gt_boxes, fg_rois_per_image,
                 rois_per_image, num_classes, name):
    """Generate a random sample of RoIs comprising foreground and background
    examples.

    all_rois shape is [-1, 4]
    gt_boxes shape is [-1, 5]. that is [x1, y1, x2, y2, label]
    """
    # overlaps: (rois x gt_boxes)

    if name == 'M1' or 'm1':
        iou_negative_threshold_up = cfgs.M1_IOU_NEGATIVE_THRESHOLD_UP
        iou_negative_threshold_down = cfgs.M1_IOU_NEGATIVE_THRESHOLD_DOWN
        iou_positive_threshold_up = cfgs.M1_IOU_POSITIVE_THRESHOLD
    elif name == 'M2' or 'm2':
        iou_negative_threshold_up = cfgs.M2_IOU_NEGATIVE_THRESHOLD_UP
        iou_negative_threshold_down = cfgs.M2_IOU_NEGATIVE_THRESHOLD_DOWN
        iou_positive_threshold_up = cfgs.M2_IOU_POSITIVE_THRESHOLD
    elif name == 'M3' or 'm3':
        iou_negative_threshold_up = cfgs.M3_IOU_NEGATIVE_THRESHOLD_UP
        iou_negative_threshold_down = cfgs.M3_IOU_NEGATIVE_THRESHOLD_DOWN
        iou_positive_threshold_up = cfgs.M3_IOU_POSITIVE_THRESHOLD
    else:
        raise Exception('parameters error')

    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois, dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :-1], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, -1]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= iou_positive_threshold_up)[0]

    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < iou_negative_threshold_up) &
                       (max_overlaps >= iou_negative_threshold_down))[0]
    # print("first fileter, fg_size: {} || bg_size: {}".format(fg_inds.shape, bg_inds.shape))
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)

    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_this_image), replace=False)
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_this_image), replace=False)

    # print("second fileter, fg_size: {} || bg_size: {}".format(fg_inds.shape, bg_inds.shape))
    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)

    # Select sampled values from various arrays:
    labels = labels[keep_inds]

    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_this_image):] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois, gt_boxes[gt_assignment[keep_inds], :-1], labels)
    bbox_targets = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, keep_inds
