import torch
import numpy as np


def compute_iou_dist(anchor_boxes, target_box):
    """
    Input:
        anchor_boxes : [N, 4]
        gt_box :       [1, 4]
    Output:
        iou : [N,]
        dist: [N,]
    """
    # anchor box: [HW x KA, 4]
    # convert  [xc, yc, w, h] -> [x1, y1, x2, y2]
    anchor_boxes_ = anchor_boxes.copy()
    anchor_boxes_[..., :2] = anchor_boxes[..., :2] - anchor_boxes[..., 2:] * 0.5 # x1y1
    anchor_boxes_[..., 2:] = anchor_boxes[..., :2] + anchor_boxes[..., 2:] * 0.5 # x2y2
    anchor_width = anchor_boxes_[:, 2] - anchor_boxes_[:, 0]
    anchor_height = anchor_boxes_[:, 3] - anchor_boxes_[:, 1]
    anchor_area = anchor_height * anchor_width
    
    # gt_box: [1, 4] -> [N, 4]
    target_box = np.repeat(target_box, anchor_boxes_.shape[0], axis=0)
    target_width = target_box[:, 2] - target_box[:, 0]
    target_height = target_box[:, 3] - target_box[:, 1]
    target_area = target_height * target_width

    # Area of intersection
    intersecion_width = np.minimum(anchor_boxes_[:, 2], target_box[:, 2]) - \
                            np.maximum(anchor_boxes_[:, 0], target_box[:, 0])
    intersection_height = np.minimum(anchor_boxes_[:, 3], target_box[:, 3]) - \
                            np.maximum(anchor_boxes_[:, 1], target_box[:, 1])
    intersection_area = intersection_height.clip(0.) * intersecion_width.clip(0.)
    # Area of union
    union_area = anchor_area + target_area - intersection_area + 1e-20
    # IoU
    iou = intersection_area / union_area

    # distance
    dist = np.abs(anchor_boxes_ - target_box).sum(-1)

    return iou, dist


def static_label_assignment_with_anchorbox(
                            img_size=640,
                            targets=None, 
                            anchor_boxes=None,
                            num_classes=80,
                            topk=4,
                            iou_t=0.15,
                            igt=0.7):
    """
        img_size: (Int) the size of input image
        targets: (Dict) {'boxes': array([[x1, y1, x2, y2], ...]), 
                         'labels':array([label1, label2, ...])}
        anchor_boxes: (Tensor) [N, 4]
        stride: (Int) the output stride of network
        num_classes: (Int) the number of categories
    """
    # prepare
    batch_size = len(targets)
    N = anchor_boxes.shape[0]

    # [B, N, cls+box+pos]
    target_tensor = np.zeros([batch_size, N, num_classes + 4 + 1])
    # [N, 4]
    anchor_boxes = anchor_boxes.cpu().numpy().copy()

    # generate gt datas  
    for bi in range(batch_size):
        target_i = targets[bi]
        boxes_i = target_i["boxes"].numpy()
        labels_i = target_i["labels"].numpy()

        for box, label in zip(boxes_i, labels_i):
            cls_id = int(label)
            x1, y1, x2, y2 = box
            box_w, box_h = x2 - x1, y2 - y1
            # check box
            if box_w < 1. or box_h < 1.:
                continue

            gt_box = np.array([[x1, y1, x2, y2]])

            # compute IoU
            iou, dist = compute_iou_dist(anchor_boxes, gt_box)

            # keep the topk anchor boxes
            dist_sorted_idx = np.argsort(dist)

            # make labels
            positive_sample_indx = []
            for k in range(topk):
                grid_idx = dist_sorted_idx[k]
                iou_score = iou[grid_idx]
                if iou_score > iou_t:
                    target_tensor[bi, grid_idx, :num_classes] = 0.0 # avoiding the multi labels for one grid cell
                    target_tensor[bi, grid_idx, cls_id] = 1.0
                    target_tensor[bi, grid_idx, num_classes:num_classes+4] = np.array([x1, y1, x2, y2])
                    target_tensor[bi, grid_idx, -1] = 1.0 # 2.0 - (box_w / img_size) * (box_h / img_size)
                    positive_sample_indx.append(grid_idx)
            
            # ignore samples
            if igt is not None:
                iou_p_indx = np.where(iou > igt)[0].tolist()
                if len(iou_p_indx) > 0:
                    for k in iou_p_indx:
                        if k in positive_sample_indx:
                            continue
                        else:
                            target_tensor[bi, k, -1] = -1.0 # ignore sample
    
    return torch.from_numpy(target_tensor).float()


def dynamic_label_assignment():
    # TODO
    return
