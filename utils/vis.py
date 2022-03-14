import numpy as np
import cv2
import torch


def vis_targets(images, targets, anchor_boxes):
    """
        images: (tensor) [B, 3, H, W]
        targets: (tensor) [B, N, C+4+1]
        anchor_boxes: (tensor) [N, 4]
    """
    batch_size = images.size(0)
    # vis data
    rgb_mean=np.array((0.485, 0.456, 0.406), dtype=np.float32)
    rgb_std=np.array((0.229, 0.224, 0.225), dtype=np.float32)

    for bi in range(batch_size):
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        # denormalize
        image = ((image * rgb_std + rgb_mean)*255).astype(np.uint8)
        # to BGR
        image = image[..., (2, 1, 0)]
        image1 = image.copy()
        image2 = image.copy()

        target_i = targets[bi] # [N, C+4+1]
        tgt_pos = (target_i[..., -1] > 0.).float()        # [N,]
        tgt_ign = (target_i[..., -1] == -1.).float()      # [N,]
        pos_mask = torch.nonzero(tgt_pos).view(-1).tolist()
        ign_mask = torch.nonzero(tgt_ign).view(-1).tolist()
        tgt_boxes = target_i[..., -5:-1][pos_mask]        # [M, 4]
        pos_anchors = anchor_boxes[pos_mask]              # [M, 4]
        ign_anchors = anchor_boxes[ign_mask]              # [I, 4]

        # draw positive samples
        for i in range(len(pos_mask)):
            # gt box
            x1, y1, x2, y2 = tgt_boxes[i]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # anchor box
            xc, yc, bw, bh = pos_anchors[i]
            x1 = int(xc - 0.5 * bw)
            y1 = int(yc - 0.5 * bh)
            x2 = int(xc + 0.5 * bw)
            y2 = int(yc + 0.5 * bh)
            cv2.rectangle(image1, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # draw ignore samples
        for i in range(len(ign_mask)):
            # anchor box
            xc, yc, bw, bh = ign_anchors[i]
            x1 = int(xc - 0.5 * bw)
            y1 = int(yc - 0.5 * bh)
            x2 = int(xc + 0.5 * bw)
            y2 = int(yc + 0.5 * bh)
            cv2.rectangle(image2, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('groundtruth', np.concatenate([image1, image2], axis=1))
        cv2.waitKey(0)