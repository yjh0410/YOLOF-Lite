import numpy as np
import torch
from torch import nn
from .box_ops import box_xyxy_to_cxcywh


class UniformMatcher(nn.Module):
    """
    Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors.

    Args:
        match_times(int): Number of positive anchors for each gt box.
        选择topk个与gt的l1_distance最小的预测框, 再选择topk个与gt的l1_distance最小的anchor, 将其作为正例框
    """

    def __init__(self, match_times: int = 4):
        super().__init__()
        self.match_times = match_times

    @torch.no_grad()
    def forward(self, img_size, pred_boxes, anchor_boxes, targets):
        """
            pred_boxes: (Tensor)   [B, num_queries, 4]
            anchor_boxes: (Tensor) [num_queries, 4]
            targets: (Dict) dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}
        """

        bs, num_queries = pred_boxes.shape[:2]
        # [M, 4] -> [1, M, 4] -> [B, M, 4]
        anchor_boxes = anchor_boxes[None].repeat(bs, 1, 1)

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_anchors, 4]
        out_bbox = pred_boxes.flatten(0, 1)
        anchor_boxes = anchor_boxes.flatten(0, 1)

        # Also concat the target boxes
        tgt_bbox = torch.cat([v['boxes'] for v in targets]).to(out_bbox.device)
        tgt_bbox = tgt_bbox * img_size

        # Compute the L1 cost between boxes
        # Note that we use anchors and predict boxes both
        cost_bbox = torch.cdist(box_xyxy_to_cxcywh(out_bbox), 
                                box_xyxy_to_cxcywh(tgt_bbox), 
                                p=1)
        cost_bbox_anchors = torch.cdist(anchor_boxes, 
                                        box_xyxy_to_cxcywh(tgt_bbox), 
                                        p=1)

        # Final cost matrix
        C = cost_bbox
        C = C.view(bs, num_queries, -1).cpu()
        C1 = cost_bbox_anchors
        C1 = C1.view(bs, num_queries, -1).cpu()

        sizes = [len(v['boxes']) for v in targets]  # the number of object instances in each image
        all_indices_list = [[] for _ in range(bs)]
        # positive indices when matching predict boxes and gt boxes
        indices = [
            tuple(
                torch.topk(
                    c[i],
                    k=self.match_times,
                    dim=0,
                    largest=False)[1].numpy().tolist()
            )
            for i, c in enumerate(C.split(sizes, -1))
        ]
        # positive indices when matching anchor boxes and gt boxes
        indices1 = [
            tuple(
                torch.topk(
                    c[i],
                    k=self.match_times,
                    dim=0,
                    largest=False)[1].numpy().tolist())
            for i, c in enumerate(C1.split(sizes, -1))]

        # concat the indices according to image ids
        for img_id, (idx, idx1) in enumerate(zip(indices, indices1)):  # indices:每张图片中每个目标对应的top_k个预测框的序号 indices1:每张图片中每个目标对应的top_k个anchor的序号（b, top_k, 目标数）
            img_idx_i = [
                np.array(idx_ + idx1_)
                for (idx_, idx1_) in zip(idx, idx1)
            ]  # 合并，前面是预测框序号，后面是anchor序号，[top_k, 2*目标数]
            img_idx_j = [
                np.array(list(range(len(idx_))) + list(range(len(idx1_))))
                for (idx_, idx1_) in zip(idx, idx1)
            ]  # 目标在图片中的编号[0,1,2,3,...,0,1,2,3,...]
            all_indices_list[img_id] = [*zip(img_idx_i, img_idx_j)]

        # re-organize the positive indices
        all_indices = []
        for img_id in range(bs):  # 变换存储结构
            all_idx_i = []
            all_idx_j = []
            for idx_list in all_indices_list[img_id]:
                idx_i, idx_j = idx_list
                all_idx_i.append(idx_i)
                all_idx_j.append(idx_j)
            all_idx_i = np.hstack(all_idx_i)
            all_idx_j = np.hstack(all_idx_j)
            all_indices.append((all_idx_i, all_idx_j))


        return [(torch.as_tensor(i, dtype=torch.int64), 
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in all_indices]
