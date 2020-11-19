import torch

if __name__ == "__main__":
    from iou import jaccard_iou
else:
    from .iou import jaccard_iou


def match(prior_boxes, target_boxes, iou_threshold=0.4):
    """prior_boxes, target_boxes 都是 x1, y1, x2, y2 形式
    Args:
        prior_boxes: 预设框 shape [num_prior, 4]
        target_boxes: 物体框 shape [num_obj, 4]
    Return:
        pos_priors_mask: 与 target 匹配的 prior 的 mask ，正样本
        idx_obj: 相应的 pos_priors 对应的目标在 target_boxes 中的索引
    """
    # print(target_boxes)
    # 剔除全为 -1 的内容
    target_boxes = target_boxes[target_boxes.sum(1)>0]
    if target_boxes.shape[0] == 0:
        return None, None
        
    iou_table = jaccard_iou(prior_boxes, target_boxes)
    # 1. 获取每个 prior 对所有 target 的最大 iou
    max_iou_each_prior = iou_table.max(dim=1)[0]
    # 2. 最大 iou 大于阈值 的 prior 的 mask
    # pos_priors_mask = torch.where(max_iou_each_prior>iou_threshold,
    #                                 torch.ones_like(max_iou_each_prior, dtype=torch.bool),
    #                                 torch.zeros_like(max_iou_each_prior, dtype=torch.bool))
    pos_priors_mask = torch.gt(max_iou_each_prior, iou_threshold).type(torch.bool)
    
    # print(iou_table[pos_priors_mask].shape)
    try:
        idx_obj = torch.argmax(iou_table[pos_priors_mask], dim=1)
    except Exception:
        return None, None
    # 3. 保证一个 target 至少有一个 prior 与之匹配（这一步没有实现, 只要prior的wh设计得合理，且iou_threshold不设得过大，一般就不需要这一步）
    return pos_priors_mask, idx_obj
