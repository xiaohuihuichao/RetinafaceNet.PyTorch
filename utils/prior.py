import math
import torch
from itertools import product


def get_prior(feature_map_size, image_size, min_size, max_size, ratio):
    """生成 prior\n
    prior 为 cx, cy, w, h 形式
    Args:
        feature_map_size: feature_map 的 h w
        image_size: 原始图片大小, 短边？
        min_size: 小 scale
        max_size: 大 scale
        ratio: 宽高比 w / h
    Return:
        prior_boxes: shape [N, 4]
    """
    h, w = feature_map_size

    half_grid_w, half_grid_h = 0.5 / w, 0.5 / h
    prior_boxes = []
    for y, x in product(range(h), range(w)):
        cx = x / w + half_grid_w
        cy = y / h + half_grid_h

        # 短边
        s_k = min_size / image_size
        # prior_boxes += [cx, cy, s_k, s_k]
        # 长边
        s_k_ = math.sqrt(s_k * max_size / image_size)
        # prior_boxes += [cx, cy, s_k_, s_k_]
        
        for r in ratio:
            prior_boxes += [cx, cy, s_k*math.sqrt(r), s_k/math.sqrt(r)]
            prior_boxes += [cx, cy, s_k_*math.sqrt(r), s_k_/math.sqrt(r)]
            # prior_boxes += [cx, cy, s_k/math.sqrt(r), s_k*math.sqrt(r)]
    prior_boxes = torch.Tensor(prior_boxes).view(-1, 4)
    return prior_boxes.clamp(max=1, min=0)
