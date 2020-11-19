import torch


def intersect(box_a, box_b):
    """box_a ∩ box_b, 相交区域的面积, [x_min, y_min, x_max, y_max]\n
    Args:
        box_a: shape [A, 4]
        box_b: shape [B, 4]
    Return:
        inte: shape [A, B]
    """
    A = box_a.size(0)
    B = box_b.size(0)

    max_xy = torch.min(box_a[:, 2:4].unsqueeze(1).expand(A, B, 2),
                        box_b[:, 2:4].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, 0:2].unsqueeze(1).expand(A, B, 2),
                        box_b[:, 0:2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard_iou(box_a, box_b):
    """iou
    Args:
        box_a: shape [A, 4]
        box_b: shape [B, 4]
    Return:
        inte: shape [A, B]
    """
    inter = intersect(box_a, box_b) # shape [A, B]
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]) # shape [A]
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]) # shape [B]

    area_a = area_a.unsqueeze(1).expand_as(inter) # (A, B)
    area_b = area_b.unsqueeze(0).expand_as(inter) # (A, B)

    return inter / (area_a + area_b - inter)
