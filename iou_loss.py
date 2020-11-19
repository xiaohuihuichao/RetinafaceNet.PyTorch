import torch


def get_inter_area(box_preds, box_gts):
    """inter_area [x_min, y_min, x_max, y_max]\n
    Args:
        box_preds: shape [N, 4]
        box_gts: shape [N, 4]
    Return:
        inter_area: shape [N]
    """
    left_top = torch.max(box_preds[:, 0:2], box_gts[:, 0:2])
    right_bottem = torch.min(box_preds[:, 2:4], box_gts[:, 2:4])
    inter_wh = right_bottem - left_top
    return inter_wh[:, 0] * inter_wh[:, 1]

def get_area(boxes):
    """
    Args
        box: shape [N, 4]
    Return:
        area: shape [N]
    """
    return (boxes[:, 2]-boxes[:, 0]) * (boxes[:, 3]-boxes[:, 1])

def get_iou(box_preds, box_gts):
    """iou [x_min, y_min, x_max, y_max]\n
    Args:
        box_preds: shape [N, 4]
        box_gts: shape [N, 4]
    Return:
        iou: shape [N]
    """
    box_preds_ares = get_area(box_preds)
    box_gt_ares = get_area(box_gts)
    
    inter_area = get_inter_area(box_preds, box_gts)
    return inter_area / (box_preds_ares + box_gt_ares - inter_area + 1e-6)


def iou_loss(box_preds, box_gts):
    """iou [x_min, y_min, x_max, y_max]\n
    Args:
        box_preds: shape [N, 4]
        box_gts: shape [N, 4]
    Return:
        iou_loss
    """
    return 1 - get_iou(box_preds, box_gts).mean()


def get_Ac(box_preds, box_gts):
    """同时包含 pred 和 gt 的最小 box
    """
    Ac_left_top = torch.min(box_preds[:, 0:2], box_gts[:, 0:2])
    Ac_right_bottem = torch.max(box_preds[:, 2:4], box_gts[:, 2:4])
    return torch.cat([Ac_left_top, Ac_right_bottem], dim=1)


def giou_loss(box_preds, box_gts):
    """iou [x_min, y_min, x_max, y_max]\n
    Args:
        box_preds: shape [N, 4]
        box_gts: shape [N, 4]
    Return:
        iou_loss
    """
    iou = get_iou(box_preds, box_gts)
    inter_area = get_inter_area(box_preds, box_gts)
    
    Ac = get_Ac(box_preds, box_gts)
    Ac_area = get_area(Ac)
    
    giou = iou - (Ac_area-inter_area) / (Ac_area + 1e-6)
    return 1 - giou.mean()
    
    
def get_box_center(boxes):
    """
    Args:
        boxes: shape [N, 4]
    Return:
        center: shape [N, 2]
    """
    box_x = (boxes[:, 0] + boxes[:, 2]) / 2
    box_y = (boxes[:, 1] + boxes[:, 3]) / 2
    return torch.cat([box_x, box_y], dim=1)

def get_distance(pointsA, pointsB):
    """
    Args:
        pointsA: shape [N, 2]
        pointsB: shape [N, 2]
    Return:
        distance: shape [N]
    """
    offset = pointsA - pointsB
    return torch.sqrt(offset[0]**2+offset[1]**2)

def diou_loss(box_preds, box_gts):
    """iou [x_min, y_min, x_max, y_max]\n
    Args:
        box_preds: shape [N, 4]
        box_gts: shape [N, 4]
    Return:
        iou_loss
    """
    iou = get_iou(box_preds, box_gts)
    
    center_preds = get_box_center(box_preds)
    center_gts = get_box_center(box_gts)
    center_distance = get_distance(center_preds, center_gts)
    
    Ac = get_Ac(box_preds, box_gts)
    c_2 = get_distance(Ac[:, 0:2], Ac[:, 2:4])
    
    diou = iou - center_distance / (c_2 + 1e-6)
    return 1 - diou.mean()


def ciou_loss(box_preds, box_gts):
    """
    Args:
        box_preds: shape [N, 4]
        box_gts: shape [N, 4]
    Return:
        ciou_loss
    """
    pass
