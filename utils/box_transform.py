import torch


def point_form(boxes):
    """将 cx, cy, w, h 形式的点转换为 x1, y1, x2, y2 形式
    
    Args:
        boxes: shape [N, 4]
    Return:
        shape [N, 4]
    """
    return torch.cat((boxes[:, 0:2]-boxes[:, 2:4]/2, boxes[:, 0:2]+boxes[:, 2:4]/2), dim=1)

def center_form(boxes):
    """将 x1, y1, x2, y2 形式的点转换为 cx, cy, w, h 形式
    
    Args:
        boxes: shape [N, 4]
    Return:
        shape [N, 4]
    """
    return torch.cat(((boxes[:, 0:2]+boxes[:, 2:4])/2, boxes[:, 2:4]-boxes[:, 0:2]), dim=1)


def encode(boxes, priors, config):
    """boxes 和 priors 为 cx, cy, w, h 形式
        boxes 与相匹配的 priors 进行 encode
    Args:
        boxes: shape [N, 4]
        priors: shape [N, 4]
    Return:
        shape [N, 4]
    """
    cxcy_target = (boxes[:, 0:2] - priors[:, 0:2]) / (config["variance"][0] * priors[:, 2:4])
    wh_target = torch.log(boxes[:, 2:4] / priors[:, 2:4]) / config["variance"][1]
    return torch.cat((cxcy_target, wh_target), dim=1)

def decode(out_loc, priors, config):
    """priors 为 cx, cy, w, h 形式
        priors 与 输出进行 decode
    Args:
        out_loc: shape [N, 4]
        priors: shape [N, 4]
    Return:
        boxes: shape [N, 4], 为 x1, y1, x2, y2 形式
    """
    boxes_cxcy = out_loc[:, 0:2] * config["variance"][0] * priors[:, 2:4] + priors[:, 0:2]
    boxes_wh = torch.exp(out_loc[:, 2:4] * config["variance"][1]) * priors[:, 2:4]
    boxes = torch.cat((boxes_cxcy, boxes_wh), dim=1)
    
    # 前面为解码过程
    # 转换为 x1, y1, x2, y2 形式
    boxes[:, 0:2] -= boxes[:, 2:4] / 2
    boxes[:, 2:4] += boxes[:, 0:2]
    return boxes


def encode_landmark(gt_landmark, priors, config):
    """priors 为 cx, cy, w, h 形式
        gt_landmark 与相匹配的 priors 进行 encode
    Args:
        gt_landmark: shape [N, n*2], n 为 landmark 点的个数
        priors: shape [N, 4]
    Return:
        shape [N, 8]
    """
    gt_landmark = gt_landmark.reshape(gt_landmark.shape[0], -1, 2)
    # b, 5
    # N, n, 2
    # N, 4
    landmark_x = (gt_landmark[:, :, 0] - priors[:, 0:1]) / priors[:, 2:3] / config["variance"][0]
    landmark_y = (gt_landmark[:, :, 1] - priors[:, 1:2]) / priors[:, 3:4] / config["variance"][0]
    return torch.stack([landmark_x, landmark_y], dim=2).reshape(gt_landmark.shape[0], -1)
    

def decode_landmark(out_landmark, priors, config):
    """priors 为 cx, cy, w, h 形式
        priors 与 输出进行 decode
    Args:
        out_landmark: shape [n, 2]
        priors: shape [N, 4]
    Return:
        landmark: shape [n, 2]
    """
    # N, n, 2
    out_landmark = out_landmark.reshape(out_landmark.shape[0], -1, 2)
    # N, n, 2
    landmark_x = out_landmark[:, :, 0] * priors[:, 2:3] * config["variance"][0] + priors[:, 0:1]
    landmark_y = out_landmark[:, :, 1] * priors[:, 3:4] * config["variance"][0] + priors[:, 1:2]
    return torch.stack([landmark_x, landmark_y], dim=-1)
    
