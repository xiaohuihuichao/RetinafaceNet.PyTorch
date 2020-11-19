import os
import cv2
import numpy as np
from glob import glob
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from net import model
from config import config
from utils import decode, decode_landmark, get_prior, jaccard_iou

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def bright_adjust(img, alpha=1.2, beta=127):
    img_mean = img.mean()
    img_adjust = (img - img_mean) * alpha + beta
    img_adjust = img_adjust.clip(0, 255)
    return np.asarray(img_adjust, np.uint8)

def cv2pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


num_classes = 1
class_txt = "classes.txt"
img_dir = "imgs"
model_path = "model/batch_4.pth"
save_path = "r.jpg"
threshold = 0.8
iou_threshold = 0.3

cuda = torch.cuda.is_available() and True


def imshow(win_name, img, t=0):
    cv2.namedWindow(win_name, cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(win_name, img)
    return cv2.waitKey(t)


def nms(conf, box, iou_threshold=0.5):
    """最原始的 NMS \n
    Args:
        conf: shape [N]
        box: shape [N, 4]
    Return:
        keep_index: list
    """
    keep_index = []

    _, order_index = conf.sort(descending=True) # 降序

    while order_index.numel() > 0:
        keep_index.append(order_index[0].item())
        if order_index.numel() == 1:
            break

        ious = jaccard_iou(box[order_index[0:1]], box[order_index]).reshape(-1)
        k = ious <= iou_threshold
        order_index = order_index[k]
    return keep_index

def idx2cls(class_txt):
    with open(class_txt, "r") as f:
        lines = f.readlines()
    return [i.strip() for i in lines]
    

priors = None
with torch.no_grad():
    net = model(config, num_classes, mode="eval").eval()
    if os.path.isfile(model_path):
        net.load_state_dict(torch.load(model_path))
    else:
        print(f"没有这个文件:{model_path}")
    if cuda:
        net = net.cuda()
    
    class_names = idx2cls(class_txt)
    
    img_paths = glob(f"{img_dir}/*.jpg")
    print(len(img_paths))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_cv = cv2.resize(img, dsize=(config["image_size"], config["image_size"]))
        imshow("i", img, 1)
        img_cv = bright_adjust(img_cv)

        img_pil = cv2pil(img_cv)
        img_tensor = ToTensor()(img_pil).unsqueeze(0)
        img_tensor = (img_tensor-0.5) / 0.5

        if cuda:
            img_tensor = img_tensor.cuda()

        t = time.time()
        classifications, bbox_regression, landmark_regression = net(img_tensor)

        if priors == None:
            feature_map_sizes = [i.shape[1:3] for i in bbox_regression]
            priors = [get_prior(feature_map_size, config["image_size"], min_size, max_size, ratio)for feature_map_size, min_size, max_size, ratio in zip(feature_map_sizes, config["min_sizes"], config["max_sizes"], config["ratios"])]
            priors = torch.cat(priors, dim=0)
            if cuda:
                priors = priors.cuda()

        bbox_regression = torch.cat([i.reshape(-1, 4) for i in bbox_regression], dim=0)
        landmark_regression = torch.cat([i.reshape(-1, 8) for i in landmark_regression], dim=0)
        classifications = torch.cat([i.reshape(-1) for i in classifications], dim=0)
        classifications = classifications.reshape(landmark_regression.shape[0], -1)


        max_conf, max_idx = torch.max(classifications[:, 1:], dim=1)
        obj_mask = torch.gt(max_conf, threshold)#.type(torch.uint8)
        if obj_mask.sum() < 1:
            print("没有检测到物体")
            continue

        obj_conf = max_conf[obj_mask]
        # 0是第一个类别，不是背景
        obj_clses = max_idx[obj_mask]


        obj_boxes = decode(bbox_regression[obj_mask], priors[obj_mask], config)
        obj_boxes *= config["image_size"]
        obj_landmarks = decode_landmark(landmark_regression[obj_mask], priors[obj_mask], config)
        obj_landmarks *= config["image_size"]
        
        keep_index = nms(obj_conf, obj_boxes, iou_threshold)
        print(f"检测到{len(keep_index)}个物体")
        t = time.time() - t
        for idx in keep_index:
            
            x1, y1, x2, y2 = obj_boxes[idx]
            img_cv = cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=1)
            
            class_name = class_names[obj_clses[idx]]
            img_cv = cv2.putText(img_cv, f"{class_name}:{obj_conf[idx]:.3f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_TRIPLEX, 1, color=(0, 0, 255), thickness=2)
            
            lx1, ly1, lx2, ly2, lx3, ly3, lx4, ly4 = obj_landmarks[idx].reshape(-1)
            img_cv = cv2.circle(img_cv, (int(lx1), int(ly1)), radius=3, color=(0, 255, 0), thickness=3)
            img_cv = cv2.circle(img_cv, (int(lx2), int(ly2)), radius=3, color=(0, 255, 0), thickness=3)
            img_cv = cv2.circle(img_cv, (int(lx3), int(ly3)), radius=3, color=(0, 255, 0), thickness=3)
            img_cv = cv2.circle(img_cv, (int(lx4), int(ly4)), radius=3, color=(0, 255, 0), thickness=3)
            
        cv2.imwrite(save_path, img_cv)
        print(f"用时{t*1000:.1f}ms")
        
        c = imshow("img", img_cv, 0)
        if c == ord("q"):
            break
        elif c == ord("l"):
            net.load_state_dict(torch.load(model_path))
            continue
        