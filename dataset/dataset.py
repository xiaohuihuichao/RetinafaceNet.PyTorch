import os
import numpy as np
from PIL import Image

import cv2
import torch
from torch.utils.data import Dataset
# ============================================
from torchvision.transforms import ToTensor
# ============================================

def cv2pil(img_cv):
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def bright_adjust(img, alpha=1.2, beta=127):
    img_mean = img.mean()
    img_adjust = (img - img_mean) * alpha + beta
    img_adjust = img_adjust.clip(0, 255)
    return np.asarray(img_adjust, np.uint8)


class dataset(Dataset):
    def __init__(self, data_path, cls_path, config):
        assert os.path.isfile(data_path), f"{data_path} 不存在"
        assert os.path.isfile(cls_path), f"{cls_path} 不存在"
        super().__init__()
        
        self.num_landmark_points = config["num_landmark_points"]
        self.hw = (config["image_size"], config["image_size"])
        self.cls2idx = self.__parse_cls_file(cls_path)
        self.num_classes = len(self.cls2idx)
        self.img_paths, self.clses, self.boxes, self.landmarks = self.__parse_data_path(data_path)
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        
        img_tensor = self.__get_img_tensor(img_path)
        # ============================================
        # img_tensor = torch.rand(self.hw[0], self.hw[1], 3)
        # ============================================
        
        clses_tensor = torch.Tensor([self.clses[index]]).type(torch.int64)
        boxes_tensor = torch.Tensor(self.boxes[index])
        landmark_list = self.landmarks[index]
        return {"img": img_tensor, "cls": clses_tensor, "box": boxes_tensor, "landmark": landmark_list, "num_landmark_points": self.num_landmark_points}
        
    def __get_img_tensor(self, img_path):
        img_cv = cv2.imread(img_path)
        img_cv = cv2.resize(img_cv, dsize=self.hw)
        img_cv = bright_adjust(img_cv)
        
        # ============================================
        # img_tensor = torch.Tensor(img_cv).permute(2, 0, 1) / 255
        # ============================================
        
        img_pil = cv2pil(img_cv)
        img_tensor = ToTensor()(img_pil)
        return (img_tensor-0.5) / 0.5
        
    
    def __parse_cls_file(self, cls_path):
        cls2idx = {}
        with open(cls_path, "r") as f:
            lines = f.readlines()
        lines = [i.strip() for i in lines]
        for idx, name in enumerate(lines, 1):
            cls2idx[name] = idx
        return cls2idx
    
    def __parse_data_path(self, data_path):
        img_paths = []
        clses = []
        boxes = []
        landmarks = []
        with open(data_path, "r") as f:
            lines= f.readlines()
        for line in lines:
            cls_per_img = []
            boxes_per_img = []
            landmarks_per_img= []
            
            img_path, messages = line.strip().split(" ")
            
            img_paths += [img_path]
            messages = messages.split(";")
            for message in messages:
                message = message.split(",")
                cls_name = message[-1]
                cls_per_img += [self.cls2idx[cls_name]]
                
                box_landmark = [float(i) for i in message[0:-1]]
                x1, y1, x2, y2 = box_landmark[0:4]
                boxes_per_img += [[x1, y1, x2, y2]]
                
                landmark = box_landmark[4:(4+self.num_landmark_points*2)]
                landmarks_per_img += [landmark]
            clses += cls_per_img
            boxes += [boxes_per_img]
            landmarks += [landmarks_per_img]
        return img_paths, clses, boxes, landmarks


def collate_fn(batch_data):
    batch_size = len(batch_data)
    num_landmark = batch_data[0]["num_landmark_points"] * 2
    imgs_tensor = torch.stack([i["img"] for i in batch_data], dim=0)
    
    num_objs = [i["cls"].shape[0] for i in batch_data]
    max_num_obj = max(num_objs)
    clses_tensor = torch.zeros((batch_size, max_num_obj), dtype=torch.int64) - 1
    boxes_tensor = torch.zeros((batch_size, max_num_obj, 4)) - 1
    landmark_tensor = torch.zeros((batch_size, max_num_obj, num_landmark)) - 1
    for idx, (data, num_obj) in enumerate(zip(batch_data, num_objs)):
        clses_tensor[idx, 0:num_obj] = data["cls"]
        boxes_tensor[idx, 0:num_obj, :] = data["box"]
        # 有些 obj 没有标注 landmark, 没有标注的 obj 默认为 -1
        for idx_obj, landmark in enumerate(data["landmark"]):
            if len(landmark) == 0:
                continue
            landmark_tensor[idx, idx_obj, :] = torch.Tensor(landmark)
    return imgs_tensor, clses_tensor, boxes_tensor, landmark_tensor
    