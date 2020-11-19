## 基于 PyTorch 的 RetinaFace 目标检测算法实现
---
这是基于 PyTorch 实现的一个简易版 RetinaFace。包含目标检测和关键点检测的部分，Mesh Decoder 部分没有实现。

代码中， prior 和 gt 匹配的 match 函数有一些些小缺陷，但只要 anchor 设计得合理就不会有问题，具体可以看代码 [match](./utils/match.py)。
RetinaFace中的人脸关键点是5个，分别为双眼、鼻尖和2个嘴角，而这里实现的RetinaFace是本人用于车牌检测的，所以只用了4个关键点。为了扩展其通用性，如果需要改变关键点数量，只需修改config文件夹下相应的config中的"num_landmark_points"即可。

### 目录
1. [训练步骤与使用方法](#训练步骤与使用方法)
2. [检测方法的使用](#检测方法的使用)


### 训练步骤与使用方法
#### a. 在 [classes.txt](./classes.txt) 文件填上类别名称，一行一个类别。
#### b. [label.txt](./label.txt) 里是训练数据文件示例。一行文本就是一个图片的数据信息。每一行文本诸如以下形式：
```
path/to/image.jpg x1,y1,x2,y2,landmark_1x,landmark_1y,landmark_2x,landmark_2y,...,classes_name;x1,y1,x2,y2,classes_name;...
```
比如
```
/home/license_plate/license_plate_data/0100395114943-90_82-177&357_391&427-392&426_183&431_173&366_382&361-16_16_26_33_27_28_29-147-60.jpg 0.24583333333333332,0.30775862068965515,0.5430555555555555,0.36810344827586206,0.5444444444444444,0.36724137931034484,0.25416666666666665,0.371551724137931,0.24027777777777778,0.31551724137931036,0.5305555555555556,0.3112068965517241,license_plate
```
表示图片 有一个物体，类别为license_plate，0.24583333333333332,0.30775862068965515,0.5430555555555555,0.36810344827586206为目标检测bbox框的左上与右下坐标，剩下的数字为landmark的xy坐标。
#### c. 设置配置文件 [config.py](./config.py)
    "cuda": 表示是否使用 cuda（实际上如果 cuda 设为 True，如果在无 GPU 机器上还是会使用 cpu且不会报错）。
    "backbone_scale": mobilenet的通道参数。
    "out_channel": retinaface的ssh和head的输出通道数。
    "image_size": 原始图像大小。
    "min_sizes", "max_sizes": 预设框边长设置。
    "ratio": 预设框宽高比设置。
    "variance": 与 decode 和 encode 有关。
    "num_landmark_points": 关键点检测中关键点的个数，retinaface的人脸检测中关键点是5个，我在做车牌检测的时候，关键点设的是4个，为车牌的4个角。

    实际根据检测物体的宽高比和大小修改"min_sizes"， "max_sizes"， "ratio"就可以了。

#### 4. 训练。
运行 [main_train.py](./main_train.py)
关于训练中的一些参数，比如batch_size、lr之类的，可以在[main_train.py](./main_train.py)中设置与更改。
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 main_train.py
```
或者使用训练bash
```
sh train.sh
```

### 检测方法的使用
运行 [infer.py](./infer.py)。
```
python infer.py
```
