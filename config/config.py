import torch

config = {
    "cuda": True and torch.cuda.is_available(),
    "backbone_scale": 1,
    "out_channel": 256,
    "image_size": 640,
    "min_sizes": [42, 91, 140],
    "max_sizes": [91, 140, 189],
    "ratios": [[1, 2, 3, 4, 5.5], [1, 2, 3, 4, 5.5], [1, 2, 3, 4, 5.5]],
    "variance": [0.1, 0.2],
    "num_landmark_points": 4,
}
