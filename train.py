import os
import time

import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader

from net import model
from config import config
from utils import loss_func
from dataset import dataset, collate_fn


def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt)
    return rt / dist.get_world_size()


def save_model(net, path):
    state = {}
    for k, v in net.state_dict().items():
        state[k.replace("module.", "")] = v.clone().cpu()
    torch.save(state, path)

def log(msg, file, args, append=True):
    if args.local_rank == 0:
        if append:
            t = ">>"
        else:
            t = ">"
        os.system(f"echo '{msg}' {t} '{file}'")


def train_batch(net, batch_data, criterion, optimizer, cuda, args):
    optimizer.zero_grad()
    if cuda:
        batch_data = [i.cuda() for i in batch_data]
    imgs = batch_data[0]
    gts = batch_data[1:]
    
    preds = net(imgs)
    box_loss, landmark_loss, cls_loss = criterion(preds, gts)
    loss = args.box_rate*box_loss + args.landmark_rate*landmark_loss + args.cls_rate*cls_loss
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(net.parameters())
    optimizer.step()
    return loss, box_loss, landmark_loss, cls_loss
    

def main(args):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    
    
    ds = dataset(args.data_file, args.class_file, config)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=False, sampler=sampler)
    
    batch_save_path = f"{args.model_dir}/batch_4.pth"
    epoch_save_path = f"{args.model_dir}/epoch_4.pth"
    
    net = model(config, ds.num_classes)
    if os.path.isfile(batch_save_path):
        log("载入模型中...", args.log_detail_path, args)
        try:
            net.load_state_dict(torch.load(batch_save_path))
            log("模型载入完成！", args.log_detail_path, args)
        except Exception as e:
            log(f"{e}\n载入模型失败: {batch_save_path}", args.log_detail_path, args)
    else:
        log(f"没找到模型: {batch_save_path}", args.log_detail_path, args)
    
    config["cuda"] = config["cuda"] and torch.cuda.is_available()
    if config["cuda"]:
        # net = torch.nn.DataParallel(net.cuda())
        net = torch.nn.parallel.DistributedDataParallel(net.cuda(), device_ids=[args.local_rank])
        log("cuda", args.log_detail_path, args)
    
    criterion = loss_func(config)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, factor=args.lr_decay, threshold=1e-3)
    schedule_loss = []
    
    net.train()
    for epoch in range(1, args.epochs+1):
        log(f"{'='*30}\n[{epoch}|{args.epochs}]", args.log_detail_path, args)
        for num_batch, batch_data in enumerate(dl, 1):
            t = time.time()
            loss, box_loss, landmark_loss, cls_loss = train_batch(net, batch_data, criterion, optimizer, config["cuda"], args)
            t = time.time() - t
            
            loss, box_loss, landmark_loss, cls_loss = [reduce_tensor(i).item() for i in [loss, box_loss, landmark_loss, cls_loss]]
            
            msg = f"  [{epoch}|{args.epochs}] num_batch:{num_batch}" \
                + f" loss:{loss:.4f} box_loss:{box_loss:.4f} landmark_loss:{landmark_loss:.4f} cls_loss:{cls_loss:.4f} time:{t*1000:.1f}ms"
            log(msg, args.log_detail_path, args)
            if num_batch % args.num_show == 0:
                log(msg, args.log_path, args)
                
            if args.local_rank == 0:
                if num_batch % args.num_save == 0:
                    save_model(net, batch_save_path)
                    
                schedule_loss += [loss]
                if num_batch % args.num_adjuest_lr == 0:
                    scheduler.step(np.mean(schedule_loss))
                    schedule_loss = []
        if args.local_rank == 0:
            save_model(net, epoch_save_path)
