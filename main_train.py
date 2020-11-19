import time
import argparse

from train import log, main


def get_args():
    parser = argparse.ArgumentParser(description="训练参数")
    
    parser.add_argument("--local_rank", default=-1, type=int)
    
    parser.add_argument("--data_file", default="/home/hzc/OCR/retinaface/label.txt", type=str)
    parser.add_argument("--class_file", default="/home/hzc/OCR/retinaface/cls.txt", type=str)
    
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_decay", default=0.8, type=float)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--num_workers", default=16, type=int)
    
    parser.add_argument("--box_rate", type=float, default=0.25)
    parser.add_argument("--landmark_rate", type=float, default=0.1)
    parser.add_argument("--cls_rate", type=float, default=1)
    
    parser.add_argument("--num_adjuest_lr", default=400, type=int)
    parser.add_argument("--num_show", default=100, type=int)
    parser.add_argument("--num_save", default=400, type=int)
    parser.add_argument("--model_dir", default="model", type=str)
    
    parser.add_argument("--log_detail_path", default="log/detail_train_4.log", type=str)
    parser.add_argument("--log_path", default="log/train_4.log", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    if args.local_rank == 0:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log(f"训练开始于 {start_time}", args.log_detail_path, args)
        log(f"训练开始于 {start_time}", args.log_path, args)
    
    main(args)
    
    if args.local_rank == 0:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log(f"训练结束于 {start_time}", args.log_detail_path, args)
        log(f"训练结束于 {start_time}", args.log_path, args)
