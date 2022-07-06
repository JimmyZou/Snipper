import argparse
import random
import time
import datetime
import json
import numpy as np
import os
import pickle
import torch
import collections
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils

from models.model import build_model
from datasets import build_dataset
from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr_drop', default=1, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')
    parser.add_argument('--use_pytorch_deform', default=1, type=int,
                        help='use compiled cuda operator of deformable attention module or pytorch implementation')

    parser.add_argument('--output_dir', default='C:/Users/shihaozou/Desktop/exps',
                        help='path where to save, empty for no saving')
    # parser.add_argument('--output_dir', default=None,
    #                     help='path where to save, empty for no saving')

    # pretrained model
    parser.add_argument('--pretrained_dir', type=str, default=None,
                        help="Path to the pretrained model.")

    # * dataset parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--input_width', default=800, type=int,
                        help="input image shape (H, W)")
    parser.add_argument('--input_height', default=600, type=int,
                        help="input image shape (H, W)")
    parser.add_argument('--max_depth', type=int, default=15)

    parser.add_argument('--dataset_file', default='hybrid')
    parser.add_argument('--posetrack_dir', type=str, default='C:/Users/shihaozou/Desktop/posetrack2018/')
    parser.add_argument('--muco_dir', type=str, default='C:/Users/shihaozou/Desktop/muco/')
    parser.add_argument('--coco_dir', type=str, default='C:/Users/shihaozou/Desktop/MSCOCO/')
    parser.add_argument('--jta_dir', type=str, default='C:/Users/shihaozou/Desktop/jta_dataset/')
    parser.add_argument('--panoptic_dir', type=str, default='C:/Users/shihaozou/Desktop/panoptic-toolbox-master/data/')
    parser.add_argument('--use_posetrack', type=int, default=0)
    parser.add_argument('--use_muco', type=int, default=0)
    parser.add_argument('--use_coco', type=int, default=0)
    parser.add_argument('--use_jta', type=int, default=0)
    parser.add_argument('--use_panoptic', type=int, default=1)
    parser.add_argument('--protocol', type=int, default=1, help="train/test protocol of Panoptic dataset")

    parser.add_argument('--num_frames', default=4, type=int, help="Number of frames")
    parser.add_argument('--num_future_frames', default=2, type=int, help="Number of frames")
    parser.add_argument('--seq_max_gap', default=4, type=int, help="Number of maximum gap frames")
    parser.add_argument('--seq_min_gap', default=4, type=int, help="Number of minimum gap frames")
    parser.add_argument('--num_workers', type=int, default=4)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=3, type=int, help='number of feature levels')

    # * transformer
    parser.add_argument('--hidden_dim', default=192, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--num_queries', default=60, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_kpts', default=15, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * matcher
    parser.add_argument('--set_cost_is_human', default=1, type=float)

    parser.add_argument('--set_cost_root', default=1, type=float)
    parser.add_argument('--set_cost_root_depth', default=1, type=float)
    parser.add_argument('--set_cost_root_vis', default=0.1, type=float)

    parser.add_argument('--set_cost_joint', default=1, type=float)
    parser.add_argument('--set_cost_joint_depth', default=1, type=float)
    parser.add_argument('--set_cost_joint_vis', default=0.1, type=float)

    # * Segmentation
    parser.add_argument('--masks', default=False, type=bool,
                        help="Train segmentation head if the flag is provided")

    # Loss
    # parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
    #                     help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--two_stage', default=False, type=bool)

    # * Loss coefficients
    parser.add_argument('--is_human_loss_coef', default=1, type=float)

    parser.add_argument('--root_loss_coef', default=1, type=float)
    parser.add_argument('--root_depth_loss_coef', default=1, type=float)
    parser.add_argument('--root_vis_loss_coef', default=0.1, type=float)

    parser.add_argument('--joint_loss_coef', default=1, type=float)
    parser.add_argument('--joint_depth_loss_coef', default=1, type=float)
    parser.add_argument('--joint_vis_loss_coef', default=1, type=float)

    parser.add_argument('--joint_disp_loss_coef', default=1, type=float)
    parser.add_argument('--joint_disp_depth_loss_coef', default=1, type=float)

    parser.add_argument('--cont_loss_coef', default=0.1, type=float)
    parser.add_argument('--heatmap_loss_coef', default=0.01, type=float)

    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    print('output_dir {}'.format(args.output_dir))
    print('jta dataset dir {}'.format(args.jta_dir))
    print('input shape ({}, {})'.format(args.input_height, args.input_width))
    print('aux loss {}'.format(args.aux_loss))
    print('hidden dim {}'.format(args.hidden_dim))
    print('pytorch deform {}'.format(args.use_pytorch_deform))

    if args.pretrained_dir:
        os.environ['TORCH_HOME'] = args.pretrained_dir

    utils.init_distributed_mode(args)
    # print("git:\n  {}\n".format(utils.get_sha()))

    device = torch.device(args.device)
    print(device)

    # fix the seed for reproducibility
    # print(utils.get_rank())
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # for n, p in model_without_ddp.named_parameters():
    #     print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and
                    not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if
                       match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # # TODO learning rate decay
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[600, 1200, 1800], gamma=0.1)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='test', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    output_dir = args.output_dir
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # train
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            # checkpoint_paths = [output_dir / 'checkpoint.pth']
            checkpoint_paths = ['{}/checkpoint.pth'.format(output_dir)]
            # extra checkpoint before LR drop and every epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append('{}/checkpoint{:04d}.pth'.format(output_dir, epoch))
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # evaluate
        if (epoch + 1) % 1 == 0:
            if utils.is_main_process():
                if not os.path.exists('{}/eval_results_{:03d}'.format(output_dir, epoch)):
                    os.mkdir('{}/eval_results_{:03d}'.format(output_dir, epoch))
                if not os.path.exists('{}/vis_results_{:03d}'.format(output_dir, epoch)):
                    os.mkdir('{}/vis_results_{:03d}'.format(output_dir, epoch))

            test_stats, _save_data, _save_data_coco = evaluate(
                model, criterion, postprocessors, data_loader_val, device,
                args.output_dir, (epoch + 1) % 1 == 0, epoch,
                args.num_frames, args.num_future_frames
            )
            print(test_stats.keys())
            print(test_stats.values(), '\n')

            if args.distributed and args.output_dir:
                with open('{}/intermediate_results_{}.pkl'.format(output_dir, utils.get_rank()), 'wb') as f:
                    pickle.dump(_save_data, f)

                with open('{}/intermediate_results_coco_{}.pkl'.format(output_dir, utils.get_rank()), 'wb') as f:
                    pickle.dump(_save_data_coco, f)

            log_stats = {
                **{f'test_{k}:': v for k, v in test_stats.items()},
                # **{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if args.output_dir and utils.is_main_process():
                if args.distributed:
                    save_data = collections.defaultdict(list)
                    save_data_coco = collections.defaultdict(list)
                    time.sleep(10)
                    for i in range(torch.cuda.device_count()):
                        with open('{}/intermediate_results_{}.pkl'.format(output_dir, i), 'rb') as f:
                            tmp = pickle.load(f)
                        for k, v in tmp.items():
                            save_data[k] += v

                        with open('{}/intermediate_results_coco_{}.pkl'.format(output_dir, i), 'rb') as f:
                            tmp = pickle.load(f)
                        for k, v in tmp.items():
                            save_data_coco[k] += v
                else:
                    save_data = _save_data
                    save_data_coco = _save_data_coco

                if len(save_data) > 0:
                    dataset_val.write_val_results(dataset_val, save_data,
                                                  output_dir='{}/eval_results_{:03d}'.format(output_dir, epoch))
                    dataset_val.eval_posetrack(gt_dir='{}/annotations/val_joints15/'.format(dataset_val.data_dir),
                                               pred_dir='{}/eval_results_{:03d}/'.format(output_dir, epoch))
                if len(save_data_coco) > 0:
                    dataset_val.write_val_results_coco(save_data_coco,
                                                       output_dir='{}/eval_results_{:03d}'.format(output_dir, epoch))
                    info_str = dataset_val.eval_coco_val_results(
                        gt_dir='{}/annotations/person_keypoints_val2017_joint15.json'.format(dataset_val.coco_data_dir),
                        pred_dir='{}/eval_results_{:03d}/coco_val2017_predictions.json'.format(output_dir, epoch))
                    if info_str is not None:
                        log_stats['coco_val'] = info_str

                with open('{}/log.txt'.format(output_dir), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Snipper training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
