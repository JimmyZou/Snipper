import argparse
import random
import time
import datetime
import json
import numpy as np
import os
import collections
import pickle
import torch
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
    parser.add_argument('--use_amp', default=0, type=int)
    parser.add_argument('--use_pytorch_deform', default=1, type=int)

    parser.add_argument('--output_dir', default='/home/shihao/data/exps',
                        help='path where to save, empty for no saving')
    # parser.add_argument('--output_dir', default=None,
    #                     help='path where to save, empty for no saving')

    # pretrained model
    parser.add_argument('--pretrained_dir', type=str, default=None,
                        help="Path to the pretrained model.")

    # * dataset parameters
    parser.add_argument('--device', default='cuda:3',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='/home/shihao/data/4.pth',
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
    parser.add_argument('--muco_dir', type=str, default='/home/shihao/data/mupots')
    parser.add_argument('--coco_dir', type=str, default='C:/Users/shihaozou/Desktop/MSCOCO/')
    parser.add_argument('--jta_dir', type=str, default='C:/Users/shihaozou/Desktop/jta_dataset/')
    parser.add_argument('--panoptic_dir', type=str, default='C:/Users/shihaozou/Desktop/panoptic-toolbox-master/data/')
    parser.add_argument('--use_posetrack', type=int, default=0)
    parser.add_argument('--use_muco', type=int, default=1)
    parser.add_argument('--use_coco', type=int, default=0)
    parser.add_argument('--use_jta', type=int, default=0)
    parser.add_argument('--use_panoptic', type=int, default=0)
    parser.add_argument('--protocol', type=int, default=1)

    parser.add_argument('--num_frames', default=4, type=int, help="Number of frames")
    parser.add_argument('--num_future_frames', default=0, type=int, help="Number of frames")
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
    parser.add_argument('--hidden_dim', default=384, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
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
    parser.add_argument('--root_vis_loss_coef', default=1, type=float)

    parser.add_argument('--joint_loss_coef', default=1, type=float)
    parser.add_argument('--joint_depth_loss_coef', default=1, type=float)
    parser.add_argument('--joint_vis_loss_coef', default=1, type=float)

    parser.add_argument('--joint_disp_loss_coef', default=1, type=float)
    parser.add_argument('--joint_disp_depth_loss_coef', default=1, type=float)

    parser.add_argument('--cont_loss_coef', default=0.1, type=float)
    parser.add_argument('--heatmap_loss_coef', default=0.01, type=float)

    parser.add_argument('--eos_coef', default=0.25, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    print('output_dir {}'.format(args.output_dir))
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

    dataset_val = build_dataset(image_set='test', args=args)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch'] + 1
    else:
        raise ValueError('cannot find model {}'.format(args.resume))

    output_dir = args.output_dir

    if utils.is_main_process():
        if not os.path.exists('{}/eval_results_{:03d}'.format(output_dir, epoch)):
            os.mkdir('{}/eval_results_{:03d}'.format(output_dir, epoch))
        if not os.path.exists('{}/vis_results_{:03d}'.format(output_dir, epoch)):
            os.mkdir('{}/vis_results_{:03d}'.format(output_dir, epoch))

    print("Start evaluating")
    start_time = time.time()
    # evaluate
    test_stats, _save_data, _save_data_coco = evaluate(
        model, criterion, postprocessors, data_loader_val, device,
        args.output_dir, True, epoch,
        args.num_frames, args.num_future_frames, final_evaluation=True
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
        'epoch': epoch,
        'n_parameters': n_parameters
    }

    if args.output_dir and utils.is_main_process():
        with open('{}/log.txt'.format(output_dir), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

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
            dataset_val.eval_posetrack(gt_dir='{}/annotations/val_joints15/'.format(dataset_val.posetrack_dir),
                                       pred_dir='{}/eval_results_{:03d}/'.format(output_dir, epoch))
        if len(save_data_coco) > 0:
            dataset_val.write_val_results_coco(save_data_coco,
                                               output_dir='{}/eval_results_{:03d}'.format(output_dir, epoch))
            dataset_val.eval_coco_val_results(
                gt_dir='{}/annotations/person_keypoints_val2017_joint15.json'.format(dataset_val.coco_dir),
                pred_dir='{}/eval_results_{:03d}/coco_val2017_predictions.json'.format(output_dir, epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Snipper training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
