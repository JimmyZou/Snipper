import argparse
import os
import torch
from tqdm import tqdm
from models.model import build_model
from inference_utils import get_all_samples, associate_snippets, save_visual_results, save_as_videos, visualize_heatmaps


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

    parser.add_argument('--output_dir', default='C:/Users/shihaozou/Desktop/exps',
                        help='path where to save, empty for no saving')
    parser.add_argument('--data_dir', default='C:/Users/shihaozou/Desktop/exps/seq3',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='C:/Users/shihaozou/Desktop/models/12-06_20-17-34/checkpoint.pth',
                        help='resume from checkpoint')
    parser.add_argument('--vis_heatmap_frame_name', default=None,
                        help='visualize heatmap of a frame, None means all the sampled frames')

    # * dataset parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    parser.add_argument('--input_width', default=800, type=int,
                        help="input image shape (H, W)")
    parser.add_argument('--input_height', default=600, type=int,
                        help="input image shape (H, W)")
    parser.add_argument('--max_depth', type=int, default=15)
    parser.add_argument('--num_frames', default=4, type=int, help="Number of frames")
    parser.add_argument('--num_future_frames', default=0, type=int, help="Number of frames")
    parser.add_argument('--seq_gap', default=5, type=int, help="Number of maximum gap frames")
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
    return parser


def main(args):
    device = torch.device(args.device)
    print(device)

    print('loading model from {}'.format(args.resume))
    model, criterion, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    else:
        raise ValueError('cannot find model {}'.format(args.resume))
    model.eval()

    print('predicting multi-person poses')
    all_samples, frame_indices, all_filenames = get_all_samples(args)  # snippet of images
    # count = 0
    results, results_heatmaps = [], {}
    with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
        for samples in tqdm(all_samples):
            imgs = samples['imgs'].to(device).unsqueeze(dim=0)  # batchsize = 1
            input_size = samples['input_size'].to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]
            outputs, _ = model(imgs)

            max_depth = args.max_depth
            bs, num_queries = outputs["pred_logits"].shape[:2]
            for i in range(bs):
                human_prob = outputs["pred_logits"][i].softmax(-1)[..., 1]

                _out_kepts_depth = outputs["pred_depth"][i]  # n x T x num_kpts x 1
                # root + displacement
                _out_kepts_depth[:, :, 1:, :] = _out_kepts_depth[:, :, 0:1, :] + _out_kepts_depth[:, :, 1:, :] / max_depth
                out_kepts_depth = max_depth * _out_kepts_depth  # scale to original depth

                out_score = outputs["pred_kpts2d"][i, :, :, :, 2:3]  # n x T x num_kpts x 1
                out_kepts2d = outputs["pred_kpts2d"][i, :, :, :, 0:2]  # n x T x num_kpts x 2
                # root + displacement
                out_kepts2d[:, :, 1:, :] = out_kepts2d[:, :, :1, :] + out_kepts2d[:, :, 1:, :]
                out_kepts2d = out_kepts2d * input_size  # scale to original image size

                inv_trans = samples['inv_trans']
                input_size = samples['input_size']
                img_size = samples['img_size']
                filenames = samples['filenames']
                results.append(
                    {
                        'human_score': human_prob.cpu().numpy(),  # [n]
                        'pred_kpt_scores': out_score.cpu().numpy(),  # [n, T, num_joints, 1]
                        'pred_kpts': out_kepts2d.cpu().numpy(),  # [n, T, num_kpts, 2]
                        'pred_depth': out_kepts_depth.cpu().numpy(),  # [n, T, num_kpts, 1]
                        'inv_trans': inv_trans.cpu().numpy(),  # [2, 3]
                        'filenames': filenames,  # [filename_{t}, filename_{t+gap}, ...]
                        'input_size': input_size.cpu().numpy(),  # (w, h)
                        'img_size': img_size.cpu().numpy(),  # (w, h)
                    }
                )

                _heatmaps = [heatmap[i].mean(dim=-2) for heatmap in outputs['heatmaps']]
                heatmaps = _heatmaps[0].cpu().numpy()  # [T, h, w, num_joints]
                _, _, h, w = imgs.shape
                imgs = imgs[0].reshape(-1, 3, h, w).permute(0, 2, 3, 1).cpu().numpy()
                for t in range(args.num_frames):
                    results_heatmaps[filenames[t]] = (heatmaps[t], imgs[t])  # [h, w, num_joints]

            # count += 1
            # if count > 3:
            #     break

    print('associating multi-person prediction between snippets')
    all_frames_results, max_pid = associate_snippets(results, frame_indices, all_filenames, args)

    seq_name = args.data_dir.split('/')[-1]
    save_dir = '{}/{}_predictions'.format(args.output_dir, seq_name)
    print('save visual results of each frame in {}'.format(save_dir))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_visual_results(all_frames_results, all_filenames, args.data_dir, save_dir,
                        max_pid, args.max_depth, args.seq_gap)

    print('save visual results as gif')
    all_frames_idx = list(sorted(all_frames_results.keys()))
    save_as_videos(save_dir, all_frames_idx, all_filenames)

    print('save heatmaps')
    if not os.path.exists('{}/heatmaps'.format(save_dir)):
        os.mkdir('{}/heatmaps'.format(save_dir))
    if args.vis_heatmap_frame_name is None:
        # save heatmap of all the frames
        for filename in results_heatmaps.keys():
            heatmaps, img = results_heatmaps[filename]
            visualize_heatmaps(heatmaps, img, save_dir, filename)
    else:
        frame_name = args.vis_heatmap_frame_name.split('.')[0]
        if not os.path.exists('{}/{}_predictions/track2d/{}_track.jpg'.format(args.output_dir, seq_name, frame_name)):
            raise ValueError('frame {} is not used for prediction'.format(args.vis_heatmap_frame_name))
        heatmaps, img = results_heatmaps[args.vis_heatmap_frame_name]
        visualize_heatmaps(heatmaps, img, save_dir, args.vis_heatmap_frame_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Snipper inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)



