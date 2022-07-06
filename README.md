## Snipper 
### This is the re-implementation of paper "Snipper: A Spatiotemporal Transformer for Simultaneous Multi-Person 3D Pose Estimation Tracking and Forecasting on a Video Snippet"

---
#### Dataset preprocess
- [JTA dataset](https://github.com/fabbrimatteo/JTA-Dataset)
- [MS COCO](https://cocodataset.org/#download)
- [MuCo](https://drive.google.com/drive/folders/1yL2ey3aWHJnh8f_nhWP--IyC9krAPsQN?usp=sharing) parsed and composited data by [Moon](https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/a199d50be5b0a9ba348679ad4d010130535a631d)
- MuPoTS [[images]](https://vcai.mpi-inf.mpg.de/projects/SingleShotMultiPerson/) [[annotations]](https://drive.google.com/drive/folders/1WmfQ8UEj6nuamMfAdkxmrNcsQTrTfKK_?usp=sharing) parsed by [Moon](https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/a199d50be5b0a9ba348679ad4d010130535a631d)
- [PoseTrack2018](https://posetrack.net/)

---
#### Dependencies
- compile cuda version of deformable attention module according to [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
  ```angular2html
  cd ./models/ops
  sh ./make.sh
  # unit test (should see all checking is True)
  python test.py
  ```
- Python 3.6
- PyTorch 1.7.0
- scipy 1.5.2

---
#### Inference

Pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1YxSktW5AzoYz7EfjwXgebusrrhVh5y2o?usp=sharing) or [OneDrive](https://ualbertaca-my.sharepoint.com/:f:/g/personal/szou2_ualberta_ca/EoIi5M1q-GtOnlOd5AD_rz4BWsLH_pW7fCwxpMxR2uImrw?e=G2qZg3).
```
trained models
|-- model/12-06_13-31-59/checkpoint.pth  # T=1, encoder_layer=6, decoder_layer=6
|-- model/12-06_20-17-34/checkpoint.pth  # T=4, encoder_layer=6, decoder_layer=6
|-- model/12-06_20-18-30/checkpoint.pth  # T=4+2, encoder_layer=6, decoder_layer=6
|
|-- model/12-05_06-37-50/checkpoint.pth  # T=1, encoder_layer=2, decoder_layer=4
|-- model/12-05_06-39-49/checkpoint.pth  # T=4, encoder_layer=2, decoder_layer=4
|-- model/12-05_06-39-03/checkpoint.pth  # T=4+2, encoder_layer=2, decoder_layer=4
```

The model ```model/12-06_20-17-34/checkpoint.pth  # T=4, encoder_layer=6, decoder_layer=6``` is used to generate 
the three example demos. For new sequence inference, set the following arguments ```data_dir``` to the target folder.
```angular2html
python inference.py \
    # the path to trained model
    --resume               'model/12-06_20-17-34/checkpoint.pth' \  
    # path to the test sequence
    --data_dir             'demos/seq1' \  
    # path to save predicitions
    --output_dir           'demos' \  
    # number of observed frames
    --num_frames           4 \ 
    # number of forecasting frames
    --num_future_frames    0 \  
    # select snippet every 5 frames (30fps --> 6fps of a snippet)
    --seq_gap              5 \  
    # frame filename to see its heatmaps
    --vis_heatmap_frame_name '000005.jpg'
```
Please remember to remove the comments ```# ...``` before run the command.


---
### Train

Settings to train on multiple datasets  (4 observed frames pose tracking only).
```angular2html
python -u -m torch.distributed.launch --nproc_per_node=8 main.py \
    --output_dir           "$LOG_OUTDIR" \
    --dataset_file         'hybrid' \
    --posetrack_dir        '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/posetrack2018'\
    --use_posetrack        1 \
    --coco_dir             '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/coco' \
    --use_coco             1 \
    --muco_dir             '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/muco' \
    --use_muco             1 \
    --jta_dir              '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/jta_dataset' \
    --use_jta              1 \
    --panoptic_dir         '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/panoptic' \
    --use_panoptic         0 \
    --protocol             1 \
    --pretrained_dir       '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/pretrained_models' \
    --resume               '' \
    --input_height         600 \
    --input_width          800 \
    --seq_max_gap          4 \
    --seq_min_gap          4 \
    --num_frames           4 \
    --num_future_frames    0 \
    --max_depth            15 \
    --batch_size           2 \
    --num_queries          60 \
    --num_kpts             15 \
    --set_cost_is_human    1 \ 
    --set_cost_root        1 \
    --set_cost_root_depth  1 \
    --set_cost_root_vis    1 \
    --set_cost_joint       1 \
    --set_cost_joint_depth 1 \
    --set_cost_joint_vis   1 \
    --is_human_loss_coef   1 \ 
    --root_loss_coef       5 \
    --root_depth_loss_coef 5 \
    --root_vis_loss_coef   1 \
    --joint_loss_coef      5 \
    --joint_depth_loss_coef 5 \
    --joint_vis_loss_coef  1 \
    --joint_disp_loss_coef 1 \
    --joint_disp_depth_loss_coef 1 \
    --heatmap_loss_coef    0.001 \
    --cont_loss_coef       0.1 \
    --eos_coef             0.25 \
    --epochs               40 \
    --lr_drop              30 \
    --lr                   0.0001 \
    --lr_backbone          0.00001 \
    --dropout              0.1 \
    --num_feature_levels   3 \
    --hidden_dim           384 \
    --nheads               8 \
    --enc_layers           6 \
    --dec_layers           6 \
    --dec_n_points         4 \
    --enc_n_points         4 \
    --use_pytorch_deform   0 \
```

Settings to train on JTA dataset (4 observed frames pose tracking + 2 future frames motion prediction)
```angular2html
python -u -m torch.distributed.launch --nproc_per_node=8 main.py \
    --output_dir           "$LOG_OUTDIR" \
    --dataset_file         'hybrid' \
    --posetrack_dir        '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/posetrack2018'\
    --use_posetrack        0 \
    --coco_dir             '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/coco' \
    --use_coco             0 \
    --muco_dir             '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/muco' \
    --use_muco             0 \
    --jta_dir              '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/jta_dataset' \
    --use_jta              1 \
    --panoptic_dir         '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/panoptic' \
    --use_panoptic         0 \
    --protocol             1 \
    --pretrained_dir       '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/pretrained_models' \
    --resume               '' \
    --input_height         540 \
    --input_width          960 \
    --seq_max_gap          4 \
    --seq_min_gap          4 \
    --num_frames           4 \
    --num_future_frames    2 \
    --max_depth            60 \
    --batch_size           2 \
    --num_queries          60 \
    --num_kpts             15 \
    --set_cost_is_human    1 \
    --set_cost_root        5 \
    --set_cost_root_depth  5 \
    --set_cost_root_vis    0.1 \
    --set_cost_joint       1 \
    --set_cost_joint_depth 1 \
    --set_cost_joint_vis   0.1 \
    --is_human_loss_coef   1 \
    --root_loss_coef       5 \
    --root_depth_loss_coef 5 \
    --root_vis_loss_coef   0.1 \
    --joint_loss_coef      5 \
    --joint_depth_loss_coef 5 \
    --joint_vis_loss_coef  0.1 \
    --joint_disp_loss_coef 1 \
    --joint_disp_depth_loss_coef 1 \
    --heatmap_loss_coef    0.001 \
    --cont_loss_coef       0.1 \
    --eos_coef             0.25 \
    --epochs               100 \
    --lr_drop              80 \
    --lr                   0.0001 \
    --lr_backbone          0.00001 \
    --dropout              0.1 \
    --num_feature_levels   3 \
    --hidden_dim           384 \
    --nheads               8 \
    --enc_layers           6 \
    --dec_layers           6 \
    --dec_n_points         4 \
    --enc_n_points         4 \
    --use_pytorch_deform   0 \
```


Settings to train on CMU-Panoptic dataset  (4 observed frames pose tracking + 2 future frames motion prediction)
```angular2html
python -u -m torch.distributed.launch --nproc_per_node=8 main.py \
    --output_dir           "$LOG_OUTDIR" \
    --dataset_file         'hybrid' \
    --posetrack_dir        '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/posetrack2018'\
    --use_posetrack        0 \
    --coco_dir             '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/coco' \
    --use_coco             0 \
    --muco_dir             '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/muco' \
    --use_muco             0 \
    --jta_dir              '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/jta_dataset' \
    --use_jta              0 \
    --panoptic_dir         '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/panoptic' \
    --use_panoptic         1 \
    --protocol             1 \
    --pretrained_dir       '/mnt/graphics_ssd/home/minhpvo/Internship/2021/shihaozou/pretrained_models' \
    --resume               '' \
    --input_height         540 \
    --input_width          960 \
    --seq_max_gap          10 \
    --seq_min_gap          10 \
    --num_frames           4 \
    --num_future_frames    2 \
    --max_depth            5 \
    --batch_size           2 \
    --num_queries          20 \
    --num_kpts             15 \
    --set_cost_is_human    1 \
    --set_cost_root        5 \
    --set_cost_root_depth  5 \
    --set_cost_root_vis    0.1 \
    --set_cost_joint       1 \
    --set_cost_joint_depth 1 \
    --set_cost_joint_vis   0.1 \
    --is_human_loss_coef   1 \
    --root_loss_coef       5 \
    --root_depth_loss_coef 5 \
    --root_vis_loss_coef   0.1 \
    --joint_loss_coef      5 \
    --joint_depth_loss_coef 5 \
    --joint_vis_loss_coef  0.1 \
    --joint_disp_loss_coef 1 \
    --joint_disp_depth_loss_coef 1 \
    --heatmap_loss_coef    0.001 \
    --cont_loss_coef       0.1 \
    --eos_coef             0.25 \
    --epochs               10 \
    --lr_drop              8 \
    --lr                   0.0001 \
    --lr_backbone          0.00001 \
    --dropout              0.1 \
    --num_feature_levels   3 \
    --hidden_dim           384 \
    --nheads               8 \
    --enc_layers           6 \
    --dec_layers           6 \
    --dec_n_points         4 \
    --enc_n_points         4 \
    --use_pytorch_deform   0 \
```

---
### Demos

![](demos/seq2_pose_tracking.gif)
![](demos/seq3_pose_tracking.gif)
![](demos/seq1_pose_tracking.gif)
