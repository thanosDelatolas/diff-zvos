import os
from omegaconf import DictConfig
import hydra
import queue
import copy
from tqdm import tqdm


import torch
import torch.nn.functional as F

from datasets.data_reader import DatasetReader
from utils.arg_utils import get_dataset_cfg, get_model
from utils.save_utils import imwrite_indexed, save_seg, make_zip
from propagation import norm_mask


@torch.no_grad()
def extract_features(cfg, model, frame, return_hw=False):
    frame_ = frame.unsqueeze(0) if frame.ndim == 3 else frame
    unet_ft = model.forward(frame_,
        t=cfg.propagation_params.t,
        ensemble_size=cfg.propagation_params.ensemble_size,
        layer=cfg.propagation_params.layer
    ).squeeze()

    feat_res, curr_h,curr_w = unet_ft.shape[-3:]       

    unet_ft = torch.permute(unet_ft, (1, 2, 0))
    unet_ft = unet_ft.view(curr_h * curr_w, feat_res) # hw,c
    
    if return_hw:
        return unet_ft, curr_h, curr_w
    else:
        return unet_ft

@torch.no_grad()
@hydra.main(version_base='1.3.2', config_path='config', config_name='eval.yaml')
def eval_vos(cfg: DictConfig):    

    print(f'------------ Propagation params --------------------')
    print(f'Dataset: {cfg.dataset}')
    print(f'Model: {cfg.model}')
    print(f'Propagation function: {cfg.propagation_params.func}')
    print(f'Ensemble size: {cfg.propagation_params.ensemble_size}')
    print(f'Layer: {cfg.propagation_params.layer}')
    print(f't: {cfg.propagation_params.t}')
    print(f'-----------------------------------------------------')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """Load data"""
    dataset_cfg = get_dataset_cfg(cfg)
    dataset_reader = DatasetReader(cfg.datasets.data_root, dataset_cfg, video_reader_type='default', device=device)

    """Load model"""
    model, popagation_func = get_model(cfg)

    if cfg.exp_id is None:
        root_path = os.path.join('predictions', cfg.dataset, f'{cfg.model}_default', 'Annotations')
    else:
        root_path = os.path.join('predictions', cfg.dataset, f'{cfg.exp_id}', 'Annotations')

    for video_reader in tqdm(dataset_reader, desc=f'Dataset {cfg.dataset} | Model {cfg.model}'):
        
        video_folder = os.path.join(root_path, video_reader.vid_name)
        os.makedirs(video_folder, exist_ok=True)

        f0_frame = video_reader.get_f0_frame()
        frame1_feat, curr_h, curr_w = extract_features(cfg, model, f0_frame, return_hw=True)
        frame1_feat = frame1_feat.T #  dim x h*w

        video_reader.set_curr_hw(curr_h, curr_w)
        first_seg = video_reader.first_seg
        
        out_path = os.path.join(video_folder, "00000.png")
        imwrite_indexed(out_path, video_reader.first_seg_ori, video_reader.color_palette)
        
        mask_neighborhood = None
        que = queue.Queue(cfg.propagation_params.n_last_frames)

        for idx, frame_tar in enumerate(video_reader):
            idx += 1
            feat_tar = extract_features(cfg, model, frame_tar, return_hw=False)
            used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
            used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]

            frame_tar_avg, feat_tar, mask_neighborhood = popagation_func(cfg, feat_tar, curr_h, curr_w, used_frame_feats, used_segs, mask_neighborhood)

            if que.qsize() == cfg.propagation_params.n_last_frames:
                que.get()

            seg = copy.deepcopy(frame_tar_avg)
            que.put([feat_tar, seg])

            if video_reader.is_mose:
                frame_tar_avg = F.interpolate(frame_tar_avg, (video_reader.ori_h, video_reader.ori_w), mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
            else :
                frame_tar_avg = F.interpolate(frame_tar_avg, scale_factor=video_reader.scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=False)[0]
            frame_tar_avg = norm_mask(frame_tar_avg)
            _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

            save_seg(frame_tar_seg, video_reader.ori_h, video_reader.ori_w, video_reader.frames[idx].replace(".jpg", ".png"), 
                video_folder, video_reader.color_palette
            )
    
    make_zip(cfg.dataset, root_path)


if __name__ == '__main__':
    eval_vos()