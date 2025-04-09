import os
from omegaconf import DictConfig
import hydra
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

from datasets.data_reader import DatasetReader
from utils.arg_utils import get_dataset_cfg, get_model_prompt_learning


def extract_features(cfg, model, frame, text_token, layer, return_hw=False):
    frame_ = frame.unsqueeze(0) if frame.ndim == 3 else frame
    cross_attn_feats, sample, noise, unet_ft = model.forward(
        frame_,
        text_token=text_token,
        t=cfg.propagation_params.t,
        ensemble_size=cfg.propagation_params.ensemble_size,        
        layer=cfg.propagation_params.layer,
    )
   
    feat_res, curr_h,curr_w = unet_ft.shape[-3:]       

    unet_ft = torch.permute(unet_ft, (1, 2, 0))
    unet_ft = unet_ft.view(curr_h * curr_w, feat_res) # hw,c
    
    if return_hw:
        return cross_attn_feats, sample, noise, curr_h, curr_w
    else:
        return cross_attn_feats, sample, noise


@hydra.main(version_base='1.3.2', config_path='config', config_name='eval_prompt_learning.yaml')
def learn_prompts(cfg: DictConfig):    

    print(f'------------Propagation params--------------------')
    print(f'Dataset: {cfg.dataset}')
    print(f'Model: {cfg.model}')
    print(f'Propagation function: {cfg.propagation_params.func}')
    print(f'Ensemble size: {cfg.propagation_params.ensemble_size}')
    print(f't: {cfg.propagation_params.t}')
    print(f'Layer: {cfg.propagation_params.layer}')
    
    print(f'------------Prompt learning params--------------------')
    print(f'lr: {cfg.prompt_learning_params.lr}')
    print(f'n_epochs: {cfg.prompt_learning_params.n_epochs}')
    print(f'loss: {cfg.prompt_learning_params.loss}')
    print(f'------------------------------------------------------')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """Load data"""
    dataset_cfg = get_dataset_cfg(cfg)
    dataset_reader = DatasetReader(cfg.datasets.data_root, dataset_cfg, video_reader_type='per_object', device=device)

    """Load model"""
    model, popagation_func, loss_func = get_model_prompt_learning(cfg)

   
    if cfg.exp_id is None:
        root_path = os.path.join('predictions', cfg.dataset, f'{cfg.model}_default_prompt_learning', 'tokens')
    else:
        root_path = os.path.join('predictions', cfg.dataset, f'{cfg.exp_id}', 'tokens')

    for video_reader in tqdm(dataset_reader, desc=f'Prompt learning {cfg.dataset} | Model {cfg.model}'):
        
        token_folder = os.path.join(root_path, video_reader.vid_name)
        os.makedirs(token_folder, exist_ok=True)

        f0_frame = video_reader.get_f0_frame()
        text_token = torch.full((1024,), 1e-4, requires_grad=True, device='cuda')

        with torch.no_grad():
            cross_attn_feats, sample, noise, curr_h, curr_w = extract_features(cfg, model, f0_frame, text_token, return_hw=True)

        gc.collect()
        torch.cuda.empty_cache()

        video_reader.set_curr_hw(curr_h, curr_w)
        first_seg = video_reader.first_seg
        
        """Learn a token for each object"""
        
        for object_id in range(1,video_reader.n_objects+1):
            
            text_token = torch.full((1024,), 1e-4, requires_grad=True, device='cuda')

            target_seg = first_seg.cuda(non_blocking=True)
            target_seg = target_seg[:,object_id]
            optimizer = torch.optim.AdamW([text_token] , lr=cfg.prompt_learning_params.lr)

            for _ in range(cfg.prompt_learning_params.n_epochs):
                optimizer.zero_grad()
                cross_attn_feats, noise_pred, noise = extract_features(cfg, model, f0_frame, text_token)
                
                pred_seg = cross_attn_feats.view(1,curr_h,curr_w)
                
                if cfg.prompt_learning_params.loss.__contains__('_sd'):
                    loss = loss_func(pred_seg, target_seg, noise_pred, noise)
                else:
                    loss = loss_func(pred_seg, target_seg)

                loss.backward()
                optimizer.step()
            
            text_token = text_token.detach().cpu()
            torch.save(text_token, os.path.join(token_folder, f'{object_id}.pt'))
            del text_token, loss, pred_seg, target_seg, noise_pred, noise
            torch.cuda.empty_cache()
            gc.collect()
            

if __name__ == '__main__':
    learn_prompts()