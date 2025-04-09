from omegaconf import DictConfig

import torch.nn as nn

from model.sd_feature_extractor import SDFeatureExtractor
from model.sd_feature_extractor_prompt_learning import SDFeatureExtractorPromptLearning
from model.adm_feature_extractor import ADMFeatureExtractor

from propagation import label_propagation_feat_cosine_similarity, label_propagation_feat_euclidean, label_propagation_feat_l1, label_propagation_oracle

def get_dataset_cfg(cfg: DictConfig):
    dataset_name = cfg.dataset
    data_cfg = cfg.datasets[dataset_name]

    potential_overrides = [
        'image_directory',
        'mask_directory',
    ]

    for override in potential_overrides:
       
        if cfg[override] is not None:
            data_cfg[override] = cfg[override]

        if override in data_cfg:
            cfg[override] = data_cfg[override]

    return data_cfg


def get_model(cfg: DictConfig, oracle=False):
    model_name = cfg.model

    if model_name == 'sd1.5':
        model_id = "sd-legacy/stable-diffusion-v1-5"    
        model_class = SDFeatureExtractor
    elif model_name == 'sd1.4':
        model_id = "CompVis/stable-diffusion-v1-4"    
        model_class = SDFeatureExtractor
    elif model_name == 'sd1.3':
        model_id = "CompVis/stable-diffusion-v1-3"    
        model_class = SDFeatureExtractor
    elif model_name == 'sd1.2':
        model_id = "CompVis/stable-diffusion-v1-2"    
        model_class = SDFeatureExtractor
    elif model_name == 'sd1.1':
        model_id = "CompVis/stable-diffusion-v1-1"
        model_class = SDFeatureExtractor
    elif model_name == 'adm':
        model_id = 'ADM'
        model_class = ADMFeatureExtractor
    elif model_name == 'sd2.1':
        model_id = "stabilityai/stable-diffusion-2-1"
        model_class = SDFeatureExtractor
    else :
        raise AttributeError('Invalid model name')
    
    model = model_class(model_id)
    
    if oracle:
        propagation_func = label_propagation_oracle
    else:
        if cfg.propagation_params.func == 'cosine':
            propagation_func = label_propagation_feat_cosine_similarity
        elif cfg.propagation_params.func == 'euclidean':
            propagation_func = label_propagation_feat_euclidean
        elif cfg.propagation_params.func == 'l1':
            propagation_func = label_propagation_feat_l1
        else:
            raise AttributeError('Invalid propagation function')

    return model, propagation_func


def mse_sd_loss(pred_seg, target_seg, noise_pred, noise):
    loss = nn.MSELoss()(pred_seg, target_seg)
    loss += nn.MSELoss()(noise_pred, noise)
    return loss

def bce_sd_loss(pred_seg, target_seg, noise_pred, noise):
    loss = nn.BCELoss()(pred_seg, target_seg)
    loss += nn.MSELoss()(noise_pred, noise)
    return loss


def get_model_prompt_learning(cfg: DictConfig):
    model_name = cfg.model

    if model_name == 'sd1.5':
        model_id = "sd-legacy/stable-diffusion-v1-5"    
        model_class = SDFeatureExtractorPromptLearning
    elif model_name == 'sd1.4':
        model_id = "CompVis/stable-diffusion-v1-4"    
        model_class = SDFeatureExtractorPromptLearning
    elif model_name == 'sd1.3':
        model_id = "CompVis/stable-diffusion-v1-3"    
        model_class = SDFeatureExtractorPromptLearning
    elif model_name == 'sd1.2':
        model_id = "CompVis/stable-diffusion-v1-2"    
        model_class = SDFeatureExtractorPromptLearning
    elif model_name == 'sd1.1':
        model_id = "CompVis/stable-diffusion-v1-1"
        model_class = SDFeatureExtractorPromptLearning
    elif model_name == 'sd2.1':
        model_id = "stabilityai/stable-diffusion-2-1"
        model_class = SDFeatureExtractorPromptLearning
    else :
        raise AttributeError('Invalid model name')
    
    model = model_class(model_id)
    
    if cfg.propagation_params.func == 'cosine':
        propagation_func = label_propagation_feat_cosine_similarity
    elif cfg.propagation_params.func == 'euclidean':
        propagation_func = label_propagation_feat_euclidean
    elif cfg.propagation_params.func == 'l1':
        propagation_func = label_propagation_feat_l1
    else:
        raise AttributeError('Invalid propagation function')
    

    if cfg.prompt_learning_params.loss == 'mse':
        loss_func = nn.MSELoss()
    elif cfg.prompt_learning_params.loss == 'bce':
        loss_func = nn.BCELoss()    
    elif cfg.prompt_learning_params.loss == 'mse_sd':
        loss_func = mse_sd_loss
    elif cfg.prompt_learning_params.loss == 'bce_sd':
        loss_func = bce_sd_loss
    else:
        raise AttributeError('Invalid loss function')

    return model, propagation_func, loss_func