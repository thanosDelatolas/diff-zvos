import os
from pathlib import Path
import torch
from torchvision import transforms
main_path = Path(__file__).resolve().parent.parent.parent
print(f'main path: {main_path}')

import sys
sys.path.append(os.path.join(main_path, 'guided-diffusion'))
from guided_diffusion.script_util import create_model_and_diffusion
from guided_diffusion.nn import timestep_embedding


class ADMFeatureExtractor:
    def __init__(self, model_id='ADM'):
        model, diffusion = create_model_and_diffusion(
            image_size=256,
            class_cond=False,
            learn_sigma=True,
            num_channels=256,
            num_res_blocks=2,
            channel_mult="",
            num_heads=4,
            num_head_channels=64,
            num_heads_upsample=-1,
            attention_resolutions="32,16,8",
            dropout=0.0,
            diffusion_steps=1000,
            noise_schedule='linear',
            timestep_respacing='',
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
            use_checkpoint=False,
            use_scale_shift_norm=True,
            resblock_updown=True,
            use_fp16=False,
            use_new_attention_order=False,
        )
        model_path = os.path.join(main_path, 'guided-diffusion/models/256x256_diffusion_uncond.pt')
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model = model.eval().cuda()
        self.diffusion = diffusion

        self.adm_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    @torch.no_grad()
    def forward(self, img_tensor,
        t=101,
        ensemble_size=8,
        layer=None
        ):
        model = self.model
        diffusion = self.diffusion

        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda() # ensem, c, h, w
        t = torch.ones((img_tensor.shape[0],), device='cuda', dtype=torch.int64) * t
        x_t = diffusion.q_sample(img_tensor, t, noise=None)

        
        # get layer-wise features
        hs = []
        emb = model.time_embed(timestep_embedding(t, model.model_channels))
        h = x_t.type(model.dtype)
        for module in model.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = model.middle_block(h, emb)

        out_features = []
        for i, module in enumerate(model.output_blocks):
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)

            #if i == up_ft_index:
            ft = h.mean(0, keepdim=True).detach()
            #print(ft.shape)
            #return ft
            out_features.append(ft)
        #breakpoint()

        # for ii, ft in enumerate(out_features):
        #     print(f'layer {ii}: {ft.shape}')
        # breakpoint()
        
        # for i, ft in enumerate(out_features):
        #     print(f'layer {i}: {ft.shape}')
        # breakpoint()
        if layer is not None:
            return out_features[layer].mean(0, keepdim=True)
        else:
            return out_features[:8] # get first 8 only 18features


