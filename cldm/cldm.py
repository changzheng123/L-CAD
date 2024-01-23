from typing import Optional
import einops
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch as th
import torch.nn as nn
import sys
from cldm.logger import start_time

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler,DDIMSampler_withsam
from ldm.modules.diffusionmodules.model import ResnetBlock, make_attn, Normalize

import numpy as np
import os
from PIL import Image
from ldm.ptp import ptp_SD


def save_images(samples, batch, save_root, name_prompt=False):
    for i in range(samples.shape[0]):
            img_name = batch['name'][i]
            grid = samples[i].transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            if name_prompt:
                filename =img_name.split('.')[0]+'_'+batch['txt'][i][0:150]+'.png'
            else:
                filename =img_name.replace("jpg","png")
            path = os.path.join(save_root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)


class ControlLDM_cat(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        z, c, x = super().get_input(batch, self.first_stage_key, return_x=True, *args, **kwargs)
        control = batch[self.control_key] 
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w') 
        # print("control_shape",control.shape)
        print("using cldm get_input")
        control = control.to(memory_format=torch.contiguous_format).float()
        return z, dict(c_crossattn=[c], c_concat=[control]), x

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        if not self.model.diffusion_model.struct_attn:  
            cond_txt = torch.cat(cond['c_crossattn'], 1)
        else:
            cond_txt = cond['c_crossattn']
        # print('apply model cond_txt',cond_txt.shape)
        cond_hint = torch.cat(cond['c_concat'], 1)
        with torch.no_grad(): # 
            gray_z_last = self.first_stage_model.g_encoder(cond_hint)[-1]
            cond_hint = gray_z_last
                # print(cond_hint.shape)
                # sys.exit()
        # print('apply model', type(cond_txt))
        control = self.control_model(cond_hint) # Identity
   
        eps = self.model.diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
     
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N] # 
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        gray_z = self.first_stage_model.g_encoder(c_cat) # gray_z
        # log["reconstruction"] = self.decode_first_stage(z[:N], gray_z)
        log["control"] = c_cat
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        log["origin"]=x[:N] 

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples, gray_z)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg, gray_z)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.input_blocks[0].parameters()) # 
        params += list(self.model.diffusion_model.output_blocks.parameters())
        params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def test_step(self, batch, batch_idx) :
        if self.usesam:    
            self.test_step_sam(batch, batch_idx)
        else:
            self.test_step_org(batch, batch_idx)

    @torch.no_grad()
    def test_step_org(self, batch, batch_idx):

        multiColor_test = True
        ddim_steps=50 
        ddim_eta=0.0
        unconditional_guidance_scale=4.5  
        save_root = './image_log/test_%s_ug_%.1f'%(start_time,unconditional_guidance_scale)

        use_ddim = ddim_steps is not None
        z, c, x = self.get_input(batch,None)
        c_cat, c = c["c_concat"][0], c["c_crossattn"][0] # 
        N = z.shape[0]
        gray_z = self.first_stage_model.g_encoder(c_cat) 
        # print('gray_z.shape',gray_z[0].shape)
        
        uc_cross = self.get_unconditional_conditioning(N)
        uc_cat = c_cat  # torch.zeros_like(c_cat)
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
        samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                            batch_size=N, ddim=use_ddim,
                                            ddim_steps=ddim_steps, eta=ddim_eta,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc_full,
                                            )
        
 
        x_samples = self.decode_first_stage(samples_cfg, gray_z)
        x_samples = torch.clamp(x_samples, -1., 1.)
        x_samples = (x_samples + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

        if multiColor_test:
            save_images(x_samples, save_root = save_root, batch = batch, name_prompt=True)
        else:
            save_images(x_samples, save_root = save_root, batch = batch)

    # @torch.no_grad()
    def test_step_sam(self, batch, batch_idx):
        multiColor_test = False
        ddim_steps=25
        ddim_eta=0.0
        unconditional_guidance_scale=9.0 # 
        save_root = './image_log/test_%s'%start_time
        use_ddim = ddim_steps is not None

        control = batch[self.control_key] # 
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w') 
        N = control.shape[0]
        c_cat = control.to(memory_format=torch.contiguous_format).float()
        gray_z = self.first_stage_model.g_encoder(c_cat) 

        
        xc = batch['txt']
        c = self.get_learned_conditioning(xc)
        tokens = self.cond_stage_model.tokenizer.tokenize(xc[0]) # 测试batch_size = 1
        # print(tokens)

        sam_mask = batch['mask']
    
        uc_cross = self.get_unconditional_conditioning(N)
        uc_cat = c_cat  # torch.zeros_like(c_cat)
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

        ddim_sampler = DDIMSampler_withsam(self)
        cond={"c_concat": [c_cat], "c_crossattn": [c]}
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples_cfg, intermediates = ddim_sampler.sample(ddim_steps, b, shape, cond, eta=ddim_eta,
                                            unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=uc_full,verbose=False,
                                            use_attn_guidance=True, 
                                            sam_mask=sam_mask
                                            )
        
        x_samples = self.decode_first_stage(samples_cfg, gray_z)
        x_samples = torch.clamp(x_samples, -1., 1.)
        x_samples = (x_samples + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

        if multiColor_test:
            save_images(x_samples, save_root = save_root, batch = batch, name_prompt=True)
        else:
            save_images(x_samples, save_root = save_root, batch = batch,)
        
        
