from share import *
import sys,argparse,os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from colorization_dataset import MyDataset
from cldm.model import create_model, load_state_dict
import time
import torch
import einops
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from ldm.ptp import ptp_SD, ptp_utils
from ldm.models.diffusion.ddim import DDIMSampler,DDIMSampler_withsam
from PIL import Image
import numpy as np

start_time = time.strftime('%Y-%m-%d-%H-%M-%S')

def save_images(samples, batch, save_root, prefix='', name_prompt=False):
    for i in range(samples.shape[0]):
            img_name = batch['name'][i]
            grid = samples[i].transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.cpu().numpy()
            grid = (grid * 255).astype(np.uint8)
            if name_prompt:
                filename = prefix + '_' + img_name.split('.')[0]+'_'+batch['txt'][i][0:150]+'.png'
            else:
                filename = prefix + '_' + img_name.replace("jpg","png")
            path = os.path.join(save_root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)


if __name__ == "__main__":

    resume_path = '.models/xxxx.ckpt'

    batch_size = 1 

    model = create_model('configs/cldm_sample.yaml').cpu()

    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model = model.cuda()

    model.usesam = True
    dataset = MyDataset(img_dir='example', caption_dir='example', split='test',use_sam=True) 
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=False)
    for batch_idx, batch in enumerate(dataloader):
        multiColor_test = True
        ddim_steps=50
        ddim_eta=0.0
        unconditional_guidance_scale = 5.0
        save_root = './image_log/test_%s'%start_time
        use_ddim = ddim_steps is not None

        control = batch[model.control_key] 
        control = control.to(model.device)
        control = einops.rearrange(control, 'b h w c -> b c h w') 
        N = control.shape[0]
        c_cat = control.to(memory_format=torch.contiguous_format).float()
        gray_z = model.first_stage_model.g_encoder(c_cat) 
        # print('gray_z.shape',gray_z[0].shape)
        
        xc = batch['txt']
        c = model.get_learned_conditioning(xc)
        tokens = model.cond_stage_model.tokenizer.tokenize(xc[0]) 
        batch_encoding = model.cond_stage_model.tokenizer(xc[0], truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        print(tokens)
        print(batch_encoding["input_ids"])

        split_idx = [] 
        for idx, token in enumerate(tokens):
            if token == ',</w>':
                split_idx.append(idx+1) 
        split_idx.append(idx+2) 

        sam_mask = batch['mask'] # bs=1 c h w 

        uc_cross = model.get_unconditional_conditioning(N)
        uc_cat = c_cat  # torch.zeros_like(c_cat)
        uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}

        ddim_sampler = DDIMSampler_withsam(model)
        cond={"c_concat": [c_cat], "c_crossattn": [c]}
        b, c, h, w = cond["c_concat"][0].shape
        shape = (model.channels, h // 8, w // 8)
        samples_cfg, intermediates = ddim_sampler.sample(ddim_steps, b, shape, cond, eta=ddim_eta,
                                            unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=uc_full,verbose=False,
                                            use_attn_guidance=True, # 使用attn_guidance
                                            sam_mask=sam_mask, split_id=split_idx,tokens=tokens
                                            )
        
        x_samples = model.decode_first_stage(samples_cfg, gray_z)
        x_samples = torch.clamp(x_samples, -1., 1.)
        x_samples = (x_samples + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
 
        save_images(x_samples, save_root = save_root, batch = batch, name_prompt=True)


        


