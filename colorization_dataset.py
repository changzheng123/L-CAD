import json
import cv2
import numpy as np
from PIL import Image
import os
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from einops import rearrange


class MyDataset(Dataset):
    def __init__(self, img_dir, caption_dir=None, split='train',img_size=256, use_sam=False):

        assert split in ['train','val','test']
        self.split = split
        self.img_dir = os.path.join(img_dir,self.split+'2017')
        norm_mean = [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]
        self.norm = transforms.Normalize(norm_mean, norm_std)
        self.istest = False
        self.use_sam = use_sam
        self.img_size = img_size
        if split == 'train':
            caption_path = os.path.join(caption_dir,'selected_train.json')
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((img_size,img_size),scale=(0.8, 1.0), interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                ])
            self.caption_file = json.load(open(caption_path,'r'))
            self.keys = list(self.caption_file.keys())

        elif split == 'val':
            caption_path = os.path.join(caption_dir, 'selected_val.json')      
 
            self.transform = transforms.Compose([transforms.Resize((img_size,img_size)),
                                                transforms.ToTensor(),])
            self.caption_file = json.load(open(caption_path,'r'))
            self.keys = list(self.caption_file.keys())

        elif split == 'test':
            self.istest = True
            self.img_dir = 'example'
            if self.use_sam: 
                caption_path = os.path.join('sam_mask','pairs.json')
            else:    
                caption_path = os.path.join('example','test-pair-0.json')

                
            self.transform = transforms.Compose([# CenterCropLongEdge(),
                                            transforms.Resize((img_size, img_size)),
                                            transforms.ToTensor(),
                                            ])
            self.pairs = json.load(open(caption_path,'r'))
        
    def get_img(self, img_name):
        img_pth = os.path.join(self.img_dir, img_name)
        img = Image.open(img_pth).convert('RGB')
        img = self.transform(img)

        img_lab = rgb2lab(img)
      
        img_l = img_lab[[0,],:,:].repeat(3,1,1)
        img_ab = img_lab[1:,:,:]

        img = self.norm(img)

        img_l = rearrange(img_l,' c h w -> h w c')
        img = rearrange(img,' c h w -> h w c')
        img_ab = rearrange(img_ab,' c h w -> h w c')

        return img_l, img, img_ab
    
    def get_caption(self, key):
        # a list of indices for a sentence
        captions = self.caption_file[key]
        index = random.choice([i for i in range(len(captions))])
        cap = captions[index]
        return cap,index

    def get_mask(self, img_name):
        mask_dir = 'sam_mask/select_masks'
        mask_list = []
        mask_path = os.path.join(mask_dir,img_name.split('.')[0])
        for mask_name in sorted(os.listdir(mask_path)):
            mask = np.load(os.path.join(mask_path, mask_name))
            mask = mask.astype('float')
            mask = cv2.resize(mask,(self.img_size,self.img_size)) # 放缩一下mask
            mask = np.expand_dims(mask,axis=0)
            # print('mask.shape',mask.shape)
            # print(mask)
            mask_list.append(mask)      
        masks = np.concatenate(mask_list,axis=0)
        return masks
        

    def __len__(self):
        if not self.istest:    
            return len(self.keys)
        else:
            return len(self.pairs)

    def __getitem__(self, idx):
        if not self.istest:
            key = self.keys[idx]
            img_l, img, img_ab = self.get_img(key)
            cap, cap_idx = self.get_caption(key)
        else:
            key, cap = self.pairs[idx]
            img_l, img, img_ab = self.get_img(key)
        target = img
        prompt = cap
        source = img_l
        if not self.use_sam: 
            return dict(jpg=target, txt=prompt, hint=source, name=key)
        else:
            mask = self.get_mask(key)
            return dict(jpg=target, txt=prompt, hint=source, name=key, mask=mask)



def rgb2xyz(rgb): 

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[0,:,:]+.357580*rgb[1,:,:]+.180423*rgb[2,:,:]
    y = .212671*rgb[0,:,:]+.715160*rgb[1,:,:]+.072169*rgb[2,:,:]
    z = .019334*rgb[0,:,:]+.119193*rgb[1,:,:]+.950227*rgb[2,:,:]
    out = torch.cat((x[None,:,:],y[None,:,:],z[None,:,:]),dim=0)

    return out


def xyz2lab(xyz):
    sc = torch.Tensor((0.95047, 1., 1.08883))[:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[1,:,:]-16.
    # print("L",L)
    a = 500.*(xyz_int[0,:,:]-xyz_int[1,:,:])
    # print("a",a)
    b = 200.*(xyz_int[1,:,:]-xyz_int[2,:,:])
    # print("b",b)
    out = torch.cat((L[None,:,:],a[None,:,:],b[None,:,:]),dim=0)

    return out


def rgb2lab(rgb):
    lab = xyz2lab(rgb2xyz(rgb)) 
    l_rs = (lab[[0],:,:])/127.5 - 1 
    ab_rs = lab[1:,:,:]/110. 
    out = torch.cat((l_rs,ab_rs),dim=0)
    return out


def lab2rgb(lab_rs):
    l = lab_rs[:,[0],:,:]/2.*100. + 50.
    ab = lab_rs[:,1:,:,:]*110.
    lab = torch.cat((l,ab),dim=1)
    out = xyz2rgb(lab2xyz(lab))
    return out

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)

    out = out*sc

    return out

def xyz2rgb(xyz):

    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]

    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    rgb = torch.max(rgb,torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    return rgb