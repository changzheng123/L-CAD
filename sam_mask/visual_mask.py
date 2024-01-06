import os
import numpy as np
import cv2


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