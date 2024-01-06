from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
import os


img_list =['09eacd04461a94ac.jpg',
           '5b68696f625d1572.jpg',
           'vg_2377448.jpg',
           'openimages_d273dd1b7d814d98.jpg',
           'openimages_c650143cc9a47340.jpg',
           '2f75c4549b695351.jpg',
           'openimages_1b8d52cb603f71ad.jpg',
           '000000084477.jpg']
for img_name in img_list:
    sam = sam_model_registry["default"](checkpoint="../models/sam_vit_h_4b8939.pth").cuda()
    mask_generator = SamAutomaticMaskGenerator(sam)
    img = np.array(Image.open(os.path.join('../example/',img_name))) #图片路径
    masks = mask_generator.generate(img)
    print(len(masks))
    print(masks[0].keys())
    for i in range(len(masks)):
        img_mask = img*np.expand_dims(masks[i]['segmentation'],axis=-1)
        Image.fromarray(img_mask).save('./seg_img/%s_mask_%d.png'%(img_name.split('.')[0],i))
        np.save('./masks/%s_%d.npy'%(img_name.split('.')[0], i),masks[i]['segmentation'])
