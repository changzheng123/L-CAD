# L-CAD: Language-based Colorization with Any-level Descriptions using Diffusion Priors

## Abstract
Language-based colorization produces plausible and visually pleasing colors under the guidance of user-friendly natural language descriptions. Previous methods implicitly assume that users provide comprehensive color descriptions for most of the objects in the image, which leads to suboptimal performance. In this paper, we propose a unified model to perform language-based colorization with anylevel descriptions. We leverage the pretrained cross-modality generative model for its robust language understanding and rich color priors to handle the inherent ambiguity of any-level descriptions. We further design modules to align with input conditions to preserve local spatial structures and prevent the ghosting effect. With the proposed novel sampling strategy, our model achieves instance-aware colorization in diverse and complex scenarios. Extensive experimental results demonstrate our advantages of effectively handling any-level descriptions and outperforming both language-based and automatic colorization methods.

<img src="teaser.png" align=center />


## Prerequisites
* Python 3.9
* PyTorch 1.12
* NVIDIA GPU + CUDA cuDNN

## Installation
Clone this repo: 
```
https://github.com/changzheng123/L-CAD.git
```
Install PyTorch and dependencies
```
http://pytorch.org
```
Install other python requirements
```
pip install -r requirement.txt
```



## Testing with pretrained model
```
python colorization_main.py 
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

## Citation
If you use this code for your research, please cite our papers [L-CAD: Language-based Colorization with Any-level Descriptions using Diffusion Priors](https://ci.idm.pku.edu.cn/Weng_CVPR23f.pdf)](https://ci.idm.pku.edu.cn/Weng_NeurIPS23.pdf)
```
@InProceedings{lcad,
  author = {Chang, Zheng and Weng, Shuchen and Zhang, Peixuan and Li, Yu and Li, Si and Shi, Boxin},
  title = {L-CAD: Language-based Colorization with Any-level Descriptions using Diffusion Priors},
  booktitle = {{NeurIPS}},
  year = {2023}
}
```
