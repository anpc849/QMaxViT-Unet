
# QMaxViT-Unet+
This repository is the official implementation of the paper QMaxViT-Unet+: A Query-Based MaxViT-Unet with Edge Enhancement for Scribble-Supervised Segmentation of Medical Images.

🚀 **This project is currently under active development. Please stay tuned for updates!**

## Datasets

1. The ACDC dataset with mask annotations can be downloaded from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/). The scribble annotations of ACDC have been released in [ACDC scribbles](https://vios-s.github.io/multiscale-adversarial-attention-gates/data). 

2. The MSCMR dataset with mask annotations can be downloaded from [MSCMRseg](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html). The scribble annotations of MSCMRseg have been released in [MSCMR_scribbles](https://github.com/BWGZK/CycleMix/tree/main/MSCMR_scribbles). 

3. The SUN-SEG dataset can be obtained by following the paper [S2ME](https://github.com/lofrienger/S2ME?tab=readme-ov-file#usage).

4. The BUSI dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset). The Scribble-BUSI dataset from our paper can be downloaded [here](https://www.kaggle.com/datasets/anhoangvo/scribble-busi).

## Related Code and Pretrained Weights

To ensure proper functionality, you need to download the related code and pretrained weights.

Run the following commands to clone the necessary repository:

```python
%cd QMaxViT-Unet/model
!git clone https://github.com/huggingface/pytorch-image-models
%cd QMaxViT-Unet
```

Download the pretrained weights for MaxViT Blocks from the link below:

[MaxViT Pretrained Weights](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights-maxx/maxxvit_rmlp_small_rw_256_sw-37e217ff.pth)

This file is used in 
```python
from model.qemaxvit_unet import QEMaxViT_Unet
model = QEMaxViT_Unet(num_classes=4, backbone_pretrained_pth="./maxxvit_rmlp_small_rw_256_sw-37e217ff.pth")
```

## Usage
### Generate edge masks
```bash
%cd dataloaders
git clone https://github.com/ZhouCX117/UAED 
gdown 1nv2_TZRyiQh5oU9TnGMzu313OrspD2l5  ## check point of edge detector
%cd UAED
```
```python
class Args:
    def __init__(self):
        self.distribution = None

args = Args()
args.distribution = 'gs'

from model.sigma_logit_unetpp import Mymodel
import torch
from ..utils import generate_edge_mask
edge_detector = Mymodel(in_channels=3, args=args)
checkpoint = torch.load("../epoch-19-checkpoint.pth")
edge_detector.load_state_dict(checkpoint['state_dict'])
edge_detector.cuda()
edge_detector.eval()
# old_path_data is the folder contains h5 files slices like /MSCMR_training_slices
# new_path_data is the new folder contains h5 files, each file includes image, label(scribble) and edge mask.
generate_edge_mask(old_path_data, new_path_data, edge_detector)
```

### Notebooks
We provide several Jupyter notebooks to make it easy to explore our method. These notebooks are designed to simplify understanding and implementation of our approach:

1.  **`train_notebook.ipynb`**  
    This is the most important notebook, offering a comprehensive guide for loading the dataset, creating the dataloader, defining the model, training, evaluation, and inference. It allows you to easily replace or modify each part, while keeping the core code intact, to perform training, evaluation, and inference seamlessly.
    
2.  **`scribble_busi_dataset_generate.ipynb`**  
    This notebook demonstrates how to automatically generate scribble masks using the code provided in [WSL4MIS](https://github.com/HiLab-git/WSL4MIS/blob/main/code/scribbles_generator.py).
    
3.  **`load_scribble_busi_dataset.ipynb`**  
    This notebook shows how to load the dataset and plot sample images from the Scribble-BUSI dataset. You can use it as a replacement in `train_notebook.ipynb`. With slight modifications (e.g., setting `num_classes=2`), it will work seamlessly with the training process.

4. For training on the SUN-SEG dataset, you should follow the code provided in the paper [S2ME](https://github.com/lofrienger/S2ME/tree/main). We also plan to publish our own code for this in the near future.
    

These notebooks provide a modular framework, making it easier to adapt and experiment with different components of our method.

## Citation
```bibtex
@article{nguyen2025qmaxvit,
  title={QMaxViT-Unet+: A query-based MaxViT-Unet with edge enhancement for scribble-supervised segmentation of medical images},
  author={Nguyen-Tat, Thien B and Vo, Hoang-An and Dang, Phuoc-Sang},
  journal={Computers in Biology and Medicine},
  volume={187},
  pages={109762},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgement
Some of the codes are borrowed/refer from below repositories:
- [WSL4MIS](https://github.com/HiLab-git/WSL4MIS)
- [S2ME](https://github.com/lofrienger/S2ME/tree/main)
- [MaxViT-UNet](https://github.com/PRLAB21/MaxViT-UNet)
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [SparseInst](https://github.com/hustvl/SparseInst)




