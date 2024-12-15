# QEMaxViT-Unet
This repository is the official implementation of the paper QEMaxViT-Unet: A Query-Based MaxViT-Unet with Edge Enhancement for Scribble-Supervised Segmentation of Medical Images.

## Datasets

1. The ACDC dataset with mask annotations can be downloaded from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/). The scribble annotations of ACDC have been released in [ACDC scribbles](https://vios-s.github.io/multiscale-adversarial-attention-gates/data). 

2. The MSCMR dataset with mask annotations can be downloaded from [MSCMRseg](https://zmiclab.github.io/zxh/0/mscmrseg19/data.html). The scribble annotations of MSCMRseg have been released in [MSCMR_scribbles](https://github.com/BWGZK/CycleMix/tree/main/MSCMR_scribbles). 

3. The SUN-SEG dataset follow [S2ME](https://github.com/lofrienger/S2ME)

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

## Training




