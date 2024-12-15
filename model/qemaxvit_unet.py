import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision

from .maxxvit_4out import maxxvit_rmlp_small_rw_256 as maxxvit_rmlp_small_rw_256_4out
from .maxvit_unet import MaxViTDecoder
from .twowaytrans import PositionEmbeddingSine, TwoWayTransformer, MLP
from .ppm_fpn import PyramidPoolingModuleFPN
from .edge_enhancer import EdgeGuidanceModule, QueryCombiner
from torch.nn import functional as F


class QEMaxViT_Unet(nn.Module):
    def __init__(self, num_classes, backbone_pretrained_pth):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = self.load_pretrained_weights(backbone_pretrained_pth)
        
        self.decoder = MaxViTDecoder(
            in_channels=(96, 192, 384, 768),
            depths=(2, 2, 2),
            grid_window_size=(8, 8),
            attn_drop=0.2,
            drop=0.2,
            drop_path=0.2,
            debug=True,
            num_classes=num_classes,
        )

        self.conv_3channels = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        self.queries = nn.Embedding(num_classes, 384)
        
        N_steps = 768 // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        self.transformer = TwoWayTransformer(
            depth=2,
            embedding_dim=768,
            num_heads=8,
            mlp_dim=512,
            activation=nn.GELU,
            attention_downsample_rate=2,
        )
        
        self.edge_module = EdgeGuidanceModule(in_channels1=96, in_channels2=192)
        
        self.enhance_queries = QueryCombiner(192,384,num_classes)
        
    
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(768, 768, 768 // 8, 3)
                for i in range(num_classes)
            ]
        )
        
        self.aug_module = PyramidPoolingModuleFPN(convs_dim=384, mask_dim=None, norm="GN")
    
    def load_pretrained_weights(self, pretrained_pth):
        backbone = maxxvit_rmlp_small_rw_256_4out()
        print('Loading:', pretrained_pth)
        state_dict = torch.load(pretrained_pth)
        #state_dict = torch.load('./pretrained_pth/maxvit/trained_backbone.pth')
        backbone.load_state_dict(state_dict, strict=False)
        print('Pretrained weights loaded.')
        return backbone
        
    def forward(self, x):
        
        x_size = x.size()
        bs = x_size[0]
        if x.size()[1] == 1:
            x = self.conv_3channels(x)
            
        ## encode features
        x1,x2,x3,x4 = self.backbone(x)
        
        ## edge guidance
        attention_edge_features, edge_map = self.edge_module(x1,x2)

        ## Mask Transformer
        img_pe = self.pe_layer(x4)
        queries = self.queries.weight.unsqueeze(0).repeat(x_size[0], 1 , 1)
        enhance_queries = self.enhance_queries(attention_edge_features, queries)

        updated_queries, updated_features = self.transformer(x4, img_pe, enhance_queries)
        updated_features = updated_features.transpose(1, 2).reshape(bs, 768, 8, 8)
    
        ## main decoder
        output_main_seg = self.decoder([x1,x2,x3,updated_features], attention_edge_features)
        
        ## aux decoder
        output_aux_seg = self.aug_module([x2,x3,x4])
        
        #upscaled_embedding = self.output_upscaling(updated_features)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_classes):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](updated_queries[:, i, :]))  
        hyper_in = torch.stack(hyper_in_list, dim=1)
        #print(upscaled_embedding.shape, hyper_in.shape)
        
        #b, c, h, w = upscaled_embedding.shape
        b,c,h,w = output_aux_seg.shape
        output_aux_seg = (hyper_in @ output_aux_seg.view(b, c, h * w)).view(b, -1, h, w)
        
        return output_main_seg, output_aux_seg, edge_map