import torch
import torch.nn as nn
import torch.nn.functional as F
from .maxvit_unet import MaxViTBlock

class EdgeGuidanceModule(nn.Module):
    def __init__(self, in_channels1, in_channels2):
        super(EdgeGuidanceModule, self).__init__()
        
        # Convolution layers for E-Block1 feature map
        self.conv1_1x1 = nn.Conv2d(in_channels1, 64, kernel_size=1)
        self.conv1_3x3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Convolution layers for E-Block2 feature map after upsampling
        self.conv2_1x1 = nn.Conv2d(in_channels2, 64, kernel_size=1)
        self.conv2_3x3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Final convolution layer for edge map
        self.conv_final = nn.Conv2d(128, 1, kernel_size=1)
        
        self.edge_attention_block = nn.Sequential(*[
            MaxViTBlock(
                in_channels=128 if index == 0 else 192,
                out_channels=192,
                num_heads=16,
                grid_window_size=(8,8),
            )
            for index in range(2)
        ])
        
    def forward(self, e_block1, e_block2):
        # Upsample E-Block2 output to match E-Block1 output size
        e_block2_up = F.interpolate(e_block2, size=e_block1.shape[2:], mode='bilinear', align_corners=True)
        
        # Processing E-Block1 output
        x1 = self.conv1_1x1(e_block1)
        x1 = self.conv1_3x3(x1)
        
        # Processing upsampled E-Block2 output
        x2 = self.conv2_1x1(e_block2_up)
        x2 = self.conv2_3x3(x2)
        
        # Concatenate the two feature maps
        x = torch.cat((x1, x2), dim=1)
        
        attention_edge_features = self.edge_attention_block(x)
        # Apply final convolution to get edge map
        edge_map = self.conv_final(x)
        edge_map_upsampled = F.interpolate(edge_map, size=(256, 256), mode='bilinear', align_corners=True)
        return attention_edge_features, edge_map_upsampled

class QueryCombiner(nn.Module):
    def __init__(self, edge_feature_dim=192, query_dim=384, num_classes=4):
        super(QueryCombiner, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_edge = nn.Linear(edge_feature_dim, query_dim * num_classes)

    def forward(self, edge_features, learnable_queries):
        batch_size = edge_features.size(0)
        
        # Process edge features
        edge_features = self.adaptive_pool(edge_features).view(batch_size, -1)  # Reduce spatial dimensions
        edge_features_transformed = self.fc_edge(edge_features)  # Transform edge features to (query_dim * num_classes)
        edge_features_transformed = edge_features_transformed.view(batch_size, learnable_queries.size(1), -1)  # Reshape to (batch_size, num_classes, query_dim)

        # Combine with learnable queries using concatenation
        combined_queries = torch.cat((learnable_queries, edge_features_transformed), dim=2)

        return combined_queries