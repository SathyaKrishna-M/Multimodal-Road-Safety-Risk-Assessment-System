"""
Road Safety Segmentation Models

This module contains:
1. RoadSafetySegmentationModel: Custom U-Net with ResNet18/MobileNetV2 backbone.
2. PretrainedRoadSegmentationModel: Wrapper for DeepLabV3+ (COCO pretrained).
3. HybridRoadSafetyModel: Combines Pretrained (Road) + Custom (Pothole).
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.segmentation as segmentation

class UpBlock(nn.Module):
    """
    Upsampling block with skip connections for U-Net architecture.
    """
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle padding issues if dimensions don't match exactly
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                  diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class RoadSafetySegmentationModel(nn.Module):
    """
    Custom U-Net style segmentation model.
    Default backbone: ResNet18 (pretrained encoder).
    """
    def __init__(self, num_classes=3, backbone='resnet18'):
        super(RoadSafetySegmentationModel, self).__init__()
        
        self.backbone_name = backbone
        
        if backbone == 'resnet18':
            self.encoder = models.resnet18(pretrained=True)
            self.base_layers = list(self.encoder.children())
            
            # Encoder layers for skip connections
            self.layer0 = nn.Sequential(*self.base_layers[:3]) # size/2
            self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size/4
            self.layer2 = self.base_layers[5] # size/8
            self.layer3 = self.base_layers[6] # size/16
            self.layer4 = self.base_layers[7] # size/32
            
            # Decoder
            self.up1 = UpBlock(512, 256)
            self.up2 = UpBlock(256, 128)
            self.up3 = UpBlock(128, 64)
            self.up4 = UpBlock(64, 64)
            
            self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
            
        elif backbone == 'mobilenet_v2':
            self.encoder = models.mobilenet_v2(pretrained=True).features
            self.layer0 = self.encoder[:2]    # /2
            self.layer1 = self.encoder[2:4]   # /4
            self.layer2 = self.encoder[4:7]   # /8
            self.layer3 = self.encoder[7:14]  # /16
            self.layer4 = self.encoder[14:]   # /32
            
            self.up1 = UpBlock(1280, 96)
            self.up2 = UpBlock(96, 32)
            self.up3 = UpBlock(32, 24)
            self.up4 = UpBlock(24, 16)
            
            self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)
        else:
            raise ValueError("Backbone not supported. Choose 'resnet18' or 'mobilenet_v2'")

    def forward(self, x):
        # Encoder
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Decoder
        d1 = self.up1(x4, x3)
        d2 = self.up2(d1, x2)
        d3 = self.up3(d2, x1)
        d4 = self.up4(d3, x0)
        
        # Final prediction
        out = self.final_conv(d4)
        out = nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        return out


class PretrainedRoadSegmentationModel(nn.Module):
    """
    Wrapper around torchvision's DeepLabV3+ (ResNet101) pretrained on COCO.
    Maps COCO classes to our 3-class system.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = segmentation.deeplabv3_resnet101(pretrained=True)
        self.backbone.eval()
        self.num_classes = num_classes
        
    def forward(self, x):
        with torch.no_grad():
            coco_output = self.backbone(x)['out']
        return self._map_coco_to_road_safety(coco_output)
    
    def _map_coco_to_road_safety(self, coco_output):
        B, C, H, W = coco_output.shape
        device = coco_output.device
        coco_preds = torch.argmax(coco_output, dim=1)
        
        our_output = torch.zeros(B, self.num_classes, H, W, device=device)
        
        # COCO Road (0) -> Our Road (1)
        road_mask = (coco_preds == 0)
        
        # Default Background (0) high confidence
        our_output[:, 0, :, :] = 10.0
        
        # Road high confidence
        our_output[:, 1, :, :][road_mask] = 15.0
        our_output[:, 0, :, :][road_mask] = -10.0
        
        # Potholes (2) - Placeholder heuristic, usually handled by Hybrid model
        return our_output


class HybridRoadSafetyModel(nn.Module):
    """
    Combines Pretrained DeepLabV3+ (for Road) and Custom U-Net (for Pothole).
    """
    def __init__(self, trained_model_path=None):
        super().__init__()
        self.road_segmenter = PretrainedRoadSegmentationModel()
        self.pothole_detector = None
        
        if trained_model_path:
            self.pothole_detector = RoadSafetySegmentationModel(num_classes=3)
            # Load weights safely (CPU map)
            state_dict = torch.load(trained_model_path, map_location='cpu')
            self.pothole_detector.load_state_dict(state_dict)
            self.pothole_detector.eval()
    
    def forward(self, x):
        # 1. Road Segmentation (Pretrained)
        road_output = self.road_segmenter(x)
        
        if self.pothole_detector is None:
            return road_output
        
        # 2. Pothole Detection (Custom Trained)
        with torch.no_grad():
            pothole_output = self.pothole_detector(x)
        
        # 3. Decision-Level Combination
        return self._combine_outputs(road_output, pothole_output)
    
    def _combine_outputs(self, road_output, pothole_output):
        road_preds = torch.argmax(road_output, dim=1)
        pothole_preds = torch.argmax(pothole_output, dim=1)
        
        combined = road_output.clone()
        
        # Pothole only valid if it appears on a detected road
        # Or if confidence is extremely high (implicit via trained model)
        # Here we enforce: Pothole (from trained) AND Road (from pretrained)
        pothole_mask = (pothole_preds == 2) & (road_preds == 1)
        
        combined[:, 2, :, :][pothole_mask] = 25.0 # High confidence for pothole
        combined[:, 1, :, :][pothole_mask] = -10.0
        
        return combined
