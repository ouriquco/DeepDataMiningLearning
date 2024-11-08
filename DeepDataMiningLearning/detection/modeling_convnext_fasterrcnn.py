import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models.detection.rpn import AnchorGenerator


from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torch import nn
import torch.nn.functional as F

class FPN_PAN(nn.Module):
    def __init__(self, backbone, in_channels_list, out_channels):
        super().__init__()
        
        # FPN part
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
            
        # PAN part (bottom-up path)
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels_list) - 1):
            pan_block = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1),
                nn.LayerNorm([out_channels, None, None]),  # ConvNeXt uses LayerNorm
                nn.GELU()  # ConvNeXt uses GELU
            )
            self.pan_blocks.append(pan_block)
            
    def forward(self, x):
        # Bottom-up pathway (FPN)
        last_inner = self.inner_blocks[-1](x[-1])
        results = []
        results.append(self.layer_blocks[-1](last_inner))
        
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](x[idx])
            inner_top_down = F.interpolate(last_inner, size=inner_lateral.shape[-2:], mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
            
        # Top-down pathway (PAN)
        pan_results = [results[0]]
        last_pan = results[0]
        
        for idx in range(len(self.pan_blocks)):
            pan_feat = self.pan_blocks[idx](last_pan)
            last_pan = pan_feat + results[idx + 1]  # Skip connection
            pan_results.append(last_pan)
            
        return pan_results

def create_backbone_with_fpn_pan(trainable_layers=3):
    # Create ConvNeXt-Tiny backbone
    weights = ConvNeXt_Tiny_Weights.DEFAULT
    backbone = convnext_tiny(weights=weights).features
    
    # ConvNeXt-Tiny channel dimensions for each stage
    in_channels_list = [96, 192, 384, 768]  # These are the channel dimensions for each stage
    
    # Create FPN-PAN
    fpn_pan = FPN_PAN(
        backbone=backbone,
        in_channels_list=in_channels_list,
        out_channels=256  # You can adjust this based on your needs
    )
    
    return fpn_pan



class ConvNextFasterRCNN(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained_backbone: bool = False, trainable_backbone_layers: int = 3, **kwargs):
        super().__init__()
        
        # # Load ConvNeXT backbone
        # backbone = torchvision.models.convnext_tiny(
        #     pretrained=pretrained_backbone
        # ).features
        norm_layer = nn.LayerNorm
                # Get weights
        weights = ConvNeXt_Tiny_Weights.DEFAULT  # or ConvNeXt_Tiny_Weights.IMAGENET1K_V1

        # Create backbone
        backbone = convnext_tiny(weights=weights, norm_layer=norm_layer)
        # Set output channels
        backbone.out_channels = 768
        
        # Freeze layers if specified
        if trainable_backbone_layers < 0 or trainable_backbone_layers > 5:
            raise ValueError("Trainable backbone layers should be in the range [0,5]")
        
        layers_to_train = ['3', '2', '1', '0'][:trainable_backbone_layers]
        for name, parameter in backbone.named_parameters():
            parameter.requires_grad = False
            for layer_name in layers_to_train:
                if layer_name in name:
                    parameter.requires_grad = True
                    break
        
        # Define RPN anchor generator
        rpn_anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # Define RoI pooling
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )
        
        # Create the Faster R-CNN model
        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_pooler,
            **kwargs
        )
    
    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]], optional): ground-truth boxes present in the image
        
        Returns:
            result (list[Dict[str, Tensor]]): detection results
            losses (Dict[str, Tensor]): losses during training
        """
        return self.model(images, targets)
    
    def train(self, mode=True):
        """
        Sets the module in training mode.
        """
        self.training = mode
        self.model.train(mode)
        return self

# Example usage:
def create_model(num_classes=91, pretrained=False):
    model = ConvNextFasterRCNN(
        num_classes=num_classes,
        pretrained_backbone=pretrained,
        trainable_backbone_layers=3,
        # Additional FasterRCNN parameters can be added here
        min_size=800,
        max_size=1333,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
    )
    return model
    
def setup_training(model, learning_rate=0.005, momentum=0.9, weight_decay=0.0005):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    return optimizer, lr_scheduler