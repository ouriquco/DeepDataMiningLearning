from typing import Callable, Dict, List, Optional, Union
import torch
from torch import nn, Tensor
import torchvision
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from backbone_convnext import ConvNextEncoder
from torchvision.models import resnet #, resnet50, ResNet50_Weights
from torchvision.models import get_model, get_model_weights, get_weight, list_models

def create_convnext_fpnpan(variant='tiny'):
    if variant == 'tiny':
        return CustomBackboneWithFPNPAN(
            in_channels=3,
            out_channels=256,
            depths=[3, 3, 9, 3],
            widths=[96, 192, 384, 768]
        )
    elif variant == 'base':
        return CustomBackboneWithFPNPAN(
            in_channels=3,
            out_channels=256,
            depths=[3, 3, 27, 3],
            widths=[128, 256, 512, 1024]
        )
    elif variant == 'large':
        return CustomBackboneWithFPNPAN(
            in_channels=3,
            out_channels=256,
            depths=[3, 3, 27, 3],
            widths=[192, 384, 768, 1536]
        )

class CustomBackboneWithFPNPAN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        depths: List[int] = [3, 3, 9, 3],
        widths: List[int] = [96, 192, 384, 768],
        drop_p: float = 0.1,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.body = ConvNextEncoder(
            in_channels=in_channels,
            stem_features=widths[0],  
            depths=depths,
            widths=widths,
            drop_p=drop_p
        )
        
        self.features = {}
        for i, stage in enumerate(self.body.stages):
            stage.register_forward_hook(self._get_features_hook(f"stage_{i}"))
        
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=widths,  
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        
        self.pan = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                for _ in range(len(widths))
            ]
        )
        
        self.out_channels = out_channels

    def _get_features_hook(self, name: str):
        def hook(module, input, output):
            self.features[name] = output
        return hook

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        self.features.clear()
        
        _ = self.body(x)
        
        x = {str(i): self.features[f"stage_{i}"] for i in range(len(self.body.stages))}
        
        fpn_outs = self.fpn(x)
        
        # PAN forward pass
        pan_outs = {}
        prev_feature = None
        for idx, (name, feature) in enumerate(sorted(fpn_outs.items())):
            if prev_feature is not None:
                target_size = (feature.shape[2], feature.shape[3])
                upsampled = nn.functional.interpolate(
                    prev_feature, 
                    size=target_size,
                    mode="nearest"
                )
                feature = feature + upsampled
            pan_outs[name] = feature
            if idx < len(self.pan):
                prev_feature = self.pan[idx](feature)
        
        return pan_outs


class BackboneWithFPNAndPAN(nn.Module):
    def __init__(
        self,
        model_name: str = 'tiny',
        trainable_layers: int = 5,
        out_channels: int = 256,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        # model_name == 'convnext_large'
        if model_name == 'base':
            backbone = torchvision.models.convnext_base(weights="DEFAULT").features
            in_channels_list = [128, 256, 512, 1024]  
        elif model_name == 'large':
            backbone = torchvision.models.convnext_large(weights="DEFAULT").features
            in_channels_list = [192, 384, 768, 1536]  
        else:  # convnext_tiny
            backbone = torchvision.models.convnext_tiny(weights="DEFAULT").features
            in_channels_list = [96, 192, 384, 768]  


        # Set trainable layers
        layers_to_train = ["3", "2", "1", "0"][:trainable_layers]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        # ConvNeXt uses numbered stages instead of 'layerX'
        return_layers = {
            "1": "0",  # stride 4
            "3": "1",  # stride 8
            "5": "2",  # stride 16
            "7": "3"   # stride 32
        }
        
        self.body = torchvision.models._utils.IntermediateLayerGetter(
            backbone, 
            return_layers=return_layers
        )
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        
        self.pan = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                for _ in range(len(in_channels_list))
            ]
        )
        
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # FPN forward pass
        x = self.body(x)
        fpn_outs = self.fpn(x)
        
        # PAN forward pass
        pan_outs = {}
        prev_feature = None
        
        # Process features in order from highest to lowest resolution
        for idx, (name, feature) in enumerate(sorted(fpn_outs.items())):
            if prev_feature is not None:
                target_size = (feature.shape[2], feature.shape[3])
                upsampled = nn.functional.interpolate(
                    prev_feature, 
                    size=target_size,
                    mode="nearest"
                )
                feature = feature + upsampled
            pan_outs[name] = feature
            if idx < len(self.pan):
                prev_feature = self.pan[idx](feature)
        
        return pan_outs