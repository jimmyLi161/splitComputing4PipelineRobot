from collections import OrderedDict

from bn.base import BottleneckBase
from bn.processor import get_bottleneck_processor
from bn.register import register_model_class, register_model_func
from torch import nn
from torchvision.models import resnet101, resnet152


@register_model_class
class BottleneckFORResNet(BottleneckBase):
    def __init__(self, bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None):
        modules = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=1)
        ]
        encoder = nn.Sequential(*modules[:bottleneck_idx])
        decoder = nn.Sequential(*modules[bottleneck_idx:])
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)

@register_model_class
class SplitResNet(nn.Sequential):
    def __init__(self, bottleneck, short_module_names, orginal_resnet):
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        short_module_set = set(short_module_names)
        for child_name, child_module in orginal_resnet.named_children():
            if child_name in short_module_set:
                if child_name == 'fc':
                    module_dict['flatten'] = nn.Flatten(1)
                module_dict[child_name] = child_module

        super().__init__(module_dict)

@register_model_func
def SplitResNet101(bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None,
                     short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['layer3', 'layer4', 'avgpool', 'fc']
    if compressor is not None:
        compressor = get_bottleneck_processor(compressor['name'], **compressor['params'])
    if decompressor is not None:
        decompressor = get_bottleneck_processor(decompressor['name'], **decompressor['params'])

    bottleneck = BottleneckFORResNet(bottleneck_channel, bottleneck_idx, compressor, decompressor)
    orginal_model = resnet101(**kwargs)
    return SplitResNet(bottleneck, short_module_names, orginal_model)


@register_model_func
def SplitResNet152(bottleneck_channel=12, bottleneck_idx=7, compressor=None, decompressor=None,
                     short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['layer3', 'layer4', 'avgpool', 'fc']

    if compressor is not None:
        compressor = get_bottleneck_processor(compressor['name'], **compressor['params'])
        
    if decompressor is not None:
        decompressor = get_bottleneck_processor(decompressor['name'], **decompressor['params'])

    bottleneck = BottleneckFORResNet(bottleneck_channel, bottleneck_idx, compressor, decompressor)
    orginal_model = resnet152(**kwargs)
    return SplitResNet(bottleneck, short_module_names, orginal_model)