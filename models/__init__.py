import os
import pathlib


from .head import replace_head
from .mnistnet import MnistNet
from .cifar_resnet import (resnet20,
                           resnet32,
                           resnet44,
                           resnet56,
                           resnet110,
                           resnet1202)

from .cifar_vgg import vgg_bn_drop, vgg_bn_drop_100
from .dfcnet import dfcnet, dfcnet_xavier, dfcnet_debug, cnn
from .mlp_tabular import mlpnet, mlpnet_xavier, mlpnet_debug, mlpnet_dropout
