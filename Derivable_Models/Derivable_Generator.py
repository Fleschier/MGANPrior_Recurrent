import torch
import torch.nn as nn
import torch.nn.functional as F

# 为了解决import出错
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'..'))

# 使用绝对路径引入自己的包
from Derivable_Models.gan_utils import get_gan_model
