import numpy as np
import cv2
from math import sqrt, ceil
from PIL import Image

import torch
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

from utils.file_utils import Tensor2PIL, PIL2Tensor
from utils.image_precossing import _add_batch_one


BOUNDARY_DIR = './boundaries'
LEVEL = {  # for style mixing
    'coarse': (0, 4),
    'middle': (4, 8),
    'fine'  : (8, 18)
}
M = torch.Tensor([[0.412453, 0.357580, 0.180423],
                [0.212671, 0.715160, 0.072169],
                [0.019334, 0.119193, 0.950227]])



def SR_loss(loss_function, down_type='bilinear', factor=8):     # mode默认为双线性插值
    def loss(x, gt):
        x = F.interpolate(x, scale_factor=1/factor, mode=down_type) # 下采样
        gt = F.interpolate(gt, scale_factor=1/factor, mode=down_type)
        return loss_function(x, gt)   # 返回CombinationLoss类的forward(self, x, gt)方法的返回值
                                          #  return self.l1_lambda * l1 + \
                                          #  self.l2_lambda * l2 + \
                                          #  self.vgg_lambda * vgg
    return loss


def upsample_images(image, factor=4, mode='nearest', size=256):
    """
    默认将256x256的图片放大到1024
    保证factor x size = 1024即可
    """
    down = F.interpolate(image, scale_factor=1/factor, mode=mode)
    up_nn = F.interpolate(down, scale_factor=factor, mode='nearest')
    up_bic = F.interpolate(down, scale_factor=factor, mode='bilinear')
    return up_nn, up_bic

# 降低分辨率, 但保持图像尺寸不变
def downsample_images(image, factor, mode):
    down = F.interpolate(image, scale_factor=1/factor, mode=mode)
    up_nn = F.interpolate(down, scale_factor=factor, mode='nearest')
    up_bic = F.interpolate(down, scale_factor=factor, mode='bilinear')
    return up_nn, up_bic


def convert_array_to_images(np_array):
  """Converts numpy array to images with data type `uint8`.

  This function assumes the input numpy array is with range [-1, 1], as well as
  with shape [batch_size, channel, height, width]. Here, `channel = 3` for color
  image and `channel = 1` for gray image.

  The return images are with data type `uint8`, lying in range [0, 255]. In
  addition, the return images are with shape [batch_size, height, width,
  channel]. NOTE: the channel order will be the same as input.

  Inputs:
    np_array: The numpy array to convert.

  Returns:
    The converted images.

  Raises:
    ValueError: If this input is with wrong shape.
  """
  input_shape = np_array.shape
  if len(input_shape) != 4 or input_shape[1] not in [1, 3]:
    raise ValueError('Input `np_array` should be with shape [batch_size, '
                     'channel, height, width], where channel equals to 1 or 3. '
                     'But {} is received!'.format(input_shape))

  images = (np_array + 1.0) * 127.5
  images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
  images = images.transpose(0, 2, 3, 1)
  return images


def style_mixing(source_A, source_B, level):
    """
    :param source_A: of size [1, 18, 512]
    :param source_B: of size [1, 18, 512]
    :param level:
    :return:
    """
    assert level in LEVEL.keys(), 'Please Check Your Mixing Level.'
    start, end = LEVEL[level]
    new_latent = torch.zeros_like(source_A)
    new_latent[:, :, :] = source_A
    new_latent[:, start: end, :] = source_B[:, start: end, :]
    return new_latent


