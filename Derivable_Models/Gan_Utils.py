import numpy as np
import os

import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

# 使用绝对路径
from GAN.Model_Settings import MODEL_POOL
from GAN.pggan_generator import PGGANGenerator
from GAN.stylegan_generator import StyleGANGenerator
from GAN.stylegan2_generator import StyleGAN2Generator


PGGAN_Inter_Output_Layer_256 = [-1, 17, 14, 11, 8, 5, 2]
PGGAN_Inter_Output_Layer_1024 = [-1, 23, 20, 17, 14, 11, 8, 5, 2]

# 生成器
def build_generator(model_name, logger=None):
  """Builds generator module by model name."""
  if not model_name in MODEL_POOL:
    raise ValueError(f'Model `{model_name}` is not registered in '
                     f'`MODEL_POOL` in `model_settings.py`!')

  gan_type = MODEL_POOL[model_name]['gan_type']

  if gan_type == 'pggan':
    return PGGANGenerator(model_name, logger=logger)
  if gan_type == 'stylegan':
    return StyleGANGenerator(model_name, logger=logger)
  if gan_type == 'stylegan2':
    return StyleGAN2Generator(model_name, logger=logger)
  raise NotImplementedError(f'Unsupported GAN type `{gan_type}`!')



def standard_z_sample(size, depth, device=None):
    '''
    Generate a standard set of random Z as a (size, z_dimension) tensor.
    With the same random seed, it always returns the same z (e.g.,
    the first one is always the same regardless of the size.)
    '''
    # Use numpy RandomState since it can be done deterministically
    # without affecting global state
    rng = np.random.RandomState(None)
    result = torch.from_numpy(rng.standard_normal(size * depth).reshape(size, depth)).float()
    if device is not None:
        result = result.to(device)
    return result


def get_gan_model(model_name):
    """
    :param model_name: Please refer `GAN_MODELS`
    :return: gan_model(nn.Module or nn.Sequential)
    """
    gan = build_generator(model_name)
    if model_name.startswith('pggan'):
        gan_list = list(gan.net.children())
        remove_index = PGGAN_Inter_Output_Layer_1024 if model_name == 'pggan_celebahq' else PGGAN_Inter_Output_Layer_256
        for output_index in remove_index:
            gan_list.pop(output_index)    # 去除一些层
        return nn.Sequential(*gan_list)   # *表示可以接收多个参数
    elif model_name.startswith('style'):
        return gan
