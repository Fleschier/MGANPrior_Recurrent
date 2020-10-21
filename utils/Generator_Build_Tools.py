"""Helper functions for generator building"""

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/'+'..'))

from GAN.Model_Settings import MODEL_POOL
from GAN.pggan_generator import PGGANGenerator
from GAN.stylegan_generator import StyleGANGenerator
from GAN.stylegan2_generator import StyleGAN2Generator

__all__ = ['build_generator']


def build_generator(model_name, logger=None):
    # 通过model的名称来建立生成器模块
    if not model_name in MODEL_POOL:    # 如果给的model名称不存在
        raise ValueError(f'Model {model_name} is not registered in MODEL_POOL or not exists!')

    # gan的类型
    gan_type = MODEL_POOL[model_name]['gan_type']

    if(gan_type == 'pggan'):
        return PGGANGenerator(model_name, logger=logger)
    elif(gan_type == 'stylegan'):
        return StyleGANGenerator(model_name,logger=logger)
    elif(gan_type == 'stylegan2'):
        return StyleGAN2Generator(model_name, logger=logger)
    else:
        raise NotImplementedError(f'不支持的GAN类型! {gan_type}')









