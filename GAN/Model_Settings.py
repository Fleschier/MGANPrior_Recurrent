# python 3.7
"""Contains basic configurations for models used in this project.

在使用之前需要先到下面三个官方库中下载已经训练好的模型,即pkl文件,保存在如下路径中
`pretrain/tensorflow`.

PGGAN: https://github.com/tkarras/progressive_growing_of_gans
StyleGAN: https://github.com/NVlabs/stylegan
StyleGAN2: https://github.com/NVlabs/stylegan2

如果要加入自己的训练模型,需要将信息录入此文件
"""

import os

BASE_DIR = os.path.dirname(os.path.relpath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, 'pretrain')
PTH_MODEL_DIR = 'pytorch'
TF_MODEL_DIR = 'tensorflow'

if not os.path.exists(os.path.join(MODEL_DIR, PTH_MODEL_DIR)):
  os.makedirs(os.path.join(MODEL_DIR, PTH_MODEL_DIR))

# pylint: disable=line-too-long
MODEL_POOL = {
    # PGGAN Official.
    'pggan_celebahq': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_celebahq1024_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-celebahq-1024x1024.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'celebahq',
        'z_space_dim': 512,
        'resolution': 1024,
        'fused_scale': False,
    },
    'pggan_bedroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_bedroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-bedroom-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-bedroom',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_livingroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_livingroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-livingroom-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-livingroom',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_diningroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_diningroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-dining_room-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-diningroom',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_kitchen': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_kitchen256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-kitchen-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-kitchen',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_churchoutdoor': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_churchoutdoor256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-churchoutdoor-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-churchoutdoor',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_tower': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_tower256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-tower-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-tower',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_bridge': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_bridge256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-bridge-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-bridge',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_restaurant': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_restaurant256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-restaurant-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-restaurant',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_classroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_classroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-classroom-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-classroom',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_conferenceroom': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_conferenceroom256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-conferenceroom-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-conferenceroom',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_person': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_person256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-person-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-person',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_cat': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_cat256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-cat-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-cat',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_dog': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_dog256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-dog-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-dog',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_bird': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_bird256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-bird-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-bird',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_horse': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_horse256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-horse-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-horse',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_sheep': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_sheep256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-sheep-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-sheep',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_cow': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_cow256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-cow-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-cow',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_car': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_car256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-car-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-car',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_bicycle': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_bicycle256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-bicycle-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-bicycle',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_motorbike': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_motorbike256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-motorbike-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-motorbike',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_bus': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_bus256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-bus-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-bus',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_train': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_train256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-train-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-train',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_boat': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_boat256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-boat-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-boat',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_airplane': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_airplane256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-airplane-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-airplane',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_bottle': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_bottle256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-bottle-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-bottle',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_chair': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_chair256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-chair-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-chair',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_pottedplant': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_pottedplant256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-pottedplant-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-pottedplant',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_tvmonitor': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_tvmonitor256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-tvmonitor-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-tvmonitor',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_diningtable': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_diningtable256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-diningtable-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-diningtable',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
    'pggan_sofa': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'pggan_sofa256_generator.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'karras2018iclr-lsun-sofa-256x256.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'pggan_tf_official'),
        'gan_type': 'pggan',
        'dataset_name': 'lsun-sofa',
        'z_space_dim': 512,
        'resolution': 256,
        'fused_scale': False,
    },
}
# pylint: enable=line-too-long

# # Settings for StyleGAN.
# # 截断
# STYLEGAN_TRUNCATION_PSI = 0.7  # 1.0 means no truncation
# STYLEGAN_TRUNCATION_LAYERS = 8  # 0 means no truncation
# STYLEGAN_RANDOMIZE_NOISE = False

# # Settings for StyleGAN2.
# STYLEGAN2_TRUNCATION_PSI = 0.5  # 1.0 means no truncation
# STYLEGAN2_TRUNCATION_LAYERS = 18  # 0 means no truncation
# STYLEGAN2_RANDOMIZE_NOISE = False

# Settings for model running.
# 默认使用GPU
USE_CUDA = True

MAX_IMAGES_ON_DEVICE = 8

MAX_IMAGES_ON_RAM = 1600
