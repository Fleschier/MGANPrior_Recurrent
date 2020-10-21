import torch
import argparse
import cv2
import os

# 参数
# 加载的模型名称
tf_model_name = 'pggan_churchoutdoor'
# 模仿的图片的路径
default_load_img_pth='./examples/gan_inversion/church'
# 输出结果图片的路径
output_pth='./inversion_output'
# 学习率
learning_rate = 1.
# 迭代次数
iterations = 3000

def run(args):
    os.makedirs(args.outputs,exist_ok=True) # 生成输出路径文件夹，存在则跳过
    # 生成器
    generator = get_derivable_generator(args.gan_model, args.inversion_type, args)

if(__name__ == '__main__'):
    # 可以通过命令行输入参数
    parser = argparse.ArgumentParser(description='Multi-Code GAN Inversion')

    # 添加参数
    # 加载模型选择
    parser.add_argument('--gan_model',
        default=tf_model_name,
        help='The name of model used.', type=str)
    # 加载图像路径选择
    parser.add_argument('--target_images',
        default=default_load_img_pth,
        help='Target images to invert.')
    # 保存路径选择
    parser.add_argument('--outputs',
        default=output_pth,
        help='Path to save results.')

    # Multi-code-inversion参数
    # 默认使用multi-code反演类型
    parser.add_argument('--inversion_type', default='PGGAN-Multi-Z',
                        help='Inversion type, "PGGAN-Multi-Z" for Multi-Code-GAN prior.')
    # 层数
    parser.add_argument('--composing_layer', type=int, default=6,
                        help='Composing layer in multi-code gan inversion methods.')
    # 使用的latent layer的数量
    parser.add_argument('--z_number', type = int, default=30,
                        help='Number of the latent codes.')
    # 计算Loss相关的参数
    # 图片的size，默认说256*256
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size of images for perceptual model')
    # 损失函数类型，默认采用混合类型，即L2和perceptual结合在一起使用
    parser.add_argument('--loss_type', default='Combine',
                        help="['VGG', 'L1', 'L2', 'Combine']. 'Combine' means using L2 and Perceptual Loss.")
    # 
    parser.add_argument('--vgg_loss_type', default='L1',
                        help="['L1', 'L2']. The loss used in perceptual loss.")
    # 
    parser.add_argument('--vgg_layer', type=int, default=16,
                        help='The layer used in perceptual loss.')
    # 
    parser.add_argument('--l1_lambda', default=0.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for L1 loss.", type=float)
    # 
    parser.add_argument('--l2_lambda', default=1.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for L2 loss.", type=float)
    # 
    parser.add_argument('--vgg_lambda', default=1.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for Perceptual loss.", type=float)
    # 优化的参数
    # 优化器的类型选择
    parser.add_argument('--optimization', default='GD',
                        help="['GD', 'Adam']. Optimization method used.")
    # 初始化类型
    parser.add_argument('--init_type', default='Normal',
                        help="['Zero', 'Normal']. Initialization method. Using zero init or Gaussian random vector.")
    # 学习率
    parser.add_argument('--lr', default=learning_rate,
                        help='Learning rate.', type=float)
    # 迭代次数
    parser.add_argument('--iterations', default=iterations,
                        help='Number of optimization steps.', type=int)







    args, other_args = parser.parse_known_args()
    run(args)
