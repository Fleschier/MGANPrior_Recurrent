import torch
import argparse
import cv2
import os

from Derivable_Models.Derivable_Generator import get_derivable_generator
from inversion.losses import get_loss
from inversion.inversion_methods import get_inversion
from utils.file_utils import image_files,  load_as_tensor, Tensor2PIL, split_to_batches
from GAN.Model_Settings import MODEL_POOL
from utils.image_precossing import _sigmoid_to_tanh, _tanh_to_sigmoid, _add_batch_one
from utils.manipulate import convert_array_to_images

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
# batch size
BATCH_SIZE = 1

def run(args):
    os.makedirs(args.outputs,exist_ok=True) # 生成输出路径文件夹，存在则跳过
    # 生成器
    generator = get_derivable_generator(args.gan_model, args.inversion_type, args)
    loss = get_loss(args.loss_type, args)       # 计算误差
    generator.cuda()    # pytorch需要手动放入GPU进行运算
    loss.cuda()
    inversion = get_inversion(args.optimization, args)
    image_list = image_files(args.target_images)        # 获取输入图片路径
    frameSize = MODEL_POOL[args.gan_model]['resolution']        # 获取图像分辨率

    # 按照batch大小分批处理图像
    for i, images in enumerate(split_to_batches(image_list, 1)):
        print('%d: Inverting %d images :' % (i + 1, 1), end='')
        # pt_image_str = '%s\n'
        print('%s\n' % tuple(images))

        image_name_list = []
        image_tensor_list = []
        for image in images:
            image_name_list.append(os.path.split(image)[1])
            image_tensor_list.append(_add_batch_one(load_as_tensor(image)))
        # torch.cat(tensors, dim=0, out=None) → Tensor
        # tensors (sequence of Tensors) – any python sequence of tensors of the same type. Non-empty tensors provided must have the same shape, except in the cat dimension.
        # dim (int, optional) – the dimension over which the tensors are concatenated
        # out (Tensor, optional) – the output tensor.
        y_gt = _sigmoid_to_tanh(torch.cat(image_tensor_list, dim=0)).cuda() # 在维度0上连接所有的tensor并且将值域映射到[-1, 1]
        # 逆映射, 生成图像tensor
        latent_estimates, history = inversion.invert(generator, y_gt, loss, batch_size=BATCH_SIZE, video=args.video)
        # 将值域从[-1,1]映射到[0,1], 使用torch.clamp()进一步保证值域在[0,1]
        y_estimate_list = torch.split(torch.clamp(_tanh_to_sigmoid(generator(latent_estimates)), min=0., max=1.).cpu(), 1, dim=0)
        # Save
        for img_id, image in enumerate(images):
            y_estimate_pil = Tensor2PIL(y_estimate_list[img_id])        # 从tensor转化为PIL image并保存
            y_estimate_pil.save(os.path.join(args.outputs, image_name_list[img_id]))

            # Create video
            if args.video:
                print('Create GAN-Inversion video.')
                video = cv2.VideoWriter(
                    filename=os.path.join(args.outputs, '%s_inversion.avi' % image_name_list[img_id]),
                    fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                    fps=args.fps,
                    frameSize=(frameSize, frameSize))
                print('Save frames.')
                for i, sample in enumerate(history):
                    image = generator(sample)
                    image_cv2 = convert_array_to_images(image.detach().cpu().numpy())[0][:, :, ::-1]
                    video.write(image_cv2)
                video.release()

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
    # 使用的latent codes的数量
    parser.add_argument('--z_number', type = int, default=30,
                        help='Number of the latent codes.')

    # VGG作为一种广泛使用的深度神经网络，其卷积层在一定程度上能够提取图像的特征信息.
    # 令pre-trained VGG的卷积层作为误差网络，将生成网络生成的图像 y~ 输入误差网络计算每个卷积层得到的特征，
    # 再将这些特征跟y-truth(即原始图像)作比较得到感知误差(perceptual loss)
    # 计算Loss相关的参数
    # 图片的size，默认是256*256. 输入图片用来放入VGG网络计算与GAN的generator生产的图片之间的loss
    parser.add_argument('--image_size', type=int, default=256,
                        help='Size of images for perceptual model')
    parser.add_argument('--loss_type', default='Combine',
                        help="['VGG', 'L1', 'L2', 'Combine']. 'Combine' means using L2 and Perceptual Loss.")
    parser.add_argument('--vgg_loss_type', default='L1',
                        help="['L1', 'L2']. The loss used in perceptual loss.")
    parser.add_argument('--vgg_layer', type=int, default=16,        # 计算感知误差用到的VGG卷积层层数,默认为16
                        help='The layer used in perceptual loss.')
    parser.add_argument('--l1_lambda', default=0.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for L1 loss.", type=float)
    parser.add_argument('--l2_lambda', default=1.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for L2 loss.", type=float)
    parser.add_argument('--vgg_lambda', default=1.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for Perceptual loss.", type=float)
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

    # Video Settings
    parser.add_argument('--video', type=bool, default=True, help='Save video. False for no video.')
    parser.add_argument('--fps', type=int, default=24, help='Frame rate of the created video.')


    args, other_args = parser.parse_known_args()
    run(args)
