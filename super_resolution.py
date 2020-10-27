import os
import argparse
import torch
import cv2

from utils.file_utils import image_files, load_as_tensor, Tensor2PIL, split_to_batches
from utils.image_precossing import _sigmoid_to_tanh, _tanh_to_sigmoid, _add_batch_one
from Derivable_Models.Derivable_Generator import get_derivable_generator
from utils.manipulate import SR_loss, downsample_images
from inversion.inversion_methods import get_inversion
from inversion.losses import get_loss
from GAN.Model_Settings import MODEL_POOL
from utils.manipulate import convert_array_to_images
import warnings
warnings.filterwarnings("ignore")

from utils.img_PreProcessing import convert2target


# 超参
# 加载的模型名称
tf_model_name = 'pggan_celebahq'
# 输入的图片的路径
# default_load_img_pth='./examples/superresolution'
default_load_img_pth='./examples/SR_test'
# 输出结果图片的路径
output_pth='SR_output'
# 学习率
learning_rate = 1.
# 迭代次数
iterations = 2000
# batch size
BATCH_SIZE = 1

def main(args):
    os.makedirs(args.outputs, exist_ok=True)
    generator = get_derivable_generator(args.gan_model, args.inversion_type, args)  # 生成器
    loss = get_loss(args.loss_type, args)
    sr_loss = SR_loss(loss, args.down, args.factor)
    # to cuda
    generator.cuda()
    loss.cuda()
    inversion = get_inversion(args.optimization, args)
    image_list = image_files(args.target_images)
    frameSize = MODEL_POOL[args.gan_model]['resolution']

    for i, images in enumerate(split_to_batches(image_list, 1)):
        print('%d: Super-resolving %d images ' % (i + 1, 1), end='')
        pt_image_str = '%s\n'
        print(pt_image_str % tuple(images))

        image_name_list = []
        image_tensor_list = []
        for image in images:
            image_name_list.append(os.path.split(image)[1])
            # image = _add_batch_one(load_as_tensor(image))
            image = convert2target(_add_batch_one(load_as_tensor(image)), 'nearest')        # 转化
            image_tensor_list.append(image)
            # print("add..: ", _add_batch_one(load_as_tensor(image)).size())      # torch.Size([1, 3, 64, 64])

        y_gt = _sigmoid_to_tanh(torch.cat(image_tensor_list, dim=0)).cuda()
        # Invert
        latent_estimates, history = inversion.invert(generator, y_gt, sr_loss, batch_size=BATCH_SIZE, video=args.video)
        # Get Images
        y_estimate_list = torch.split(torch.clamp(_tanh_to_sigmoid(generator(latent_estimates)), min=0., max=1.).cpu(), 1, dim=0)
        # 保存结果
        for img_id, image in enumerate(images):
           # up_nn, up_bic, down = downsample_images(image_tensor_list[img_id], factor=args.factor, mode=args.down)
           # y_nn_pil = Tensor2PIL(up_nn)        # 低分辨率化后的图像
            y_estimate_pil = Tensor2PIL(y_estimate_list[img_id])
            y_estimate_pil.save(os.path.join(os.path.join(args.outputs, '%s.png' % image_name_list[img_id][:-4])))
            #y_nn_pil.save(os.path.join(os.path.join(args.outputs, '%s-nn.png' % image_name_list[img_id][:-4])))
            # Create video
            if args.video:
                print('Create GAN-Inversion video.')
                video = cv2.VideoWriter(
                    filename=os.path.join(args.outputs, '%s_sr.avi' % image_name_list[img_id][:-4]),
                    fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                    fps=args.fps,
                    frameSize=(frameSize, frameSize))
                print('Save frames.')
                for i, sample in enumerate(history):
                    image = generator(sample)
                    image_cv2 = convert_array_to_images(image.detach().cpu().numpy())[0][:, :, ::-1]
                    video.write(image_cv2)
                video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SR using multi-code GAN prior')
    # Generator Setting
    parser.add_argument('--gan_model', default=tf_model_name,
                        help='The name of model used.', type=str)
    # Image Path and Saving Path
    parser.add_argument('-i', '--target_images',
                        default=default_load_img_pth,
                        help='Directory with images for SR')
    parser.add_argument('-o', '--outputs',
                        default=output_pth,
                        help='Directory for storing generated images')
    # Parameters for Multi-Code GAN Inversion
    parser.add_argument('--inversion_type', default='PGGAN-Multi-Z',
                        help='Inversion type, PGGAN-Multi-Z for Multi-Code-GAN prior.')
    parser.add_argument('--composing_layer', default=6,
                        help='Composing layer in multi-code gan inversion methods.', type=int)
    parser.add_argument('--z_number', default=30,
                        help='Number of the latent codes.', type=int)
    # Experiment Settings
    # Super-resolution
    parser.add_argument('--down', type=str, default='bilinear',
                        help='Downsampling method.')
    parser.add_argument('--factor', type=int, default=16,
                        help='SR factor.')
    # Loss Parameters
    parser.add_argument('--image_size', default=256,
                        help='Size of images for perceptual model', type=int)
    parser.add_argument('--loss_type', default='Combine',
                        help="['VGG', 'L1', 'L2', 'Combine']. 'Combine' means using L2 and Perceptual Loss.")
    parser.add_argument('--vgg_loss_type', default='L1',
                        help="['L1', 'L2']. The loss used in perceptual loss.")
    parser.add_argument('--vgg_layer', default=16,
                        help='The layer used in perceptual loss.', type=int)
    parser.add_argument('--l1_lambda', default=0.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for L1 loss.", type=float)
    parser.add_argument('--l2_lambda', default=1.,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for L2 loss.", type=float)
    parser.add_argument('--vgg_lambda', default=1.0,
                        help="Used when 'loss_type' is 'Combine'. Trade-off parameter for Perceptual loss.", type=float)
    # Optimization Parameters
    parser.add_argument('--optimization', default='GD',
                        help="['GD', 'Adam']. Optimization method used.")  # inversion_type
    parser.add_argument('--init_type', default='Zero',
                        help="['Zero', 'Normal']. Initialization method. Using zero init or Gaussian random vector.")
    parser.add_argument('--lr', default=learning_rate,
                        help='Learning rate.', type=float)
    parser.add_argument('--iterations', default=iterations,
                        help='Number of optimization steps.', type=int)

    # Video Settings
    parser.add_argument('--video', type=bool, default=False,
                        help='Save video. False for no video.')
    parser.add_argument('--fps', type=int, default=24,
                        help='Frame rate of the created video.')

    args, other_args = parser.parse_known_args()

    ### RUN
    main(args)
