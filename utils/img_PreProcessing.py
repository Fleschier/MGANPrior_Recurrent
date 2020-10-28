import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch.nn.functional as F
from utils.file_utils import image_files, load_as_tensor, Tensor2PIL, split_to_batches
from utils.image_precossing import _add_batch_one
import warnings
warnings.filterwarnings("ignore")


default_load_img_pth=r'.\examples\SR_test'  # windows route
# default_load_img_pth='./examples/SR_test'   # linux route

# 降低分辨率, 但保持图像尺寸不变
def downsample_images(image, factor, mode):
    down = F.interpolate(image, scale_factor=1/factor, mode=mode)       # 先降factor倍的分辨率,再直接补回来,形成清晰的"马赛克"图片
    up_nn = F.interpolate(down, scale_factor=factor, mode='nearest')
    up_bic = F.interpolate(down, scale_factor=factor, mode='bilinear')
    return up_nn, up_bic
  
def convert2target(image, mode='bilinear'):
    # print("img: ", image.size())      # torch.Size([1, 3, 1024, 1024])
    height, weight = image.size()[2], image.size()[3]       # 获取图片维度信息
    print('img size: ', height, 'x', weight)
    img = image
    if(height != weight):
        raise ValueError('error input img size! 请确保输入图片是正方形!')

    elif(height == 1024): return image
    elif(height > 1024):
        if(height % 1024 != 0):
            raise ValueError('error input img size! 请确保输入图片分辨率是1024的整数倍!')
        pre_factor = height // 1024 
        img = F.interpolate(image, scale_factor=pre_factor, mode=mode)
    elif(height < 1024):
        if(1024 % height != 0):
            raise ValueError('error input img size! 请确保输入图片分辨率是1024的因子!')
        pre_factor = 1024 // height
        img = F.interpolate(image, scale_factor=pre_factor, mode=mode)
    return img

# 测试
def convert_test(image, factor, mode='nearest'):
     # print("img: ", image.size())      # torch.Size([1, 3, 1024, 1024])
    height, weight = image.size()[2], image.size()[3]       # 获取图片维度信息
    print('img size: ', height, 'x', weight)
    img = image
    if(height != weight):
        raise ValueError('error input img size! 请确保输入图片是正方形!')

    elif(height == 1024): return image
    elif(height > 1024):
        if(height % 1024 != 0):
            raise ValueError('error input img size! 请确保输入图片分辨率是1024的整数倍!')
        pre_factor = height // 1024 
        img = F.interpolate(image, scale_factor=pre_factor, mode=mode)
    elif(height < 1024):
        if(1024 % height != 0):
            raise ValueError('error input img size! 请确保输入图片分辨率是1024的因子!')
        pre_factor = 1024 // height
        img = F.interpolate(image, scale_factor=pre_factor, mode=mode)
    return img
    

def saveIMG():
    image_list = image_files(default_load_img_pth)

    for i, images in enumerate(split_to_batches(image_list, 1)):
        image_name_list = []
        image_tensor_list = []
        for image in images:
            image_name_list.append(os.path.split(image)[1])
            image_tensor_list.append(_add_batch_one(load_as_tensor(image)))

        # 保存结果
        for img_id, image in enumerate(images):
            down = convert_test(image_tensor_list[img_id], factor=16, mode='nearest')
            y_nn_pil = Tensor2PIL(down)
            y_nn_pil.save(os.path.join(os.path.join('img_test_output', '%s-nn.png' % image_name_list[img_id][:-4])))

if __name__ == "__main__":
    saveIMG()