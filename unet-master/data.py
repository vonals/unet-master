# 数据读入，打包处理模块
#
from __future__ import print_function
from keras.preprocessing import image
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte
from sklearn.ensemble import VotingClassifier

Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img, mask):
    img = img / 255
    mask = mask / 255  # 归一化
    mask[mask > 0.5] = 1  # 二值化
    mask[mask <= 0.5] = 0
    return (img, mask)


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode = "grayscale",
                    mask_color_mode = "grayscale", image_save_prefix  = "image", mask_save_prefix  = "mask",
                   save_to_dir = None, target_size = (256,256),seed = 1):
    image_datagen = image.ImageDataGenerator(**aug_dict)  # 创建ImageDataGenerator生成器
    mask_datagen = image.ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],  # 子文件夹参数
        class_mode = None,
        color_mode = image_color_mode,  # grayscale或rgb
        target_size = target_size,  # 图像resize尺寸
        batch_size = batch_size,
        save_to_dir = save_to_dir,  # 是否保存到目录
        save_prefix  = image_save_prefix,  # 生成文件前缀
        seed = seed)  # 随机种子
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)  # 将img和mask打包成元组的列表[(img,mask),(img,mask),......]
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)  # 调整图片
        yield (img, mask)


def testGenerator(test_path, num_image = 30, target_size = (256, 256), flag_multi_class = False, as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.tif" % i), as_gray = as_gray)
        img = img/255
        img = trans.resize(img,target_size)
        img = np.reshape(img, img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,)+img.shape)
        yield img


def QGenerator(test_path, num_image = 1, target_size = (256, 256), flag_multi_class = False, as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path), as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img, img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,)+img.shape)
        yield img


def geneTrainNpy(image_path, mask_path, image_prefix = "image", mask_prefix = "mask", image_as_gray = True, mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = io.imread(item, as_gray = image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray = mask_as_gray)
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


# 保存结果
def saveResult(save_path, npyfile, deep_supervision=False, mode_accuracy=False):
    # 试图每人一票时归一化
    # for i in range(len(npyfile)):
    #     npyfile[i][npyfile[i] > 0.1] = 1
    #     npyfile[i][npyfile[i] <= 0.1] = 0
    if(deep_supervision):
        # lenth = len(npyfile)
        # 准确模式 （有问题）
        if(mode_accuracy):
            # 4输出加权混合模式
            npyfile = (npyfile[0]*1+npyfile[1]*1+npyfile[2]*2+npyfile[3]*4)/8
            # volting = VotingClassifier(npyfile, voting='soft')

        else:
            npyfile = npyfile[-1]
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        # 对结果进行二值化
        img[img > 0.3] = 1
        img[img <= 0.3] = 0
        io.imsave(os.path.join(save_path, "%d_predict.tif" % i), img_as_ubyte(img))



