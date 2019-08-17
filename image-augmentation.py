# Generate/Transform sample images

import warnings
import cv2
import glob
import os
import numpy as np
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

for file_path in glob.glob('dataset/images/*.jpg'):
    file_name_with_extension = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name_with_extension)

    image = cv2.imread(file_path)
    image = cv2.resize(image, (512, 512))
    cv2.imwrite('dataset/transform/' + file_name_with_extension, image)

    crop_and_pad = iaa.CropAndPad(percent=(-0.25, 0.25))
    img = crop_and_pad.augment_image(image)
    cv2.imwrite('dataset/transform/' + file_name + '-crop-and-pad' + file_extension, img)

    flip_lr = iaa.Fliplr(1.0)
    img = flip_lr.augment_image(image)
    cv2.imwrite('dataset/transform/' + file_name + '-flip-lr' + file_extension, img)

    flip_ud = iaa.Flipud(1.0)
    img = flip_ud.augment_image(image)
    cv2.imwrite('dataset/transform/' + file_name + '-flip-ud' + file_extension, img)

    affine = iaa.Affine(rotate=(-90))
    img = affine.augment_image(image)
    cv2.imwrite('dataset/transform/' + file_name + '-affine' + file_extension, img)

    avg_blur = iaa.AverageBlur(k=(3, 3))
    img = avg_blur.augment_image(image)
    cv2.imwrite('dataset/transform/' + file_name + '-avg-blur' + file_extension, img)

    dropout = iaa.Dropout(p=(0, 0.1))
    img = dropout.augment_image(image)
    cv2.imwrite('dataset/transform/' + file_name + '-dropout' + file_extension, img)


def img_func(images, random_state, parents, hooks):
    for img in images:
        img[::4] = 0
    return images


def keypoint_func(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images


for file_path in glob.glob('dataset/transform/*.jpg'):
    file_name_with_extension = os.path.basename(file_path)
    file_name, file_extension = os.path.splitext(file_name_with_extension)

    image = cv2.imread(file_path)

    lamb = iaa.Lambda(img_func, keypoint_func)
    img = lamb.augment_image(image)
    cv2.imwrite('dataset/transform/' + file_name + '-lambda' + file_extension, img)
    # cv2.imshow('lamb', img)

    with_channels_add = iaa.WithChannels(0, iaa.Add((10, 100)))
    img = with_channels_add.augment_image(image)
    cv2.imwrite('dataset/transform/' + file_name + '-with-channels-add' + file_extension, img)
    # cv2.imshow('with_channels_add', img)

    with_channels_affine = iaa.WithChannels(0, iaa.Affine(rotate=(0, 45)))
    img = with_channels_affine.augment_image(image)
    cv2.imwrite('dataset/transform/' + file_name + '-with-channels-affine' + file_extension, img)
    # cv2.imshow('with_channels_affine', img)

    # cv2.waitKey()
