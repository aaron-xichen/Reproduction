#!/usr/bin/env python
# encoding: utf-8

import theano
import utils
import numpy as np
import os
import PIL.Image as Image

def synthesize_training_image(
        dir_paths,
        output_dir = '/share/blur_images/all_in_one2/',
        patch_shape = (48, 48),
        stride_shape = (48, 48),
        scale_ratio = 6,
        is_shuffle = True,
        is_scale = False,
        is_noise = False,
        ):

    # read and split images, type is list
    print "patch_shape: {}".format(patch_shape)
    print "stride_shape: {}".format(stride_shape)
    print "scale ratio: {}".format(scale_ratio)
    images = []
    for dir_path in dir_paths:
        dir_path = utils.complement_path(dir_path)
        images.extend(
                utils.sub_images(
                    dir_path = dir_path,
                    patch_shape = patch_shape,
                    stride_shape = stride_shape,
                    is_scale = is_scale,
                    is_noise = is_noise)[0])
    images = np.asarray(images, dtype=theano.config.floatX)
    print images.shape
    mean = np.mean(images, axis = 0)

    # operation on images, type is list of ndarray
    original_images = []
    corrupted_images = []
    for each_image in images:
        original_images.append(each_image - mean)
        ## gaussian noise
        noise_each_image = each_image - mean + np.random.normal(size=each_image.shape)
        im = Image.fromarray(noise_each_image.reshape(patch_shape))
        ## downsample
        im_downsampling = im.resize((patch_shape[0] // scale_ratio, patch_shape[1] // scale_ratio), Image.BICUBIC)
        ## upsampling
        im_upsampling = im_downsampling.resize(patch_shape, Image.BICUBIC)
        assert im_upsampling.size == im.size

        new_image = np.array(im_upsampling.getdata())
        corrupted_images.append(new_image)

    original_images = np.asarray(original_images, dtype=theano.config.floatX)
    corrupted_images = np.asarray(corrupted_images, dtype=theano.config.floatX)

    assert images.shape == corrupted_images.shape

    # shuffle
    if is_shuffle:
        perms = np.random.permutation(len(images))
        original_images = original_images[perms]
        corrupted_images = corrupted_images[perms]

    # save images and corrupted_images
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    train_nums = len(images) // 7 * 5
    valid_nums = len(images) // 7

    train_o = original_images[:train_nums]
    train_b = corrupted_images[:train_nums]
    valid_o = original_images[train_nums:train_nums+valid_nums]
    valid_b = corrupted_images[train_nums:train_nums+valid_nums]
    test_o = original_images[train_nums+valid_nums:]
    test_b = corrupted_images[train_nums+valid_nums:]

    np.save(os.path.join(output_dir, 'train_o_set.npy'), train_o)
    np.save(os.path.join(output_dir, 'train_b_set.npy'), train_b)
    np.save(os.path.join(output_dir, 'valid_o_set.npy'), valid_o)
    np.save(os.path.join(output_dir, 'valid_b_set.npy'), valid_b)
    np.save(os.path.join(output_dir, 'test_o_set.npy'), test_o)
    np.save(os.path.join(output_dir, 'test_b_set.npy'), test_b)

def synthesize_predicting_images(
        data_paths,
        output_dir = "/share/blur_images/all_in_one2/"
        ):
    images = []
    for data_path in data_paths:
        for root, dirs, files in os.walk(utils.complement_path(data_path)):
            for file in files:
                path = os.path.join(root, file)
                print path
                img = utils.PIL2array(path)
                images.append(img)

    print "total images size: %i" % len(images)

    output_dir = utils.complement_path(output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    np.save(os.path.join(output_dir,'testing_full_images.npy'), images)

if __name__ == "__main__":
    synthesize_training_image(
            dir_paths=['/share/blur_images/mixture/', '/share/blur_images/poor_accuracy'])
    synthesize_predicting_images(
            data_paths = ["/share/TestImage/mohu"]
            )
