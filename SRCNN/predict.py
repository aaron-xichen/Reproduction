import theano
import theano.tensor as T
import context
from layers import ConvLayer
import utils
import numpy as np
import os
import PIL.Image as Image

def predict_single(
        one,
        param_snapshot_path,
        patch_shape = (33,33),
        nkerns = [64, 32],
        ):

    height, width = one.shape
    # build the network
    print ".building network"
    x = T.fmatrix('x')

    layer0_input = x.reshape((1, 1, height, width))

    # patch extraction and representation, output shape=(33,33)
    layer0_conv = ConvLayer(
            input = layer0_input,
            image_shape=(1, 1, height, width),
            filter_shape=(nkerns[0], 1, 9, 9),
            activate='relu',
            border_mode='valid',
        )

    # non-linear mapping, output shape=(33,33)
    layer1_conv = ConvLayer(
            input = layer0_conv.output,
            image_shape=(1, nkerns[0], height, width),
            filter_shape=(nkerns[1], nkerns[0], 1, 1),
            activate='relu',
            border_mode='valid',
        )

    # reconstruction, output shape=(33,33)
    layer2_conv = ConvLayer(
            input = layer1_conv.output,
            image_shape=(1, nkerns[1], height, width),
            filter_shape=(1, nkerns[1], 5, 5),
            activate='linear',
            border_mode='valid',
        )

    # all layers in a list
    layers = [layer0_conv, layer1_conv, layer2_conv]

    # loading param snapshot
    context.load_param(layers, param_snapshot_path)

    # build predict model
    print '.building predict model'
    predict_model = theano.function(
                inputs = [ x ],
                outputs = layers[-1].output,
            )
    return predict_model(one)

def predict_batch():
    patch_shape = (33,33)
    nkerns = [64, 32]
    param_snapshot_path = 'try1/param_50.npy'
    plot_save_dir = 'output_not_scale'

    # prepare the data
    print ".preparing data"
    datasets = np.load(utils.complement_path('/share/blur_images/all_in_one/testing_full_images.npy'))

    predict_result = []
    for one in datasets:
        predict_result.extend(predict_single(
            param_snapshot_path = param_snapshot_path,
            # one = utils.shared_dataset(one),
            one = one.astype(theano.config.floatX),
            # shape = one.shape,
            patch_shape = patch_shape,
            nkerns = nkerns
            ))

    # assert len(predict_result) == len(datasets)

    # save plot image
    plot_save_dir = utils.complement_path(plot_save_dir)
    if not os.path.exists(plot_save_dir):
        os.mkdir(plot_save_dir)

    print '.saving plot image'
    for i in xrange(len(datasets)):
        origin_img = datasets[i]
        # origin_img = origin_img + origin_img.min()
        # origin_img = origin_img / origin_img.max() * 255.
        reconstruct_img = predict_result[i][0]
        # reconstruct_img = reconstruct_img + reconstruct_img.min()
        # reconstruct_img = reconstruct_img / reconstruct_img.max() * 255.

        # assert origin_img.shape == reconstruct_img.shape
        # join = np.vstack([origin_img, reconstruct_img])
        img1 = Image.fromarray(origin_img.astype(np.uint8))
        img2 = Image.fromarray(reconstruct_img.astype(np.uint8))

        img_path = os.path.join(plot_save_dir, str(i) + '_o.jpg')
        print 'saving to %s' % img_path
        img1.save(img_path)

        img_path = os.path.join(plot_save_dir, str(i) + '_r.jpg')
        print 'saving to %s' % img_path
        img2.save(img_path)


if __name__ == "__main__":
    predict_batch()
