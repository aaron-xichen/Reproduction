import theano
import theano.tensor as T
import context
from layers import ConvLayer
import utils
import numpy as np

patch_shape = (33,33)
nkerns = [64, 32]

batch_size = 5000
learning_rate = 3e-7
weight_decay = 2

# prepare the data
print ".preparing data"
dataset_paths = [
        utils.complement_path('/share/blur_images/all_in_one/train_o_set.npy'),
        utils.complement_path('/share/blur_images/all_in_one/train_b_set.npy'),
        utils.complement_path('/share/blur_images/all_in_one/valid_o_set.npy'),
        utils.complement_path('/share/blur_images/all_in_one/valid_b_set.npy'),
        utils.complement_path('/share/blur_images/all_in_one/test_o_set.npy'),
        utils.complement_path('/share/blur_images/all_in_one/test_b_set.npy')]

datasets = [
        utils.shared_dataset(np.load(dataset_path)) for dataset_path in dataset_paths]

# build the network
print ".building network"

normal  = T.fmatrix('normal')
corrupt = T.fmatrix('corrupt')

index = T.lscalar('index')

corrupt_input = corrupt.reshape((batch_size, 1, patch_shape[0], patch_shape[1]))
normal_input = normal.reshape((batch_size, 1, patch_shape[0], patch_shape[1]))

# patch extraction and representation, output shape=(33-9+1, 33-9+1)=(25, 25)
layer0_conv = ConvLayer(
        input = corrupt_input,
        initW = 0.001,
        image_shape=(batch_size, 1, patch_shape[0], patch_shape[1]),
        filter_shape=(nkerns[0], 1, 9, 9),
        activate='relu',
        border_mode='valid',
    )

# non-linear mapping, output shape=(25-1+1, 25-1+1)=(25,25)
layer1_conv = ConvLayer(
        input = layer0_conv.output,
        initW = 0.001,
        image_shape=(batch_size, nkerns[0], 25, 25),
        filter_shape=(nkerns[1], nkerns[0], 1, 1),
        activate='relu',
        border_mode='valid',
    )

# reconstruction, output shape=(25-5+1, 25-5+1)=(21 ,21)
layer2_conv = ConvLayer(
        input = layer1_conv.output,
        eps_W = 0.1,
        eps_B = 0.1,
        initW= 0.001,
        image_shape=(batch_size, nkerns[1], 25, 25),
        filter_shape=(1, nkerns[1], 5, 5),
        activate='linear',
        border_mode='valid',
    )

# output
layer3_output = T.set_subtensor(normal_input[:, :, :21, :21], layer2_conv.output)

# all layers in a list
layers = [layer0_conv, layer1_conv, layer2_conv]

# build the cost function
diff = normal_input.reshape((batch_size, 1*patch_shape[0]*patch_shape[1])) \
        - layer3_output.reshape((batch_size, 1*patch_shape[0]*patch_shape[1]))

mse = (diff**2).sum(axis=1).mean()
# mse = T.mean(T.sum(diff**2, axis=1))
cost = context.append_l1_norm(mse, layers, alpha=weight_decay) # append l1 norm
cost = context.append_l2_norm(cost, layers, alpha=weight_decay) # append l2 norm

# compute the gradient
updates = context.compute_gradient_sgd(cost, layers, learning_rate=learning_rate)
# updates = context.compute_gradient_momemtent(cost, layers, learning_rate=learning_rate, momentum=0.9)

# build the train function
print ".building train model"
train_model= theano.function(
            inputs = [index],
            outputs = cost,
            updates = updates,
            givens = {
                normal : datasets[0][index*batch_size:(index+1)*batch_size],
                corrupt : datasets[1][index*batch_size:(index+1)*batch_size]
            }
        )

# build the valid function
print ".building valid model"
valid_model= theano.function(
            inputs = [index],
            outputs = mse,
            givens = {
                normal : datasets[2][index*batch_size:(index+1)*batch_size],
                corrupt : datasets[3][index*batch_size:(index+1)*batch_size]
            }
        )

# build the test function
print ".building test model"
test_model = theano.function(
            inputs = [index],
            outputs = mse,
            givens = {
                normal : datasets[4][index*batch_size:(index+1)*batch_size],
                corrupt : datasets[5][index*batch_size:(index+1)*batch_size]
            }
        )

models = [train_model, valid_model, test_model]
n_batches = [len(datasets[0].get_value()) // batch_size, len(datasets[2].get_value()) // batch_size, len(datasets[4].get_value()) // batch_size]

print ".training"
context.ignite_fast_training(
        models = models,
        n_batches = n_batches,
        layers = layers,
        n_epochs = 100,
        # param_snapshot_path = 'params.npy',
        save_period = 5,
        patience_increase = 10,
        )
