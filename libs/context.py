#!/usr/bin/env python
# encoding: utf-8

def append_l1_norm(cost, layers, alpha):
    print '..appending l1 norm to cost function'
    for layer in layers:
        for param in layer.params:
            cost = cost + alpha * param.wc * abs(param.data).sum()
    return cost

def append_l2_norm(cost, layers, alpha):
    print '..appending l2 norm to cost function'
    for layer in layers:
        for param in layer.params:
            cost = cost + alpha * param.wc * (param.data**2).sum()
    return cost

def compute_gradient_sgd(
        cost,
        layers,
        learning_rate = 0.1,
        ):
    import theano.tensor as T
    print '..computing sgd gradient(learning_rate %0.8f)' % learning_rate
    updates = [] # final updates

    # coonstruct params and eps
    for layer in layers:
        for param in layer.params:
            updates.append((param.data, param.data - learning_rate * param.eps * T.grad(cost, param.data)))
    return updates

def compute_gradient_momemtent(
        cost,
        layers,
        learning_rate,
        momentum = 0.9,
        ):
    assert momentum >=0 and momentum <= 1
    import theano
    import theano.tensor as T
    print "..computing momentum gradient(learning_rate %0.8f, momentum %0.2f)" % (learning_rate, momentum)
    updates = []
    params = []
    param_updates = []
    for layer in layers:
        for param in layer.params:
            # store the tensor variables
            param_update = theano.shared(param.data.get_value()*0., broadcastable=param.data.broadcastable)
            params.append(param.data)
            param_updates.append(param_update)

            updates.append((param_update, momentum * param_update - learning_rate * param.eps * T.grad(cost, param.data)))

    for i in xrange(len(params)):
        updates.append((params[i], params[i] + param_updates[i]))

    return updates

def ignite_fast_training(
        models,
        n_batches,
        layers,
        n_epochs = 100,
        improvement_threshold = 0.99,
        patience = 10,
        patience_increase = 5,
        param_snapshot_path = None,
        save_period = -1,
        param_prefix = 'param_'
        ):

    import time
    import numpy as np
    print "..begin training (n_epochs %i)" % n_epochs
    if param_snapshot_path is not None:
        load_param(layers, param_snapshot_path)

    start_time = time.clock()

    train_model, valid_model, test_model = models
    n_train_batches, n_valid_batches, n_test_batches = n_batches

    best_valid_score = np.inf
    best_valid_epoch = 0

    best_test_score = np.inf
    best_test_epoch = 0

    epoch = 0
    done_looping = False

    # initialization evalution
    first_valid_score = np.mean([valid_model(i) for i in xrange(n_valid_batches)])
    first_test_score = np.mean([test_model(i) for i in xrange(n_test_batches)])
    print "..inital valid score %0.4f, test score %0.4f (%0.2f min so far)" % (
            first_valid_score, first_test_score, (time.clock()-start_time)/60.)

    # begin to loop
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        this_train_score = np.mean([train_model(i) for i in xrange(n_train_batches)])
        this_valid_score = np.mean([valid_model(i) for i in xrange(n_valid_batches)])
        this_test_score = np.mean([test_model(i) for i in xrange(n_test_batches)])

        if this_valid_score < best_valid_score * improvement_threshold:
            patience  = patience + patience_increase
            best_valid_score = this_valid_score
            best_valid_epoch = epoch

        if this_test_score < best_test_score * improvement_threshold:
            best_test_score = this_test_score
            best_test_epoch = epoch

        print '...eopch %i, train loss %0.4f, valid score %0.4f, test score %0.4f (%0.2f min so far)' % (
                epoch, this_train_score, this_valid_score, this_test_score, (time.clock()-start_time)/60.)

        if -1 != save_period and epoch % save_period == 0:
            save_param(layers, param_prefix + str(epoch) + ".npy")

        if patience <= epoch:
            done_looping = True
            break
    end_time = time.clock()

    print '..Optimization complete, consumed time: %0.2f min' % ((end_time - start_time)/ 60.)
    print '..Best valid score is %0.4f, obtained at epoch %i' % (best_valid_score, best_valid_epoch)
    print '..Best test score is %0.4f, obtained at epoch %i' % (best_test_score, best_test_epoch)

def save_param(layers, param_path):
    import utils
    import numpy as np
    param_path = utils.complement_path(param_path)
    params = []
    for layer in layers:
        params.extend(layer.params)
    print "..saving to %s (params length: %i)" % (param_path, len(params))
    np.save(param_path, params)

def load_param(layers, param_path):
    import utils
    import numpy as np
    param_path = utils.complement_path(param_path)
    params = np.load(param_path)
    print "..loading from %s (params length: %i)" % (param_path, len(params))
    assert len(params) == 2 * len(layers)
    for i in range(len(layers)):
        layers[i].params[0].data.set_value(params[2*i].data.get_value())
        layers[i].params[0].name = params[2*i].name
        layers[i].params[0].eps = params[2*i].eps
        layers[i].params[0].wc = params[2*i].wc

        layers[i].params[1].data.set_value(params[2*i+1].data.get_value())
        layers[i].params[1].name = params[2*i+1].name
        layers[i].params[1].eps = params[2*i+1].eps
        layers[i].params[1].wc = params[2*i+1].wc
