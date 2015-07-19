__author__ = "Aaron Chen"
__email__  ="aaron.xichen@gmail.com"

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

class ParamItem(object):
    def __init__(self, name, data, eps, wc):
        self.name = name
        self.data = data
        self.eps = eps
        self.wc = wc

class ConvLayer(object):
    def __init__(self,
            input,
            image_shape,
            filter_shape, border_mode = 'same', stride = 1, group = 1, sparsity = None,
            initW = 0.01, probW = 'gaussian', eps_W = 1, wc_W = 1,
            initB = 0., eps_B = 2, wc_B = 1,
            seed = 1, activate = 'relu'):

        self.image_shape = image_shape
        self.filter_shape = filter_shape
        assert self.image_shape[1] == self.filter_shape[1] * group
        self.border_mode = border_mode
        self.stride = stride
        self.group = group
        self.sparsity = sparsity
        self.initW = initW
        self.probW = probW
        self.initB = initB

        # setting params, including W and b
        self.rng = np.random.RandomState(seed)
        self.params = []
        if self.probW == 'uniform':
            W = theano.shared(np.asarray(
                self.rng.uniform(low = -self.initW, high = self.initW, size = self.filter_shape),
                dtype = theano.config.floatX), borrow = True)
        elif self.probW == 'gaussian':
            W = theano.shared(np.asarray(
                self.rng.normal(loc = 0, scale = self.initW, size = self.filter_shape),
                dtype = theano.config.floatX), borrow = True)
        else:
            print 'Unknown probW.'
            assert 1 == 0
        self.W = ParamItem(name='W',data=W,eps=eps_W,wc=wc_W)
        self.params = [self.W]

        b = theano.shared(self.initB *
                np.ones((self.filter_shape[0],),theano.config.floatX),
                borrow = True)
        self.b = ParamItem(name='B',data=b,eps=eps_B,wc=wc_B)
        self.params.append(self.b)

        # setting activation function
        if activate == 'relu':
            self.activate = lambda x: x * (x > 0)
        elif activate == 'sigmoid':
            self.activate = T.nnet.sigmoid
        elif activate == 'softmax':
            self.activate = T.nnet.softmax
        elif activate == 'tanh':
            self.activate = T.tanh
        else:
            self.activate = lambda x: x

        # symbolic computation
        self.update(input)

    def update(self, input):
        convOut = 0
        if self.sparsity is not None:
            assert (self.sparsity <= 1) & (self.sparsity > 0)
            self.mask = np.asarray(self.rng.binomial(n = 1, p = (1 - self.sparsity),
                                                size = self.filter_shape),
                                dtype = theano.config.floatX)
            convOut = conv_theano(input = input, image_shape = self.image_shape, W = self.W.data * self.mask,
                                    filter_shape = self.filter_shape, border_mode = self.border_mode,
                                    stride = self.stride, group = self.group)
        else:
            convOut = conv_theano(input = input, image_shape = self.image_shape, W = self.W.data,
                                    filter_shape = self.filter_shape, border_mode = self.border_mode,
                                    stride = self.stride, group = self.group)
        self.output = self.activate(convOut + self.b.data.dimshuffle('x', 0, 'x', 'x'))

    def setPretrain(self, W=None, b=None):
        if W is not None:
            self.W.name = W.name
            self.W.data.set_value(W.data.get_value(), borrow=True)
            self.W.eps = W.eps
            self.W.wc = W.wc
        if b is not None:
            self.b.name = b.name
            self.b.data.set_value(b.data.get_value(), borrow=True)
            self.b.eps = b.eps
            self.b.wc = b.wc

# able to handle stride
class PoolLayer(object):
    def __init__(self,
            input,
            image_shape,
            pool_type = 'max',
            pool_size = 3,
            pool_stride = 2):
        assert pool_size >= pool_stride
        self.pool_type = pool_type
        self.image_shape = image_shape
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        # symbolic computation
        self.update(input)

    def update(self, input):
        if self.pool_type == 'max':
            self.output = max_pool(bc01 = input,
                                   pool_shape = (self.pool_size, self.pool_size),
                                   pool_stride = (self.pool_stride, self.pool_stride),
                                   image_shape = self.image_shape)
        elif self.pool_type == 'average':
            self.output = avg_pool(bc01 = input,
                                pool_shape=(self.pool_size, self.pool_size),
                                pool_stride=(self.pool_stride, self.pool_stride),
                                image_shape=self.image_shape
                            )
class LrnLayer(object):
    def __init__(self,
            input,
            image_shape = None,
            lrn_type = 'crossMap',
            lrn_size = 5,
            k = 1.0,
            alpha = 1e-4,
            beta = 0.75):
        assert lrn_size % 2 == 1
        self.lrn_type = lrn_type
        self.lrn_size = lrn_size
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.filter_shape = [1, 1, self.lrn_size, self.lrn_size]
        self.W = theano.shared(np.ones(shape=self.filter_shape,dtype=theano.config.floatX),borrow=True)

        if self.lrn_type == 'sameMap' and image_shape is None:
            print "need image_shape params"
            assert 1 == 2
        self.image_shape = image_shape

        # symbolic computation
        self.update(input)

    def update(self, input):
        # transform the value of each feature according to the adjacent lrn_size feature maps
        if self.lrn_type == 'crossMap':
            b, ch, r, c = input.shape
            half = np.int32(self.lrn_size) / 2
            padSqr = T.alloc(np.float32(0.), b, ch + self.lrn_size - 1, r, c)
            padSqr = T.set_subtensor(padSqr[:, half:half + ch, :, :], T.sqr(input))
            scale = self.k
            for i in xrange(self.lrn_size):
                scale += self.alpha / self.lrn_size * padSqr[:, i:i + ch, :, :]
            self.output = input / (scale ** self.beta)

        elif self.lrn_type == 'sameMap':
            b, ch, r, c = input.shape
            assert (b>0) & (ch>0) &(r>0) & (c>0)
            half = np.int32(self.lrn_size) / 2
            sqr = T.sqr(input)
            sqr = T.reshape(sqr,(b*ch, 1, r, c))
            sqrSum = conv.conv2d(
                        input = sqr,
                        filters = self.W,
                        filter_shape=self.filter_shape,
                        image_shape=self.image_shape,
                        border_mode='full'
                    )
            # same size
            sqrSum = sqrSum[:,:,half:-half,half:-half]
            sqrSum = T.reshape(sqrSum,(b, ch, r, c))
            self.output = input / ((self.k + self.alpha / (self.lrn_size ** 2) * sqrSum) ** self.beta)
        else:
            print 'Wrong normalization type.'
            assert 1 == 0


class DropoutLayer(object):
    def __init__(self,
            input,
            p = 0.5,
            seed = 1,
            train_flag=True):
        assert (p > 0) & (p < 1)
        self.p = p
        self.rng = T.shared_randomstreams.RandomStreams(seed)
        self.trainFlag = train_flag

        # symbolic computation
        self.update(input)

    def update(self, input):
        if self.trainFlag == True:
            mask = self.rng.binomial(size = input.shape, n = 1, p = self.p).astype(theano.config.floatX)
            self.output = mask * input
        else :
            self.output = input * self.p

class HiddenLayer(object):
    def __init__(self,
            input,
            n_in,
            n_out,
            initW = 0.01, probW = 'gaussian', eps_W = 1, wc_W = None,
            initB = 0., eps_B = 2, wc_B = None, seed = 1,
            activate = 'None'):
        self.n_in = n_in
        self.n_out = n_out
        self.initW = initW
        self.probW = probW
        self.initB = initB

        rng = np.random.RandomState(seed)
        self.params = []
        if self.probW == 'uniform':
            if activate == 'tanh':
                bound = np.sqrt(6. / (self.n_in + self.n_out))
            elif activate == 'sigmoid':
                bound = 4 * np.sqrt(6. / (self.n_in + self.n_out))
            else:
                bound = initW
            W = theano.shared(np.asarray(
                rng.uniform(low=-bound, high=bound,  size = (self.n_in, self.n_out)),
                dtype = theano.config.floatX), borrow = True)
        elif self.probW == 'gaussian':
            W = theano.shared(np.asarray(
                rng.normal(loc = 0, scale = self.initW, size = (self.n_in, self.n_out)),
                dtype = theano.config.floatX), borrow = True)
        else:
            print 'Unknown probW.'
            assert 1 == 0
        self.W = ParamItem(name='W',data=W,eps = eps_W, wc=wc_W)
        self.params = [self.W]

        b = theano.shared(self.initB *
                np.ones((self.n_out,), theano.config.floatX),
                borrow = True)
        self.b = ParamItem(name='b',data=b,eps = eps_B, wc=wc_B)
        self.params.append(self.b)

        if activate == 'relu':
            print 'Activation in Hidden Layer is ReLU'
            self.activate = lambda x: x * (x > 0)
        elif activate == 'sigmoid':
            print 'Activation in Hidden Layer is sigmoid'
            self.activate = T.nnet.sigmoid
        elif activate == 'softmax':
            print 'Activation in Hidden Layer is softmax'
            self.activate = T.nnet.softmax
        elif activate == 'tanh':
            print 'Activation in Hidden Layer is tanh'
            self.activate = T.tanh
        elif activate == 'None':
            print 'Activation in Hidden Layer is None'
            self.activate = lambda x: x
        else:
            print 'wrong neuron'
            assert 1 == 2

        # symbolic computaiton
        self.update(input)

    def update(self, input):
        lin_output = T.dot(input, self.W.data) + self.b.data
        self.output = self.activate(lin_output)

    def setPretrain(self, W=None, b=None):
        if W is not None:
            self.W.name = W.name
            self.W.data.set_value(W.data.get_value(), borrow=True)
            self.W.eps = W.eps
            self.W.wc = W.wc
        if b is not None:
            self.b.name = b.name
            self.b.data.set_value(b.data.get_value(), borrow=True)
            self.b.eps = b.eps
            self.b.wc = b.wc

class LogisticRegression(object):
    def __init__(self,
            input,
            n_in,
            n_out,
            initW = 0.01, probW = 'uniform', eps_W = 1, wc_W = None,
            initB = 0.,eps_B = 2, wc_B = None, seed = 1):

        self.n_in = n_in
        self.n_out = n_out
        self.initW = initW
        self.probW = probW
        self.initB = initB

        rng = np.random.RandomState(seed)

        self.params = []
        if self.probW == 'uniform':
            W = theano.shared(np.asarray(
                rng.uniform(low = -self.initW, high = self.initW, size = (self.n_in, self.n_out)),
                dtype = theano.config.floatX), borrow = True)
        elif self.probW == 'gaussian':
            W = theano.shared(np.asarray(
                rng.normal(loc = 0, scale = self.initW, size = (self.n_in, self.n_out)),
                dtype = theano.config.floatX), borrow = True)
        else:
            print 'Unknown probW.'
            assert 1 == 0
        self.W = ParamItem(name='W', data=W, eps = eps_W, wc=wc_W)
        self.params = [self.W]

        b = theano.shared(self.initB *
                np.ones((self.n_out,), theano.config.floatX),
                borrow = True)
        self.b = ParamItem(name='b',data=b, eps = eps_B, wc=wc_W)
        self.params.append(self.b)

        #symbolic computation
        self.update(input)

    def update(self, input):
        self.output = T.dot(input, self.W.data) + self.b.data
        self.p_y_given_x = T.nnet.softmax(self.output)
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

    def setPretrain(self, W=None, b=None):
        if W is not None:
            self.W.name = W.name
            self.W.data.set_value(W.data.get_value(), borrow=True)
            self.W.eps = W.eps
            self.W.wc = W.wc
        if b is not None:
            self.b.name = b.name
            self.b.data.set_value(b.data.get_value(), borrow=True)
            self.b.eps = b.eps
            self.b.wc = b.wc

    def cross_entropy(self, y, mask=None):
        first_part = T.log(self.p_y_given_x)[T.arange(y.shape[0]),y]
        second_part = T.sum(T.log(1-self.p_y_given_x),axis=1)-T.log(1-self.p_y_given_x)[T.arange(y.shape[0]),y]
        raw_output = -T.mean(first_part + second_part)
        if mask is None:
            return raw_output
        else:
            return -T.dot(
                    raw_output,
                    mask
                    ) / mask.sum()

    def nll(self, y, mask=None):
        if mask is None:
            return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        else :
            return -T.dot(
                        T.log(self.p_y_given_x)[T.arange(y.shape[0]),y],
                        mask
                    )/mask.sum()

    def errors(self, y, mask=None):
        assert y.ndim == 1
        if mask is None:
            return T.mean(T.neq(self.y_pred, y))
        else :
            ret = T.dot(
                        T.cast(T.neq(self.y_pred, y), theano.config.floatX),
                        mask
                    )/mask.sum()
            return ret

    def mse(self, y):
        if y.ndim == 1:
            print "labels shape:{}, y.ndim is 1".format(y.shape)
            diff = self.p_y_given_x[T.arange(y.shape[0]), y] - 1
            return T.mean(diff**2)
        elif y.ndim == 2:
            print "labels shape:{}, y.ndim is 2".format(y.shape)
            diff = self.p_y_given_x - y
            return T.mean(diff**2)
        else:
            print "Unexcepted dimension of labels:{}".format(y.ndim)


def conv_theano(input, image_shape, W, filter_shape, border_mode = 'same', stride = 1, group = 1):
    if group == 1:
        # smaller than the image
        # image_shape[2] - (filter_shape[2] - 1)
        if border_mode == 'valid':
            convOut = conv.conv2d(input = input,
                                  filters = W,
                                  border_mode = 'valid',
                                  image_shape = image_shape,
                                  filter_shape = filter_shape)
        # larger than the image
        # image_shape[2] + (filter_shape[2] - 1)
        elif border_mode == 'full':
            convOut = conv.conv2d(input = input,
                                  filters = W,
                                  border_mode = 'full',
                                  image_shape = image_shape,
                                  filter_shape = filter_shape)
        # same as the image, only when filter_shape[2] and filter_shape[3] both are odd numbers
        # image_shape[2]
        elif border_mode == 'same':
            assert filter_shape[2] % 2 == 1
            assert filter_shape[3] % 2 == 1
            bound = [filter_shape[2] / 2, filter_shape[3] / 2]
            convOut = conv.conv2d(input = input,
                                  filters = W,
                                  border_mode = 'full',
                                  image_shape = image_shape,
                                  filter_shape = filter_shape)
            convOut = convOut[:, :, bound[0]:-bound[0], bound[1]:-bound[1]]
        else:
            print 'Wrong border_mode'
            assert 1 == 0

        if stride > 1:
            convOut = convOut[:, :, ::stride, ::stride]
    else:
        convOuts = []
        for i in np.arange(group):
            if border_mode == 'valid':
                convOut = conv.conv2d(input = input[:, i * filter_shape[1]:(i + 1) * filter_shape[1], :, :],
                                      filters = W[i * filter_shape[0] / group:(i + 1) * filter_shape[0] / group, :, :, :],
                                      border_mode = 'valid',
                                      image_shape = (image_shape[0], filter_shape[1], image_shape[2], image_shape[3]),
                                      filter_shape = (filter_shape[0] / group, filter_shape[1], filter_shape[2], filter_shape[3]))
            elif border_mode == 'full':
                convOut = conv.conv2d(input = input[:, i * filter_shape[1]:(i + 1) * filter_shape[1], :, :],
                                      filters = W[i * filter_shape[0] / group:(i + 1) * filter_shape[0] / group, :, :, :],
                                      border_mode = 'full',
                                      image_shape = (image_shape[0], filter_shape[1], image_shape[2], image_shape[3]),
                                      filter_shape = (filter_shape[0] / group, filter_shape[1], filter_shape[2], filter_shape[3]))
            elif border_mode == 'same':
                assert filter_shape[2] % 2 == 1
                assert filter_shape[3] % 2 == 1
                bound = [filter_shape[2] / 2, filter_shape[3] / 2]
                convOut = conv.conv2d(input = input[:, i * filter_shape[1]:(i + 1) * filter_shape[1], :, :],
                                      filters = W[i * filter_shape[0] / group:(i + 1) * filter_shape[0] / group, :, :, :],
                                      border_mode = 'full',
                                      image_shape = (image_shape[0], filter_shape[1], image_shape[2], image_shape[3]),
                                      filter_shape = (filter_shape[0] / group, filter_shape[1], filter_shape[2], filter_shape[3]))
                convOut = convOut[:, :, bound[0]:-bound[0], bound[1]:-bound[1]]
            else:
                print 'Wrong border_mode'
                assert 1 == 0

            if stride > 1:
                convOut = convOut[:, :, ::stride, ::stride]
            convOuts.append(convOut)

        convOut = T.concatenate(convOuts, axis = 1)
    return convOut

#borrow from pylearn2
def avg_pool(bc01, pool_shape, pool_stride, image_shape):
    """
    Theano's max pooling op only supports pool_stride = pool_shape
    so here we have a graph that does max pooling with strides


    Parameters
    ----------
    bc01 : theano tensor
        minibatch in format (batch size, channels, rows, cols)
    pool_shape : tuple
        shape of the pool region (rows, cols)
    pool_stride : tuple
        strides between pooling regions (row stride, col stride)
    image_shape : tuple
        avoid doing some of the arithmetic in theano


    Returns
    -------
    pooled : theano tensor
        The output of pooling applied to `bc01`
    """
    avgP = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    assert pr <= r
    assert pc <= c

    name = bc01.name
    if name is None:
        name = 'anon_bc01'

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval

    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0],
                            pool_shape[0],
                            pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1],
                            pool_shape[1],
                            pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    # set the value of the given shape tensor
    wide_zero = T.alloc(T.constant(0, dtype=T.config.floatX),
                            bc01.shape[0],
                            bc01.shape[1],
                            required_r,
                            required_c)

    bc01 = T.set_subtensor(wide_zero[:, :, 0:r, 0:c], bc01)
    bc01.name = 'infinite_padded_' + name


    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = bc01[:,
                       :,
                       row_within_pool:row_stop:rs,
                       col_within_pool:col_stop:cs]
            cur.name = ('average_pool_cur_' + bc01.name + '_' +
                        str(row_within_pool) + '_' + str(col_within_pool))
            if avgP is None:
                avgP = cur
            else:
                avgP = avgP + cur
                avgP.name = ('average_pool_sum_' + bc01.name + '_' +
                           str(row_within_pool) + '_' + str(col_within_pool))


    avgP.name = 'average_pool('+name+')'
    avgP = avgP / (pc * pr)
    return T.cast(avgP, T.config.floatX)


#borrow from pylearn2
def max_pool(bc01, pool_shape, pool_stride, image_shape):
    """
    Theano's pooling op only supports max pooling and pool_stride = pool_shape
    so here we have a graph that does avg pooling with strides


    Parameters
    ----------
    bc01 : theano tensor
        minibatch in format (batch size, channels, rows, cols)
    pool_shape : tuple
        shape of the pool region (rows, cols)
    pool_stride : tuple
        strides between pooling regions (row stride, col stride)
    image_shape : tuple
        avoid doing some of the arithmetic in theano


    Returns
    -------
    pooled : theano tensor
        The output of pooling applied to `bc01`


    See Also
    --------
    max_pool_c01b : Same functionality but with ('c', 0, 1, 'b') axes
    sandbox.cuda_convnet.pool.max_pool_c01b : Same functionality as
        `max_pool_c01b` but GPU-only and considerably faster.
    mean_pool : Mean pooling instead of max pooling
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    assert pr <= r
    assert pc <= c


    name = bc01.name
    if name is None:
        name = 'anon_bc01'


    if pool_shape == pool_stride:
        mx = downsample.max_pool_2d(bc01, pool_shape, True)
        mx.name = 'max_pool('+name+')'
        return mx


    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0],
                            pool_shape[0],
                            pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr


    last_pool_c = last_pool(image_shape[1],
                            pool_shape[1],
                            pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc




    wide_infinity = T.alloc(T.constant(-np.inf, dtype=T.config.floatX),
                            bc01.shape[0],
                            bc01.shape[1],
                            required_r,
                            required_c)


    bc01 = T.set_subtensor(wide_infinity[:, :, 0:r, 0:c], bc01)
    bc01.name = 'infinite_padded_' + name


    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = bc01[:,
                       :,
                       row_within_pool:row_stop:rs,
                       col_within_pool:col_stop:cs]
            cur.name = ('max_pool_cur_' + bc01.name + '_' +
                        str(row_within_pool) + '_' + str(col_within_pool))
            if mx is None:
                mx = cur
            else:
                mx = T.maximum(mx, cur)
                mx.name = ('max_pool_mx_' + bc01.name + '_' +
                           str(row_within_pool) + '_' + str(col_within_pool))


    mx.name = 'max_pool('+name+')'
    return mx
