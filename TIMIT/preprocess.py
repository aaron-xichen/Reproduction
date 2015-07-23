#!/usr/bin/env python
# encoding: utf-8

# mfcc will stride 10ms (160 frames)
# 6101 clasees
# 45312 training samples
# 9063 testing samples

import numpy as np
import utils
from scikits.talkbox import features as fea
import PIL.Image as Image
import os
import time

nwin = 256
nceps = 40
is_shuffle = True
train_test_ratio = 5. / 1.

merge_path = os.path.expanduser('merge.pkl')
keys2index_path = os.path.expanduser('keys2index.pkl')
train_lmdb_path = os.path.expanduser('train')
test_lmdb_path = os.path.expanduser('test')

begin = time.clock()
keys2index = dict()
features = []
labels = []

print "loading..."
merge_set = utils.load_pickle(merge_path)
print "cost {}s".format((time.clock()-begin))

print "converting..."
for key in merge_set.keys():
    keys2index[key] = len(keys2index)
    for word in merge_set[key]:
        if len(word) >= nwin:
            mel = fea.mfcc(word, nceps=nceps)[0]
            im = Image.fromarray(mel).resize((nceps, nceps), Image.BICUBIC)
            im_ndarray = np.array(im.getdata())
            features.append(im_ndarray)
            labels.append(keys2index[key])
print "cost {}s".format((time.clock()-begin))

features = np.asarray(features)
labels = np.asarray(labels)
assert len(features) == len(labels)

if is_shuffle:
    print "shuffling..."
    perm = np.random.permutation(len(features))
    features = features[perm]
    labels = labels[perm]
    print "cost {}s".format((time.clock()-begin))

print "spliting...."
train_nums = train_test_ratio / (train_test_ratio + 1) * len(features)
train_features = features[:train_nums]
train_labels = labels[:train_nums]
test_features = features[train_nums:]
test_labels = labels[train_nums:]
print "cost {}s".format((time.clock()-begin))

print "construing train lmdb dataset"
utils.np2lmdb(train_features, train_lmdb_path, labels=train_labels)
print "cost {}s".format((time.clock()-begin))
print "construing test lmdb dataset"
utils.np2lmdb(test_features, test_lmdb_path, labels=test_labels)
print "cost {}s".format((time.clock()-begin))

print "saving keys2index..."
utils.save_pickle(keys2index_path, keys2index)
print "cost {}s".format((time.clock()-begin))
