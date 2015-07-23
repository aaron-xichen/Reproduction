#!/usr/bin/env python
# encoding: utf-8
# generate merge.pkl which converted from origianl TIMIT datasets

import os
import utils
import numpy as np

def walk_all(dir_name = os.path.expanduser("~/Documents/dataset/TIMIT")):
    wav_files = []
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith("wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

train_files = walk_all("/Users/cxlyc007/Documents/dataset/TIMIT/train/")
test_files = walk_all("/Users/cxlyc007/Documents/dataset/TIMIT/test/")

train_wavs = dict()
test_wavs = dict()

print "processing training data"
for file in train_files:
    wav_file = file
    ref_file = file[:len(file)-4] + ".wrd"
    wavs = utils.split_wav(wav_file, ref_file)
    for key,value in wavs.iteritems():
        if not train_wavs.has_key(key):
            train_wavs[key] = []
        train_wavs[key].extend(value)
train_nums = [len(train_wavs[each]) for each in train_wavs.keys()]
print "train total words:{}".format(np.asarray(train_nums).sum())

print "processing testing data"
for file in test_files:
    wav_file = file
    ref_file = file[:len(file)-4] + ".wrd"
    wavs = utils.split_wav(wav_file, ref_file)
    for key,value in wavs.iteritems():
        if not test_wavs.has_key(key):
            test_wavs[key] = []
        test_wavs[key].extend(value)
test_nums = [len(test_wavs[each]) for each in test_wavs.keys()]
print "test total words:{}".format(np.asarray(test_nums).sum())


print "mergeing"
joint = dict()
for key in train_wavs.keys():
    for word in train_wavs[key]:
        if not joint.has_key(key):
            joint[key] = []
        joint[key].append(word)

for key in test_wavs.keys():
    for word in test_wavs[key]:
        if not joint.has_key(key):
            joint[key] = []
        joint[key].append(word)
total_nums = 0
for key in joint.keys():
    total_nums = total_nums + len(joint[key])
    for word in joint[key]:
        if len(word) == 0:
            print key, word
print "merge total words:{}".format(total_nums)

utils.save_pickle('merge.pkl', joint)
