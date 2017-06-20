#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 00:07:31 2017

@author: ianlo
"""

import os
# set working directory
path = '/Users/ianlo/Documents/Kaggle/QuoraQuestionPairs/'
os.chdir(path)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import seaborn as sns

import scipy as scipy
import scipy.ndimage

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import nltk

from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


# customised imports
import global_settings as gs

from multiprocessing import cpu_count

# initialise global parameters
gs.init()

pal = sns.color_palette()


# check key package versions
print('NLTK:' + nltk.__version__)
print('Pandas:' + pd.__version__)
print('Scipy:' + scipy.__version__)


# set no of groups for partitioning
_number_of_groups = int(cpu_count()*0.8)

# set no of threads
_cpu = int(cpu_count()*0.8)



##############################################################################
# Prepare Training and Test Sets
##############################################################################

# Read the training and test set from HDFStore
trn = pd.DataFrame()

tr_store = pd.HDFStore('df_all_train_pres.h5')
nrows = tr_store.get_storer('df').nrows

#for chunk in pd.read_hdf('df_all_intermediate.h5','df', chunksize=chunksize, where='type = "TR"'):
for i in range(nrows//gs.chunksize + 1):
    chunk = tr_store.select('df', start=i * gs.chunksize, stop=(i+1) * gs.chunksize)
    trn = trn.append(chunk.loc[chunk['type'] == 'TR'])
    print("Finished reading", str(i), "chunks")

    del chunk
    
tr_store.close()

# garbage collection
gc.collect()


# remove index columns and columns that have possibly low predictive power - 
# which may negatively influence the classifiers
trn.drop(['grpId',
          #'id',
          'type',
          'test_id', #for trn data set, need to drop test_id as it is -9999 column
          'tr_q1WrdCnt_zscore',
          'tr_q2WrdCnt_zscore',
          'tr_q1Len_zscore',
          'tr_q2Len_zscore'
          ], axis=1, inplace = True)


# move label to the last column
trn = trn[[col for col in trn if col != 'is_duplicate'] + ['is_duplicate']]

# check for nan rows
#nan_rows = trn[trn.isnull().T.any().T]

# during the gen_semantic_similarity and gen_word_order_similarity there could
# have been null / NA values - hence need to set the value to 0.0
trn.loc[trn.word_order_similarity.isnull(), 'word_order_similarity'] = 0.0

# during the generation of the tfidf weight matrix there could
# have been null / NA values - hence need to set the value to 0.0
trn.loc[trn.weight_q1_sw.isnull(), 'weight_q1_sw'] = 0.0
trn.loc[trn.weight_q2_sw.isnull(), 'weight_q2_sw'] = 0.0
trn.loc[trn.weight_q1_mw.isnull(), 'weight_q1_mw'] = 0.0
trn.loc[trn.weight_q2_mw.isnull(), 'weight_q2_mw'] = 0.0

# set the column names of the data frame as str for easier referencing
trn.columns = trn.columns.astype(str)

# check for nans
#idx = trn.index[trn.isnull().all(1)]
#nans = trn.ix[idx]
#nans


##############################################################################
# Normalise features
##############################################################################

# normalise the word count for specific columns
norm_col = ['tr_q1WrdCnt',
            'tr_q2WrdCnt',
            'tr_q1Len',
            'tr_q2Len',
            'weight_q1_sw',
            'weight_q2_sw']

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
trn_scaled = pd.DataFrame(scaler.fit_transform(trn[norm_col]))
trn_scaled.columns = ['scaled_' + str(i) for i in trn_scaled.columns]

for i in trn_scaled.columns: trn[i] = trn_scaled[i].values

del trn_scaled

#trn.iloc[403711]
#trn = trn[trn['id'] == 117684]





# get columns based on index and specific columns
cols = []
cols = list(trn.ix[:,trn.columns.get_loc('0'):(trn.columns.get_loc('783')+1)].columns)
cols.append('wrdmatchpct')
cols.append('sem_similarity')
cols.append('word_order_similarity')
#cols.append('scaled_0')
cols.append('scaled_1')
cols.append('scaled_2')
cols.append('scaled_3')
cols.append('scaled_4')
cols.append('scaled_5')

arr = trn.ix[1:1,cols], trn.ix[1:1,'is_duplicate']

# create matrix based on current dimension
img = np.resize(arr[0], (24, 33))
#img = np.resize(arr[0], (28, 28))

# zoom the matrix to fit 28 x 28 image
img = scipy.ndimage.interpolation.zoom(img,
                                       #((matrix_size**2) / (length**2)),
                                       (1 / 1),
                                       order = 1,
                                       mode = 'nearest').round(5)

#img = np.resize(arr[0], (28, 28))
plt.figure(figsize=(5, 3), dpi=100)
disp = plt.imshow(img, interpolation='nearest', cmap=cm.coolwarm)
# make a color bar
plt.tight_layout()
plt.grid(False)
disp.axes.get_xaxis().set_visible(False)
disp.axes.get_yaxis().set_visible(False)
plt.savefig('myfig', dpi = 200)
plt.colorbar(disp)
plt.show()








##############################################################################
# Train and Build Model(s)
##############################################################################


# Negative / Positive cases oversampling to balance the data based on the testing set
##############################################################################

# Approx 36.9% are positive labels
pos_trn = trn[trn['is_duplicate'].values == 1]
neg_trn = trn[trn['is_duplicate'].values == 0]


# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_trn) / (len(pos_trn) + len(neg_trn))) / p) - 1

while scale > 1:
    neg_trn = pd.concat([neg_trn, neg_trn])
    #pos_trn = pd.concat([pos_trn, pos_trn])
    scale -=1

neg_trn = pd.concat([neg_trn, neg_trn[:int(scale * len(neg_trn))]])
#pos_trn = pd.concat([pos_trn, pos_trn[:int(scale * len(pos_trn))]])
print(len(pos_trn) / (len(pos_trn) + len(neg_trn)))


# separate features and labels based on the new training dataset
x_train = pd.concat([pos_trn, neg_trn])
y_train = (np.zeros(len(pos_trn)) + 1).tolist() + np.zeros(len(neg_trn)).tolist()



# Apply one hot encoding on the target variable
##############################################################################
# one hot encoding - sklearn.preprocessing.OneHotEncoder

enc = OneHotEncoder()
enc.fit(np.array(y_train).reshape(-1,1))  

x = pd.DataFrame(enc.transform(np.array(y_train).reshape(-1,1)).toarray())
y_train = pd.DataFrame(y_train, columns=['is_duplicate'])
y_train['is_dup_0'] = x[0].values
y_train['is_dup_1'] = x[1].values


# remove label columns from x_train. Don't change y_train as it is already the label list
x_train.drop(['is_duplicate'], axis=1, inplace = True)
#x_train.drop(['is_dup_0'], axis=1, inplace = True)
#x_train.drop(['is_dup_1'], axis=1, inplace = True)

# remove the old target variable now that we have one hot encoded the variable
y_train.drop(['is_duplicate'], axis=1, inplace = True)

# training / test set split 80/20 based on random selection
# 4242, 12357
x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                      y_train,
                                                      test_size = 0.3,
                                                      random_state = gs.seedvalue)

# free up 
del [pos_trn, neg_trn, x]

# garbage collection
gc.collect()






##############################################################################
# TensorFlow modelling
##############################################################################

import tensorflow as tf
sess = tf.Session()

IMAGE_SIZE_X = 24
IMAGE_SIZE_Y = 33
NUM_CHANNELS = 1
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 100
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 100
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


# select features
includeCols = []
includeCols = list(x_train.ix[:,x_train.columns.get_loc('0'):(x_train.columns.get_loc('783')+1)].columns)
includeCols.append('wrdmatchpct')
includeCols.append('sem_similarity')
includeCols.append('word_order_similarity')
#includeCols.append('scaled_0')
includeCols.append('scaled_1')
includeCols.append('scaled_2')
includeCols.append('scaled_3')
includeCols.append('scaled_4')
includeCols.append('scaled_5')



# assign training and validation data for tensorflow
x_trndata = x_train.loc[:, includeCols]
y_trndata = pd.DataFrame(y_train)

x_validdata = x_valid.loc[:, includeCols]
y_validdata = pd.DataFrame(y_valid)

train_size = y_trndata.shape[0]



# ---- CNN ----------------------------------------------------------------

# placeholder X serves as a target for feeds. It is not initialised and
# contains no data.
# None to our placeholder, it means the placeholder can be fed as many examples
# as you want to give it. In this case, our placeholder can be fed any multitude
# of 784-sized values
x = tf.placeholder(tf.float32, shape = [None, IMAGE_SIZE_X * IMAGE_SIZE_Y])


# note that for MINST the label is in the form of a vector of 1 floats
# 0 - 9 where the number is labelled as 1 in the corresponding index
y_ = tf.placeholder(tf.float32, shape = [None,2])



# good practice to initialize them with a slightly positive initial bias to avoid "dead neurons"
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, seed=SEED, dtype=tf.float32)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



# To apply the layer, we first reshape x to a 4d tensor, with the second and third
# dimensions corresponding to image width and height, and the final dimension corresponding
# to the number of color channels.
# -1 means that we take all the number of samples
# 28,28,1 means we reshape to a 28x28 matrix with 1 colour channel
x_image = tf.reshape(x, [-1,IMAGE_SIZE_X,IMAGE_SIZE_Y,1])

# -----------------------------------------------------------------------------
# first convolutional layer

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

# The max_pool_2x2 method will reduce the image size to 14x14.
h_pool1 = max_pool_2x2(h_conv1)


# -----------------------------------------------------------------------------
# second convolutional layer

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

# The max_pool_2x2 method will reduce the image size to 7x7.
h_pool2 = max_pool_2x2(h_conv2)


# -----------------------------------------------------------------------------
# third convolutional layer

W_conv3 = weight_variable([5, 5, 64, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

# The max_pool_2x2 method will reduce the image size to 4x4.
h_pool3 = max_pool_2x2(h_conv3)


# -----------------------------------------------------------------------------
# dense fully connected layer

# we add a fully-connected layer with 1024 neurons to allow processing on the entire image
W_fc1 = weight_variable([3 * 5 * 64, 960])
b_fc1 = bias_variable([960])

# We reshape the tensor from the pooling layer into a batch of vectors
h_pool3_flat = tf.reshape(h_pool3, [-1, 3 * 5 * 64])

# multiply by a weight matrix, add a bias, and apply a ReLU.
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)


# -----------------------------------------------------------------------------
# dropout layer

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, seed=SEED)

# -----------------------------------------------------------------------------
# readout layer

W_fc2 = weight_variable([960, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# -----------------------------------------------------------------------------
# Train and Evaluate the Model

# Training computation: logits + cross-entropy loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))


# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))

# Add the regularization term to the cross_entropy.
cross_entropy += 5e-4 * regularizers


# Evaluate different optimizers
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
batch = tf.Variable(0, dtype=tf.float32)

# Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(0.01,                # Base learning rate
                                           batch * BATCH_SIZE,  # Current index into the dataset.
                                           train_size,          # Decay step.
                                           0.96,                # Decay rate.
                                           staircase=True)

# Use simple momentum for the optimization.
#train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy, global_step=batch)
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=batch)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, global_step=batch)


# evaluation criteron
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

# calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# -----------------------------------------------------------------------------
# Define summaries to display on tensorboard



# initialise variables
sess.run(tf.global_variables_initializer())



# Training model run
for step in range(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
    # Compute the offset of the current minibatch in the data.
    # Note that we could use better randomization across epochs.
    offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    batch_data = x_trndata.iloc[offset:(offset + BATCH_SIZE)]
    batch_labels = y_trndata.iloc[offset:(offset + BATCH_SIZE)]
    
    if step%EVAL_FREQUENCY == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch_data, y_: batch_labels, keep_prob: 1.0})
        error = cross_entropy.eval(session=sess, feed_dict={x:batch_data, y_: batch_labels, keep_prob: 1.0})
        print("step %d, training accuracy %g %g"%(step, train_accuracy, error))
    
    train_step.run(session=sess, feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})



# Validation of training model run
start = 0
end = 0

for i in range(1, round(len(x_validdata)/BATCH_SIZE)-1):
    #  batch = mnist.train.next_batch(50)
    start = end
    end = i*BATCH_SIZE
    batch = (np.array(x_validdata.iloc[start:end]), np.array(y_validdata.iloc[start:end]))
    
    if i%EVAL_FREQUENCY == 0:
        test_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, test accuracy %g"%(i, test_accuracy))









