#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 22:38:30 2017

@author: ianlo
"""

##############################################################################
# Setup Gensim model for word2vec
# - using a dict will ensure that the question list is unique
##############################################################################

import global_settings as gs
import utils as utils
import itertools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

import scipy as scipy
import scipy.ndimage

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from gensim.models import KeyedVectors
from gensim.models import Word2Vec


# 0) Do a word by word similiarity score
def create_w2vmodel(df_all = None):

    if (df_all is not None):
        # remove records with null questions so that won't process unecessary data
        df_all = df_all[df_all['q1nopunct'].notnull()]
        df_all = df_all[df_all['q2nopunct'].notnull()]

        qstns = dict()
        # IMPT - we only use the training rows to create the Word2Vec model
        for row in df_all[df_all.type == 'TR'].iterrows():
            qstns[row[1]['qid1']] = row[1]['q1nopunct']
            qstns[row[1]['qid2']] = row[1]['q2nopunct']
        
        # note that Gensim implementation of word2vec requires a list of words not just
        # the sentence as a string.
        qstnsList = []
        for i in qstns:
            #qstnsList.append(list(map(lambda x: x.lower(), utils.tokenizer(qstns[i]))))
            if (len(qstns[i]) > 0 ):
                qstnsList.append(utils.tokenizer(qstns[i]))
        
        # create Word2Vec model from current sentence set
        model = Word2Vec(qstnsList,
                         size = 200,
                         window = 5,
                         min_count = 3,
                         workers = 6,
                         seed = gs.seedvalue)
    else:
        # create Word2Vec model from Google News
        model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

    return model


# 1) Do a word by word similiarity score
def word_word_score(w1, w2):
    try:
        return gs.word2vecmodel.similarity(w1, w2)
    except Exception:
        return 0.0


# 2) Create a matrix between the similarity score of both questions and visualise it
def to_image(row, col1, col2, matrix_size, order, show=False, tofile=False):
    if (utils.is_nan(row[col1]) == True):
        c1tokens = []
    else:
        c1tokens = list(map(lambda x: x.lower(), utils.tokenizer(row[col1])))

    if (utils.is_nan(row[col2]) == True):
        c2tokens = []
    else:
        c2tokens = list(map(lambda x: x.lower(), utils.tokenizer(row[col2])))

    score = [word_word_score(a, b) for a, b in itertools.product(c1tokens, c2tokens)]
    # for questions with null values, score will be empty array so need to preset value to 0.0
    if (len(score) == 0):
        score = [0.0]
    arr = np.array(score, order='C')
    # determine the current dimensions   
    arrsize = len(arr)
    length = math.ceil(math.sqrt(len(arr)))
    # create matrix based on current dimension
    img = np.resize(arr, (length, length))
    #print('Row: {0}, Orig matrix length: {1}, Sqrt: {2}, Zoom: {3}'.format(row["id"], arrsize, length, ((matrix_size**2) / (length**2))))

    # zoom the matrix to fit 28 x 28 image
    img = scipy.ndimage.interpolation.zoom(img,
                                           #((matrix_size**2) / (length**2)),
                                           (matrix_size / length),
                                           order = order,
                                           mode = 'nearest').round(5)

    if (row['grpId'] == 0):
        if show:
            display = img
            # print img
            #fig = plt.figure()
            # tell imshow about color map so that only set colors are used
            display = plt.imshow(display, interpolation='nearest', cmap=cm.coolwarm)
            # make a color bar
            plt.colorbar(display)
            plt.grid(False)
            plt.text(0, -3, 'Is Dup:{0}'.format(row['is_duplicate']), ha='left', rotation=0, wrap=True, fontsize=10)
            plt.text(0, -2, 'Q1:{0}'.format(row[col1]), ha='left', rotation=0, wrap=True, fontsize=10)
            plt.text(0, -1, 'Q2:{0}'.format(row[col2]), ha='left', rotation=0, wrap=True, fontsize=10)
            if tofile:
                plt.savefig('./img/img_{0}'.format(row['id']), dpi = 100)
            else:
                plt.show()
            
            plt.clf()
            plt.cla()
            plt.close()
            #print('Orig matrix length: {0}, Sqrt: {1}, Zoom: {2}'.format(arrsize, length, ((matrix_size**2) / (length**2))))
            #print('New matrix length: {0}, Sqrt: {1}'.format(len(img.flatten()), math.ceil(math.sqrt(len(img.flatten())))))

    # important to set the return as a list
    return [img.flatten()]


#df_x = df_all[1:2]
#df_x = df_x.apply(lambda x: to_image(x, "q1nopunct", "q2nopunct", 28, 1, show=True), axis=1, raw=True)
#word_word_score('', 'how')
#df_x.dtypes


# wrapper function for parallel apply
# this function will also add 784 features to the data frame
def gen_img_feat(df, dest_col_ind, dest_col_name, col1, col2, matrix_size, order, show, tofile):
    df.insert(dest_col_ind, dest_col_name, df.apply(lambda x: to_image(x, col1, col2, matrix_size, order, show, tofile), axis=1, raw=True))
    df = pd.concat([df.drop([dest_col_name], axis=1), df[dest_col_name].str[0].apply(pd.Series)], axis=1) 
    return df



