#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:12:06 2017

@author: ianlo
"""
from IPython.core.display import display

import threading
import mkl
mkl.set_num_threads(6)

import math
import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import re

import wordnetutils as wnu

import mpl_toolkits
from mpl_toolkits.mplot3d.axes3d import Axes3D

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus import brown

from nltk.stem.snowball import SnowballStemmer

from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer


from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import log_loss


# define stop word list
stops = set(stopwords.words('english'))

# initialise stemmer
snowball_stemmer = SnowballStemmer('english', ignore_stopwords=True)

# initiaise tokenizer
tokenizer = TreebankWordTokenizer().tokenize


class ThreadSafeDict(dict) :
    def __init__(self, * p_arg, ** n_arg) :
        dict.__init__(self, * p_arg, ** n_arg)
        self._lock = threading.Lock()

    def __enter__(self) :
        self._lock.acquire()
        return self

    def __exit__(self, type, value, traceback) :
        self._lock.release()




def stem_question(row, column):
    stemlist = [snowball_stemmer.stem(word) for word in tokenizer(row[column])]
    return stemlist




def lemm_question(row, column):
    vec = row[column].split(' ')
    return vec




def remove_stopwords(row, column):
    sent = ''
    for word in word_tokenize(row[column]):
        if word not in wnu.stpwrd:
            sent += ' ' + word
    return sent
#    wordset = {}
#    for word in word_tokenize(row[column]):
#        if word not in wnu.stpwrd:
#            wordset[word] = 1
#    sent = ' '.join(word for word in wordset.keys())
#    return sent




def tokenize_stem(text):
    tokens = tokenizer(text)
    stems = [snowball_stemmer.stem(word) for word in tokens]
    return stems




def word_match_share(row, columnname1, columnname2):
    q1words = {}
    q2words = {}
    for word in str(row[columnname1]).lower().split():
        if word not in wnu.stpwrd:
            q1words[word] = 1
    for word in str(row[columnname2]).lower().split():
        if word not in wnu.stpwrd:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = round((len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words)),5)
    return R




def stemmed_word_match_share(row, columnname1, columnname2, stopword=False):
    wordset1 = {}
    wordset2 = {}
    for word in stem_question(row, columnname1):
        if word not in wnu.stpwrd:
            wordset1[word] = 1
    for word in stem_question(row, columnname2):
        if word not in wnu.stpwrd:
            wordset2[word] = 1
    if len(wordset1) == 0 or len(wordset2) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_set1 = [w for w in wordset1.keys() if w in wordset2]
    shared_words_in_set2 = [w for w in wordset2.keys() if w in wordset1]
    R = round((len(shared_words_in_set1) + len(shared_words_in_set2))/(len(wordset1) + len(wordset2)),5)
    return R




def create_cooccurrence_matrix(docs, window_size, tokenizer):
    count_model = CountVectorizer(ngram_range=(1,window_size),
                                  strip_accents='unicode',
                                  tokenizer=tokenizer) # default unigram model
    X = count_model.fit_transform(docs)
    Xc = (X.T * X) # this is co-occurrence matrix in sparse csr format
    Xc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
    #print(Xc.todense()) # print out matrix in dense format
    return count_model.vocabulary_, Xc.todense()




def substitute_thousands(text):
    re.compile(r'[0-9]+(?P<thousands>\s{0,2}k\b)')
    matches = re.finditer(r'[0-9]+(?P<thousands>\s{0,2}k\b)', text, flags=re.I)
    result = ''
    len_offset = 0
    for match in matches:
        result += '{}000'.format(text[len(result)-len_offset:match.start('thousands')])
        len_offset += 3 - (match.end('thousands') - match.start('thousands'))
    result += text[len(result)-len_offset:]
    return result




def is_nan(x):
    try:
        return math.isnan(x)
    except:
        return False



def none_to_float(x):
    try:
        return float(0 if x is None else x)
    except:
        return 0
    


def to_32bit(df):
    for c in df.select_dtypes(include=['float64']):
        df[c] = df[c].astype(np.float32)
        
    for c in df.select_dtypes(include=['int64']):
        df[c] = df[c].astype(np.int32)
        
    return df




# Just make a convenience function; this one wraps the VarianceThreshold
# transformer but you can pass it a pandas dataframe and get one in return

def get_low_variance_columns(dframe=None, columns=None,
                             skip_columns=None, thresh=0.0,
                             autoremove=False):
    """
    Wrapper for sklearn VarianceThreshold for use on pandas dataframes.
    """
    print("Finding low-variance features.")
    try:
        # get list of all the original df columns
        all_columns = dframe.columns

        # remove `skip_columns`
        remaining_columns = all_columns.drop(skip_columns)

        # get length of new index
        max_index = len(remaining_columns) - 1

        # get indices for `skip_columns`
        skipped_idx = [all_columns.get_loc(column)
                       for column
                       in skip_columns]

        # adjust insert location by the number of columns removed
        # (for non-zero insertion locations) to keep relative
        # locations intact
        for idx, item in enumerate(skipped_idx):
            if item > max_index:
                diff = item - max_index
                skipped_idx[idx] -= diff
            if item == max_index:
                diff = item - len(skip_columns)
                skipped_idx[idx] -= diff
            if idx == 0:
                skipped_idx[idx] = item

        # get values of `skip_columns`
        skipped_values = dframe.iloc[:, skipped_idx].values

        # get dataframe values
        X = dframe.loc[:, remaining_columns].values

        # instantiate VarianceThreshold object
        vt = VarianceThreshold(threshold=thresh)

        # fit vt to data
        vt.fit(X)

        # get the indices of the features that are being kept
        feature_indices = vt.get_support(indices=True)

        # remove low-variance columns from index
        feature_names = [remaining_columns[idx]
                         for idx, _
                         in enumerate(remaining_columns)
                         if idx
                         in feature_indices]

        # get the columns to be removed
        removed_features = []
        removed_features = list(np.setdiff1d(remaining_columns,
                                             feature_names))
        print("Found {0} low-variance columns."
              .format(len(removed_features)))

        # remove the columns
        if autoremove:
            print("Removing low-variance features.")
            # remove the low-variance columns
            X_removed = vt.transform(X)

            print("Reassembling the dataframe (with low-variance "
                  "features removed).")
            # re-assemble the dataframe
            dframe = pd.DataFrame(data=X_removed,
                                  columns=feature_names)

            # add back the `skip_columns`
            for idx, index in enumerate(skipped_idx):
                dframe.insert(loc=index,
                              column=skip_columns[idx],
                              value=skipped_values[:, idx])
            print("Succesfully removed low-variance columns.")

        # do not remove columns
        else:
            print("No changes have been made to the dataframe.")

    except Exception as e:
        print(e)
        print("Could not remove low-variance features. Something "
              "went wrong.")
        pass

    return dframe, removed_features




def prepare_text(text):
    # prepare the text by ensuring that abbreviations are standardised
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
# the following are part of punctuations that will be removed eventually
#    text = re.sub(r",", " ", text)
#    text = re.sub(r"\.", " ", text)
#    text = re.sub(r"!", " ! ", text)
#    text = re.sub(r"\/", " ", text)
#    text = re.sub(r"\^", " ^ ", text)
#    text = re.sub(r"\+", " + ", text)
#    text = re.sub(r"\-", " - ", text)
#    text = re.sub(r"\=", " = ", text)
#    text = re.sub(r"'", " ", text)
#    text = re.sub(r":", " : ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " for example ", text)
    text = re.sub(r" e.g ", " for example ", text)
    text = re.sub(r" e.g. ", " for example ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text



##############################################################################
# Plotting sub-routine
##############################################################################

def plot_real_feature(fname):
    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 2), (2, 0))
    ax4 = plt.subplot2grid((3, 2), (2, 1))
    ax1.set_title('Distribution of %s' % fname, fontsize=20)
    sns.distplot(df.loc[ix_train][fname], 
                 bins=50, 
                 ax=ax1)    
    sns.distplot(df.loc[ix_is_dup][fname], 
                 bins=50, 
                 ax=ax2,
                 label='is dup')    
    sns.distplot(df.loc[ix_not_dup][fname], 
                 bins=50, 
                 ax=ax2,
                 label='not dup')
    ax2.legend(loc='upper right', prop={'size': 18})
    sns.boxplot(y=fname, 
                x='is_duplicate', 
                data=df.loc[ix_train], 
                ax=ax3)
    sns.violinplot(y=fname, 
                   x='is_duplicate', 
                   data=df.loc[ix_train], 
                   ax=ax4)
    plt.show()




def plot_3d_scatter(A, elevation=30, azimuth=120):
    """ Create 3D scatterplot """
    
    maxpts=1000
    fig = plt.figure(1, figsize=(9, 9))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=elevation, azim=azimuth)
    ax.set_xlabel('component 0')
    ax.set_ylabel('component 1')
    ax.set_zlabel('component 2')

    # plot subset of points
    rndpts = np.sort(np.random.choice(A.shape[0], min(maxpts,A.shape[0]), replace=False))
    coloridx = np.unique(A.iloc[rndpts]['Class'], return_inverse=True)
    colors = coloridx[1] / len(coloridx[0])   
    
    sp = ax.scatter(A.iloc[rndpts,0], A.iloc[rndpts,1], A.iloc[rndpts,2]
               ,c=colors, cmap="jet", marker='o', alpha=0.6
               ,s=50, linewidths=0.8, edgecolor='#BBBBBB')

    plt.show()



def pretty_scalings(lda, X, out=False):
    ret = pd.DataFrame(lda.scalings_, index=X.columns, columns=["LD"+str(i+1) for i in range(lda.scalings_.shape[1])])
    if out:
        print("Coefficients of linear discriminants:")
        display(ret)
    return ret



def rpredict(lda, X, y, out=False):
    ret = {"class": lda.predict(X),
           "posterior": pd.DataFrame(lda.predict_proba(X), columns=lda.classes_)}
    ret["x"] = pd.DataFrame(lda.fit_transform(X, y))
    ret["x"].columns = ["LD"+str(i+1) for i in range(ret["x"].shape[1])]
    if out:
        print("class")
        print(ret["class"])
        print()
        print("posterior")
        print(ret["posterior"])
        print()
        print("x")
        print(ret["x"])
    return ret



def ldahist(data, g, sep=False):
    xmin = np.trunc(np.min(data)) - 1
    xmax = np.trunc(np.max(data)) + 1
    ncol = len(set(g))
    binwidth = 0.5
    bins=np.arange(xmin, xmax + binwidth, binwidth)
    if sep:
        fig, axl = plt.subplots(ncol, 1, sharey=True, sharex=True)
    else:
        fig, axl = plt.subplots(1, 1, sharey=True, sharex=True)
        axl = [axl]*ncol
    for ax, (group, gdata) in zip(axl, data.groupby(g)):
        sns.distplot(gdata.values, bins, ax=ax, label="group "+str(group))
        ax.set_xlim([xmin, xmax])
        if sep:
            ax.set_xlabel("group"+str(group))
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()


##############################################################################
# Scratch pad
##############################################################################

#==============================================================================
# # get word co-occurance matrix and get the word vector
# token_list = []
# token_list.append(df_train.at[100,'q1nopunct'])
# token_list.append(df_train.at[100,'q2nopunct'])
# 
# mat = utils.create_cooccurrence_matrix(token_list, 1, utils.tokenizer)
# 
# la = np.linalg
# u, s, vh = la.svd(mat[1], full_matrices=True)
# 
# # get first column of u matrix from SVD
# u[:,:1]
# 
# words = mat[0].keys()
# 
# for i in range(len(words)):
#     plt.text(u[i,0], u[i,1], list(mat[0].keys())[i])
#==============================================================================




