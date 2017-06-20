#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 13:20:19 2017

@author: ianlo
"""

import os
# set working directory
path = '/Users/ianlo/Documents/Kaggle/QuoraQuestionPairs/'
os.chdir(path)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import re
import string
import seaborn as sns
import pickle
import scipy as scipy
import scipy.ndimage
import nltk
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from multiprocessing import cpu_count

# import custom files
import utils as utils
import img_feat_gen as ifg
import global_settings as gs
import pca as pca
from parallelproc import applyParallel

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


# import training data as dataframe 
df_train = pd.read_csv('./train.csv')


##############################################################################
# basic EDA
##############################################################################

print('Total number of question pairs for training: {}'.format(len(df_train)))
# no of positive training examples
print('Total number of duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean() * 100,2)))


qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
print('Total number of questions in the training data: {}'.format(len(np.unique(qids))))
# Any duplicate questions in the list?
print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

# plot the histogram of the no. of questions that appear multiple times
plt.figure(figsize=(12, 5))
plt.hist(qids.value_counts(), bins=50)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
print()


# start looking at the questions
# get all the questions from the train dataset
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

# get the counts of words for each question in in the training and test set
dist_train = train_qs.apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True, label='train')
plt.title('Normalised histogram of word count in questions', fontsize=15)
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Probability', fontsize=15)
print()

del [dist_train]



##############################################################################
# Initial data preparation
##############################################################################

# add label to mark training / test data
df_train['type'] = 'TR'


from sklearn.cross_validation import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(df_train, df_train['is_duplicate'], test_size = 0.5, random_state = gs.seedvalue)


# append training and test data so that all processing will be done together but
# the training and testing will be done separately
df_all = x_train

# release df_train
del [df_train]


# resequence all the columns for easy processing
colseq = ['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate', 'type']
df_all = df_all[colseq]

# add grpId so as to mark records for parallel processing
df_all.insert(0,'grpId',df_all.apply(lambda row: row.name % _number_of_groups, axis=1, raw=True))


# Data cleaning

# fill na's with '' so as to avoid
df_all.question1 = df_all.question1.fillna('')
df_all.question2 = df_all.question2.fillna('')

# remove records with null questions for this example only
df_all = df_all[df_all['question1'].notnull()]
df_all = df_all[df_all['question2'].notnull()]


# convert specifc columns to appropriate data type to save memory as well as to set default values
df_all.grpId = df_all.grpId.astype(np.int32)
df_all.id = df_all.id.astype(np.int32)
df_all.qid1 = df_all.qid1.astype(np.int32)
df_all.qid2 = df_all.qid2.astype(np.int32)
df_all.is_duplicate = df_all.is_duplicate.astype(np.int32)


# setup indexes for easy accessing later
ix_train = np.where(df_all['id'] >= 0)[0]
ix_is_dup = np.where(df_all['is_duplicate'] == 1)[0]
ix_not_dup = np.where(df_all['is_duplicate'] == 0)[0]



def prepare_text(df, dest_col_ind, dest_col, src_col):
    if (src_col == dest_col):
        # apply function in place
        df[dest_col] = df.apply(lambda x: utils.prepare_text(x[src_col]), axis=1, raw=True).values
    else:
        df.insert(dest_col_ind, dest_col, df.apply(lambda x: utils.prepare_text(x, src_col), axis=1, raw=True))
    return df

print('Starting pre-processing: Basic text cleaning')
df_all = applyParallel(df_all.groupby(df_all.grpId), prepare_text, {"dest_col_ind": -1, "dest_col": "question1", "src_col": "question1"}, _cpu)
df_all = applyParallel(df_all.groupby(df_all.grpId), prepare_text, {"dest_col_ind": -1, "dest_col": "question2", "src_col": "question2"}, _cpu)



# define parallel function. Please note that this function is NOT row-wise but chunk wise parallelisim
# Change all to lower case so that frequencies will be calculated correctly
def remove_punct(df, dest_col_ind, dest_col, src_col):
    df.insert(dest_col_ind, dest_col, df.apply(lambda x: re.sub('['+string.punctuation+'’'+'‘'+']', '',x[src_col]).lower(), axis=1, raw=True))
    return df

print('Starting pre-processing: Remove punctuations')
df_all = applyParallel(df_all.groupby(df_all.grpId), remove_punct, {"dest_col_ind": df_all.shape[1]-1, "dest_col": "q1nopunct", "src_col": "question1"}, _cpu)
df_all = applyParallel(df_all.groupby(df_all.grpId), remove_punct, {"dest_col_ind": df_all.shape[1]-1, "dest_col": "q2nopunct", "src_col": "question2"}, _cpu)



# substitute_thousands values
def substitute_thousands(df, dest_col_ind, dest_col, src_col):
    #df.insert(dest_col_ind, dest_col, df.apply(lambda x: utils.substitute_thousands(x[src_col]), axis=1, raw=True))
    df[dest_col] = df.apply(lambda x: utils.substitute_thousands(x[src_col]), axis=1, raw=True).values
    return df

print('Starting pre-processing: Substitute ''K'' with 000')
df_all = applyParallel(df_all.groupby(df_all.grpId), substitute_thousands, {"dest_col_ind": df_all.shape[1]-1, "dest_col": "q1nopunct", "src_col": "q1nopunct"}, _cpu)
df_all = applyParallel(df_all.groupby(df_all.grpId), substitute_thousands, {"dest_col_ind": df_all.shape[1]-1, "dest_col": "q2nopunct", "src_col": "q2nopunct"}, _cpu)



def remove_stopwords(df, dest_col_ind, dest_col, src_col):
    if (src_col == dest_col):
        # apply function in place
        df[dest_col] = df.apply(lambda x: utils.remove_stopwords(x, src_col), axis=1, raw=True).values
    else:
        df.insert(dest_col_ind, dest_col, df.apply(lambda x: utils.remove_stopwords(x, src_col), axis=1, raw=True))
    return df

print('Starting pre-processing: Remove stop words')
df_all = applyParallel(df_all.groupby(df_all.grpId), remove_stopwords, {"dest_col_ind": df_all.shape[1]-1, "dest_col": "q1nopunct_stem", "src_col": "q1nopunct"}, _cpu)
df_all = applyParallel(df_all.groupby(df_all.grpId), remove_stopwords, {"dest_col_ind": df_all.shape[1]-1, "dest_col": "q2nopunct_stem", "src_col": "q2nopunct"}, _cpu)



def stem(df, dest_col_ind, dest_col, src_col):
    if (src_col == dest_col):
        # apply function in place
        df[dest_col] = df.apply(lambda x: utils.stem_question(x, src_col), axis=1, raw=True).values
    else:
        df.insert(dest_col_ind, dest_col, df.apply(lambda x: utils.stem_question(x, src_col), axis=1, raw=True))
    return df

print('Starting pre-processing: Stem remaining words to another column')
df_all = applyParallel(df_all.groupby(df_all.grpId), stem, {"dest_col_ind": df_all.shape[1]-1, "dest_col": "q1nopunct_stem", "src_col": "q1nopunct_stem"}, _cpu)
df_all = applyParallel(df_all.groupby(df_all.grpId), stem, {"dest_col_ind": df_all.shape[1]-1, "dest_col": "q2nopunct_stem", "src_col": "q2nopunct_stem"}, _cpu)





# plot the distribution of no of words between question1 and question2
# get all the questions from the train dataset
df_train = df_all[df_all['type'].values == 'TR']

train_qs = pd.Series(df_train['q1nopunct'].tolist() + df_train['q2nopunct'].tolist()).astype(str)

# get the counts of words for each question in in the training and test set
dist_train = train_qs.apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(15, 10))
plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True, label='train')
plt.title('Normalised histogram of word count in questions', fontsize=15)
plt.legend()
plt.xlabel('Number of words', fontsize=15)
plt.ylabel('Probability', fontsize=15)
print()

del [dist_train, train_qs]




# instead of remove the records with '', we replace it with NULL so that we capture the records
# as negative cases
#df_train.loc[df_train.q1nopunct == '', 'q1nopunct'] = 'NULL'

# remove records with null / '' questions after removal of punctuation
#df_all = df_all[(df_all['q1nopunct'].notnull()) & (df_all['q1nopunct'] != '')]
#df_all = df_all[(df_all['q2nopunct'].notnull()) & (df_all['q2nopunct'] != '')]

# df_train[(df_train['q1nopunct'] == '')]
# df_train[(df_train['id'] == 13016)]


# drop original question as well as nopunct variants to reduce df size
df_all.drop(['question1', 'question2'], inplace=True, axis=1)

print(list(df_all.columns.values))



##############################################################################
# Basic feature generation
#
##############################################################################

print('Starting basic feature generation')

##############################################################################
# initial text processing for feature generation refer to utils package
##############################################################################


# get count of words in each question
def word_count(df, dest_col_ind, dest_col, src_col):
    df.insert(dest_col_ind, dest_col, df.apply(lambda x: len(x[src_col].split(' ')), axis=1, raw=True))
    return df

df_all = applyParallel(df_all.groupby(df_all.grpId), word_count, {"dest_col_ind": df_all.shape[1]-1,
                                                                  "dest_col": "tr_q1WrdCnt",
                                                                  "src_col": "q1nopunct"}, _cpu)    
    
df_all = applyParallel(df_all.groupby(df_all.grpId), word_count, {"dest_col_ind": df_all.shape[1]-1,
                                                                  "dest_col": "tr_q2WrdCnt",
                                                                  "src_col": "q2nopunct"}, _cpu)
print('Starting feature-gen: Done get word count')



# get count of characters in each question
def char_count(df, dest_col_ind, dest_col, src_col):
    df.insert(dest_col_ind, dest_col, df.apply(lambda x: np.uint32(len(x[src_col])),axis=1, raw=True))
    return df

df_all = applyParallel(df_all.groupby(df_all.grpId), char_count, {"dest_col_ind": df_all.shape[1]-1,
                                                                  "dest_col": "tr_q1Len",
                                                                  "src_col": "q1nopunct"}, _cpu)
    
df_all = applyParallel(df_all.groupby(df_all.grpId), char_count, {"dest_col_ind": df_all.shape[1]-1,
                                                                  "dest_col": "tr_q2Len",
                                                                  "src_col": "q2nopunct"}, _cpu)
print('Starting feature-gen: Done get character count')



# normalise the word count
for col in ['tr_q1WrdCnt','tr_q2WrdCnt']:
    col_zscore = col + '_zscore'
    df_all[col_zscore] = round((df_all[col] - df_all[col].mean())/df_all[col].std(ddof=0), _cpu)


# normalise the character count
for col in ['tr_q1Len','tr_q2Len']:
    col_zscore = col + '_zscore'
    df_all[col_zscore] = round((df_all[col] - df_all[col].mean())/df_all[col].std(ddof=0), _cpu)

print('Starting feature-gen: Done normalising the counts')



#==============================================================================
# # percentage of matching stemmed words
# #stemmed_word_match = df_train.apply(lambda x: utils.stemmed_word_match_share(x, stopword=True), axis=1, raw=True)
# def stemmed_word_match(df, dest_col_ind, dest_col, columnname1, columnname2):
#     df.insert(dest_col_ind, dest_col, df.apply(lambda x: utils.stemmed_word_match_share(x, columnname1, columnname2, stopword=True), axis=1, raw=True))
#     return df
# 
# df_all = applyParallel(df_all.groupby(df_all.grpId), stemmed_word_match, {"dest_col_ind": df_all.shape[1]-1,
#                                                                           "dest_col": "stem_wrdmatchpct",
#                                                                           "columnname1": "q1nopunct",
#                                                                           "columnname2": "q2nopunct"}, _cpu)
# 
# print('Starting feature-gen: Done computing percentage of matching stemmed words')
# 
#==============================================================================


def word_match_share(df, dest_col_ind, dest_col, columnname1, columnname2):
    df.insert(dest_col_ind, dest_col, df.apply(lambda x: utils.word_match_share(x, columnname1, columnname2), axis=1, raw=True))
    return df

df_all = applyParallel(df_all.groupby(df_all.grpId), word_match_share, {"dest_col_ind": df_all.shape[1]-1,
                                                                        "dest_col": "wrdmatchpct",
                                                                        "columnname1": "q1nopunct",
                                                                        "columnname2": "q2nopunct"}, _cpu)

print('Starting feature-gen: Done computing percentage of matching words')




plt.figure(figsize=(15, 5))
train_word_match = df_all['wrdmatchpct']
plt.hist(train_word_match[df_all['is_duplicate'] == 0], bins=25, normed=True, label='Not Duplicate')
plt.hist(train_word_match[df_all['is_duplicate'] == 1], bins=25, normed=True, alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over wrdmatchpct', fontsize=15)
plt.xlabel('wrdmatchpct', fontsize=15)



# convert rest of columns to 32bit
df_all = utils.to_32bit(df_all)
#df_x = df_all
#df_x = df_x.set_index(['id', 'test_id', 'type'])
#df_all = df_all.set_index(['id', 'test_id', 'type'])




##############################################################################
# Get pre-generated wordnet gen_semantic_similarity and gen_word_order_similarity 
# scores and append to the csv file
##############################################################################

df_temp = pd.DataFrame()

# open training h5 file
tr_store = pd.HDFStore('df_all_train.0430.h5', mode='r')
nrows = tr_store.get_storer('df').nrows

for i in range(nrows//gs.chunksize + 1):
    chunk = tr_store.select('df', start=i * gs.chunksize, stop=(i+1) * gs.chunksize)
    df_temp = df_temp.append(chunk[['id', 'test_id', 'type', 'sem_similarity', 'word_order_similarity']])    
    
    print("Finished concatenating *_similarity", str(i), "chunks from training data")

    del chunk

# must always close the HDFStore
tr_store.close()


# join the *_similarity scores with the original dataframe before persisting it
df_all = pd.merge(left=df_all, right=df_temp, how='left', left_on=['id', 'type'], right_on=['id', 'type'])


del df_temp

# garbage collection
gc.collect()






# Feature Generation
##############################################################################

##############################################################################
# Use TfIdf score to set the weight of terms that are significant and hence if they
# appear in one sentence and not the other, it will be able to differentiate
##############################################################################


# create corpus for tfidf vectoriser
corpus = df_all['q1nopunct'].append(df_all['q2nopunct'], ignore_index=True)

# create tf-idf vectoriser to get word weightings for sentence
tf = TfidfVectorizer(tokenizer=utils.tokenize_stem,
                     analyzer='word',
                     ngram_range=(1,2),
                     stop_words = 'english',
                     min_df = 0)

# initialise the tfidf vecotrizer with the corpus to get the idf of the corpus
tfidf_matrix =  tf.fit_transform(corpus)
#feature_names = tf.get_feature_names()

# remove corpra once the tf object has been fitted to reduce memory size
del corpus


# using the source corpus idf, create the idf from the input text
tfidf_matrix_q1 =  tf.transform(df_all['q1nopunct'])
tfidf_matrix_q2 =  tf.transform(df_all['q2nopunct'])


#Converting the sparse matrices into dataframes
transformed_matrix_1 = tfidf_matrix_q1.tocoo(copy = False)
weights_dataframe_1 = pd.DataFrame({'index': transformed_matrix_1.row,
                                    'term_id': transformed_matrix_1.col,
                                    'weight_q1': transformed_matrix_1.data})[['index', 'term_id', 'weight_q1']].sort_values(['index', 'term_id']).reset_index(drop = True)

sum_weights_1 = weights_dataframe_1.groupby('index').sum()
mean_weights_1 = weights_dataframe_1.groupby('index').mean()


transformed_matrix_2 = tfidf_matrix_q2.tocoo(copy = False)
weights_dataframe_2 = pd.DataFrame({'index': transformed_matrix_2.row,
                                    'term_id': transformed_matrix_2.col,
                                    'weight_q2': transformed_matrix_2.data})[['index', 'term_id', 'weight_q2']].sort_values(['index', 'term_id']).reset_index(drop = True)


sum_weights_2 = weights_dataframe_2.groupby('index').sum()
mean_weights_2 = weights_dataframe_2.groupby('index').mean()


# join the matrices into dataframe
weights = sum_weights_1 \
            .join(sum_weights_2, how = 'outer', lsuffix = '_q1', rsuffix = '_q2')
#            .join(sum_weights_3, how = 'outer', lsuffix = '_cw', rsuffix = '_cw')

mean_weights = mean_weights_1 \
            .join(mean_weights_2, how = 'outer', lsuffix = '_q1', rsuffix = '_q2')

# remove working columns
del weights['term_id_q1'], weights['term_id_q2']
del mean_weights['term_id_q1'], mean_weights['term_id_q2']


weights = weights.join(mean_weights, how = 'outer', lsuffix='_sw', rsuffix='_mw')


# fill dataframe NA with 0
weights = weights.fillna(0)


# concatenate columns from weights to df_all
df_all = pd.concat([df_all, weights], axis=1)


# once done with tfidf data, remove data frames to release memory
del sum_weights_1, sum_weights_2, mean_weights_1, mean_weights_2, 
del weights_dataframe_1, weights_dataframe_2, weights, mean_weights

# garbage collection
gc.collect()





# create word2vec model and set it as a global variable - the img_feat_gen module
# will use this to determine the word2vec similarity scores
#gs.word2vecmodel = ifg.create_w2vmodel(df_all)
gs.word2vecmodel = ifg.create_w2vmodel(None)
print('Generate word2vec model')


# write out to temporary CSV file so that we can process chunk by chunk later
df_all.to_csv('df_all_temp_pres.csv', sep=',', encoding='utf-8', index=False)
print('Generate tf_all_temp_pres.csv for subsequent chunking')

# once done with main data frame, delete it to release memory
del df_all

# garbage collection
gc.collect()






# create a HDFStore file so as to persist the large feature matrix - subsequently
# we will query the training and test set from this store

tr_store = pd.HDFStore('df_all_train_pres.h5', mode='w', complevel=9, complib='blosc', chunksize=gs.chunksize)

# read in temp CSV file into chunks, process and save chunk separately - 100k rows
tfrdr = pd.read_csv('df_all_temp_pres.csv', iterator=True, chunksize=gs.chunksize)

i = 0
for chunk in tfrdr: #for each 100k rows
    
    df = chunk.drop(['qid1', 'qid2'], axis=1)
    # resequence the grpId within the chunk for chunk level parallel processing
    df['grpId'] = df.apply(lambda row: row.name % _number_of_groups, axis=1, raw=True).values


##############################################################################
# Using word2vec to generate image features
##############################################################################

    df = applyParallel(df.groupby(df.grpId), ifg.gen_img_feat, {"dest_col_ind": df.shape[1]-1,
                                                                "dest_col_name": "28_28_matrix",
                                                                "col1": "q1nopunct",
                                                                "col2": "q2nopunct",
                                                                "matrix_size": 28,
                                                                "order": 0,
                                                                "show": False,
                                                                "tofile": False}, _cpu)
    print("Finished gen_img_feat processing", str(i), "chunks")


    
##############################################################################
# Using Wordnet to determine similarity scores for feature creation
##############################################################################

#==============================================================================
#     # calculate semantic_similarity based on cosine similarity of word vectors using WordNet
#     start_time=time.time() #taking current time as starting time
#     df = applyParallel(df.groupby(df.grpId), wnu.gen_semantic_similarity, {"dest_col_ind": df.shape[1]-1,
#                                                                            "dest_col_name": "sem_similarity",
#                                                                            "col1": "q1nopunct",
#                                                                            "col2": "q2nopunct",
#                                                                            "info_content_norm": False}, _cpu)
#     elapsed_time=time.time()-start_time #again taking current time - starting time 
#     print('Starting feature-gen: Finished gen_semantic_similarity processing {0}'.format(elapsed_time))
#     
#     
#     # calculate gen_word_order_similarity based on positional similarity of words between sentences
#     start_time=time.time() #taking current time as starting time
#     df = applyParallel(df.groupby(df.grpId), wnu.gen_word_order_similarity, {"dest_col_ind": df.shape[1]-1,
#                                                                              "dest_col_name": "word_order_similarity",
#                                                                              "col1": "q1nopunct",
#                                                                              "col2": "q2nopunct"}, _cpu)
#     elapsed_time=time.time()-start_time #again taking current time - starting time 
#     print('Starting feature-gen: Finished gen_word_order_similarity processing {0}'.format(elapsed_time))
# 
#
# 
# #wnu.semantic_similarity("why are so many quora users posting questions that are readily answered on google",
# #                        "why do people ask quora questions which can be answered easily by google", info_content_norm = False)
# #
# #wnu.word_order_similarity("what does it mean that every time i look at the clock the numbers are the same",
# #                          "how many times a day do a clock s hands overlap")
# 
# #wnu.word_order_similarity('digitally','scanned')
# #wnu.word_similarity('digitally','scanned')
# 
#==============================================================================




##############################################################################
# Persist the processed chunk into HDF5 file format
##############################################################################

    # further remove unecessary fields to reduce filesize
    df = df.drop([
                  'q1nopunct',
                  'q2nopunct',
                  'q1nopunct_stem',
                  'q2nopunct_stem'
                  ], axis=1)
    
    # re-type column labels to string so that performance will not be affected due to
    # mixed types. have to do it here because of the gen_img_feat function
    df.columns = df.columns.astype(str)
    
    # convert all float to float32 and int to int32 to save data size
    df = utils.to_32bit(df)

    train_df = df[df['type'].values == 'TR']

    # write out to HDFStore by appending the chunks and index the HDFStore with type
    tr_store.append('df', train_df, data_columns=['type'], chunksize=gs.chunksize, min_itemsize=1141)

    print("Finished processing", str(i), "chunks")

    i+=1
    
    # release memory
    del [chunk, df, train_df]


# must always close the HDFStore
tr_store.close()


# garbage collection
gc.collect()














##############################################################################
# Prepare Training Sets
##############################################################################

# Read the training and test set from HDFStore
trn = pd.DataFrame()

tr_store = pd.HDFStore('df_all_train.h5')
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
# which may negatively influence the XGBoost
trn.drop(['grpId',
          'id',
          'type',
          'test_id', #for trn data set, need to drop test_id as it is -9999 column
#          'tr_q1WrdCnt',
#          'tr_q2WrdCnt',
#          'tr_q1Len',
#          'tr_q2Len'
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

# set the column names of the data frame as str
trn.columns = trn.columns.astype(str)






# Normalise features
##############################################################################
from sklearn import preprocessing

# normalise the word count for specific columns
norm_col = ['tr_q1WrdCnt',
            'tr_q2WrdCnt',
            'tr_q1Len',
            'tr_q2Len',
            'weight_q1_sw',
            'weight_q2_sw']

scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
trn_scaled = pd.DataFrame(scaler.fit_transform(trn[norm_col]))
trn_scaled.columns = ['scaled_' + str(i) for i in trn_scaled.columns]

for i in trn_scaled.columns: trn[i] = trn_scaled[i].values

#trn.iloc[403711]
#trn[trn['id'] == 403712]

del trn_scaled








# Train and Build Model(s)
##############################################################################

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split

import xgboost as xgb




##############################################################################
# Negative cases oversampling to balance the data based on the testing set
##############################################################################

# Approx 36% are positive labels
pos_trn = trn[trn['is_duplicate'].values == 1]
neg_trn = trn[trn['is_duplicate'].values == 0]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_trn) / (len(pos_trn) + len(neg_trn))) / p) - 1

while scale > 1:
    neg_trn = pd.concat([neg_trn, neg_trn])
    scale -=1

neg_trn = pd.concat([neg_trn, neg_trn[:int(scale * len(neg_trn))]])
print(len(pos_trn) / (len(pos_trn) + len(neg_trn)))


# separate features and labels based on the new training dataset
x_train = pd.concat([pos_trn, neg_trn])
y_train = (np.zeros(len(pos_trn)) + 1).tolist() + np.zeros(len(neg_trn)).tolist()

# remove label column from x_train. Don't change y_train as it is already the label list
x_train.drop(['is_duplicate'], axis=1, inplace = True)

# training / test set split 80/20 based on random selection
# 4242, 12357
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = gs.seedvalue)

# free up 
del [pos_trn, neg_trn]
# garbage collection
gc.collect()




##############################################################################
# XBG Model (with stratified k-fold CV) (with neg oversampling)
##############################################################################


# training / test set split 80/20 based on random selection
# comment out if we are using neg over sampled set
#x_train, x_valid, y_train, y_valid = train_test_split(trn.ix[:, trn.columns != 'is_duplicate'],
#                                                      trn['is_duplicate'], test_size = 0.2, random_state = gs.seedvalue)


xgb_params = {'objective': 'binary:logistic',
              'learning_rate': 0.1, #same as eta in xgboost
              'gamma':1.076,
              'seed':gs.seedvalue,
              'n_estimators': 350,
              'scale_pos_weight': 1.237
              }

xgbmodel = xgb.XGBClassifier(**xgb_params)

cv_params = {'max_depth': [8],
             'min_child_weight': [5, 6],
             'subsample': [0.7, 0.8, 0.9],
             'colsample_bytree': [0.7, 0.8],
             'max_delta_step': [1]
             }


kfold = StratifiedKFold(n_splits = 5, random_state = 7)

# by default the n_jobs is use all CPU which is ok
optimized_GBM = GridSearchCV(xgbmodel,
                             cv_params,
                             scoring = 'log_loss',
                             cv = kfold)

optimized_GBM.fit(x_train, y_train)
optimized_GBM.grid_scores_

# make predictions for validation data
y_pred = optimized_GBM.predict(x_valid)


# round up predictions < 0.5 = 0, >= 0.5 = 1
predictions = [round(value) for value in y_pred]

# Accuracy score - but log loss more important
print("Accuracy: %.2f%%" % (accuracy_score(y_valid, predictions) * 100.0))

# Save XGB model to file using pickle
pickle.dump(optimized_GBM, open("xgbst.stratkfcv.pickle.dat", "wb"))

del [y_pred, predictions]
gc.collect()




##############################################################################
# XBG Model (with neg oversampling)
##############################################################################

# Set our parameters for xgboost
params = {}
params["objective"] = "binary:logistic"
params['eval_metric'] = 'logloss'
params["eta"] = 0.01
params["subsample"] = 0.8
params["min_child_weight"] = 5
params["colsample_bytree"] = 0.7
params["max_depth"] = 9
#params["silent"] = 1
params["gamma"] = 1.036
params["seed"] = gs.seedvalue
params["scale_pos_weight"] = 1.237
params["sketch_eps"] = 0.01
params["tree_method"] = 'auto'
params["max_delta_step"] = 1
params["nthread"] = _cpu



d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds = 100, verbose_eval = 10)


# make predictions for validation data
y_pred = bst.predict(d_valid)

# round up predictions < 0.5 = 0, >= 0.5 = 1
predictions = [round(value) for value in y_pred]

# Accuracy score - but log loss more important
print("Accuracy: %.2f%%" % (accuracy_score(y_valid, predictions) * 100.0))

# Save XGB model to file using pickle
pickle.dump(bst, open("xgbst.negovrsample.pickle.dat", "wb"))


del [d_train, d_valid]
del [y_pred, predictions]
gc.collect()






##############################################################################
# SVM Model
##############################################################################


# training / test set split 80/20 based on random selection
# trn.ix[:, trn.columns != 'is_duplicate'] = features
# trn['is_duplicate'] = label column


# remove large range of columns based on index and specific columnss
excludeCols = []
excludeCols = list(trn.ix[:,trn.columns.get_loc('0'):(trn.columns.get_loc('783')+1)].columns)
excludeCols.append('is_duplicate')


x_train, x_valid, y_train, y_valid = train_test_split(trn.ix[:, list(set(trn.columns) - set(excludeCols))],
                                                      trn['is_duplicate'],
                                                      test_size = 0.2,
                                                      random_state = gs.seedvalue)


kfold = StratifiedKFold(n_splits = 3, random_state = gs.seedvalue)

mlpc_tune_param = [
        {'solver': ['sgd','lbfgs','adam'],
         'activation':['relu','logistic','tanh'],
         'learning_rate':['invscaling','adaptive']}
]

clf = GridSearchCV(estimator = MLPClassifier(hidden_layer_sizes=(100,250,200,50,10),
                                             max_iter=20,
                                             alpha=1e-4,
                                             verbose=10,
                                             tol=1e-4,
                                             random_state=gs.seedvalue,
                                             learning_rate_init=0.0025
                                  ),
                   param_grid = mlpc_tune_param,
                   scoring = 'neg_log_loss',
                   n_jobs = -1,
                   iid = True,
                   refit = True,
                   cv = kfold,
                   verbose = 1)

# Train the classifier on x_train and y_train
clf.fit(x_train, y_train)


# validate using x_valid and y_valid
y_true, y_pred = y_valid, clf.predict(x_valid)
classification_report(y_true, y_pred)


# Save SVM model to file using pickle
pickle.dump(clf, open("svm.stratkfcv.pickle.dat", "wb"))


del [x_train, x_valid, y_train, y_valid]
gc.collect()






# Generate Test Results for scikit based learners (GridSearchCV for XGBoost / SVM)
##############################################################################
import xgboost as xgb

# Set reading chunk size
#chunksize = 10**5
chunksize = 200000

# create out data frame
out_df = pd.DataFrame()

# Read the training and test set from HDFStore
tst = pd.DataFrame()


# load all the saved models
#bst1 = pickle.load(open("xgbst.stratkfcvpickle.dat", "rb"))
bst = pickle.load(open("xgbst.stratkfcv.pickle.dat", "rb"))
svm = pickle.load(open("svm.stratkfcv.pickle.dat", "rb"))

# ......

ts_store = pd.HDFStore('df_all_test.h5')
nrows = ts_store.get_storer('df').nrows

#for chunk in pd.read_hdf('df_all_intermediate.h5','df', chunksize=chunksize, where='type = "TR"'):
for i in range(nrows//chunksize + 1):
    chunk = ts_store.select('df', start=i*chunksize, stop=(i+1)*chunksize)    
    tst = chunk.loc[chunk['type'] == 'TS']
    print("Finished reading", str(i), "chunks")

    # remove index columns and columns that have possibly low predictive power - 
    # which may negatively influence the XGBoost
    tst.drop(['grpId',
              'id',
              'type',
#              'tr_q1WrdCnt',
#              'tr_q2WrdCnt',
#              'tr_q1Len',
#              'tr_q2Len',
              'tr_q1WrdCnt_zscore',
              'tr_q2WrdCnt_zscore',
              'tr_q1Len_zscore',
              'tr_q2Len_zscore',
              'is_duplicate'], axis=1, inplace = True)

    # during the gen_semantic_similarity and gen_word_order_similarity there could
    # have been null / NA values - hence need to set the value to 0.0
    tst.loc[tst.word_order_similarity.isnull(), 'word_order_similarity'] = 0.0
    
    # during the generation of the tfidf weight matrix there could
    # have been null / NA values - hence need to set the value to 0.0
    tst.loc[tst.weight_q1_sw.isnull(), 'weight_q1_sw'] = 0.0
    tst.loc[tst.weight_q2_sw.isnull(), 'weight_q2_sw'] = 0.0
    tst.loc[tst.weight_q1_mw.isnull(), 'weight_q1_mw'] = 0.0
    tst.loc[tst.weight_q2_mw.isnull(), 'weight_q2_mw'] = 0.0
    # set the column names of the data frame as str
    tst.columns = tst.columns.astype(str)


    # Create PCA columns
    includeCols = []
    includeCols = list(trn.ix[:,tst.columns.get_loc('0'):(tst.columns.get_loc('783')+1)].columns)

    pca = PCA(n_components=10)
    trn = pca.pca(tst, includeCols, pca, is_scaling = True)


##############################################################################
# XGB Model
##############################################################################

    # make predictions for test data
    # if the model is based on GridSearchCV then use predict_proba
    y_test = bst.predict_proba(tst[list(tst.columns[1::])])

    # create result dataframe and append current chunk
    y_test = pd.DataFrame(y_test)
    y_test.columns = ["is_duplicate_0", "is_duplicate_1"]
    y_test["test_id"] = tst['test_id'].values
    out_df = out_df.append(y_test, ignore_index=True)


##############################################################################
# SVM Model
##############################################################################


    # remove large range of columns based on index and specific columnss
    excludeCols = []
    excludeCols = list(trn.ix[:,trn.columns.get_loc('0'):(trn.columns.get_loc('783')+1)].columns)
    excludeCols.append('is_duplicate')

    tst = tst.ix[:, list(set(tst.columns) - set(excludeCols))]

    # make predictions for test data
    # if the model is based on GridSearchCV then use predict_proba
    y_test = svm.predict_proba(tst[list(tst.columns[1::])])

    # create result dataframe and append current chunk
    y_test = pd.DataFrame(y_test)
    y_test.columns = ["is_duplicate_0", "is_duplicate_1"]
    y_test["test_id"] = tst['test_id'].values
    out_df = out_df.append(y_test, ignore_index=True)



    print("Processed chunk: {0}".format(len(out_df)))

    del [chunk, tst, y_test]
    gc.collect()

ts_store.close()

# garbage collection
gc.collect()


out_df.test_id = out_df.test_id.astype(np.int32)
out_df.is_duplicate_1 = out_df.is_duplicate_1.astype(np.float32)
# reorder and filter the columns
out_df = out_df[['test_id', 'is_duplicate_1']]

#rename columns
out_df.columns = ["test_id", "is_duplicate"]
# remove any duplicates
out_df.drop_duplicates(subset=['test_id'], keep=False)
# sort
out_df = out_df.sort_values(by=['test_id'], ascending=[True])
# save CSV
out_df.to_csv("submision.csv", index=False)
