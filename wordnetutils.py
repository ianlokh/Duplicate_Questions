#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 18:48:11 2017

@author: ianlo
"""
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.corpus import stopwords

import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

allow_numbers = pd.read_csv('./AllowedNumbers.csv')
allow_numbers = allow_numbers['Column1'].astype(str).tolist()

allow_stopwords = pd.read_csv('./AllowedStopwords.csv')
allow_stopwords = allow_stopwords['Column1'].astype(str).tolist()

stpwrd = list(set(stopwords.words('english')) - set(allow_stopwords))
stpwrd.sort()

#==============================================================================
# A. WordNet only contains "open-class words": nouns, verbs, adjectives, and adverbs.
# Thus, excluded words include determiners, prepositions, pronouns, conjunctions, and particles.
# 
#==============================================================================

def is_stopword(x):
    return x not in stopwords.words('english')


def is_stopword_numbers(x):
    if x.isnumeric():
        return x in allow_numbers
    else:
        return x not in stopwords.words('english')



# Parameters to the algorithm. Currently set to values that was reported
# in the paper to produce "best" results.
ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0

# dictionary for synset caching
synset_d = dict()

#synset_d = utils.ThreadSafeDict()


##############################################################################
# Semantic word similartiy
##############################################################################

def get_best_synset_pair(word_1, word_2):
    """ 
    Choose the pair with highest path similarity among all pairs. 
    Mimics pattern-seeking behavior of humans.
    """
    
    # try catch with dictionary caching is faster then wn.synsets method
    try:
        synsets_1 = synset_d[word_1]
    except KeyError:
        synsets_1 = wn.synsets(word_1)
        synset_d[word_1] = synsets_1

    try:
        synsets_2 = synset_d[word_2]
    except KeyError:
        synsets_2 = wn.synsets(word_2)
        synset_d[word_2] = synsets_2

#    synsets_1 = wn.synsets(word_1)
#    synsets_2 = wn.synsets(word_2)
    
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
#==============================================================================
#         max_sim = -1.0
#         best_pair = None, None
#         for synset_1 in synsets_1:
#             for synset_2 in synsets_2:
#                sim = wn.path_similarity(synset_1, synset_2)
#                if not sim: sim = -1.0 # check for NoneType so as not to cause exception
#                if (sim > max_sim):
#                    max_sim = sim
#                    best_pair = synset_1, synset_2
#         return best_pair
#==============================================================================


        #start_time=time.time()

        #res = [[wn.path_similarity(synset_1, synset_2), synset_1, synset_2] for synset_1 in synsets_1 for synset_2 in synsets_2]
        res = [[wn.wup_similarity(synset_1, synset_2), synset_1, synset_2] for synset_1 in synsets_1 for synset_2 in synsets_2]
        res.sort(key=lambda tup: float(0 if tup[0] is None else tup[0]), reverse=True)
 
        #print('path_similarity: {0} {1} {2}'.format(word_1, word_2, time.time()-start_time))

        return res[0][1], res[0][2]


def length_dist(synset_1, synset_2):
    """
    Return a measure of the length of the shortest path in the semantic 
    ontology (Wordnet in our case as well as the paper's) between two 
    synsets.
    """
    l_dist = sys.maxsize
    if synset_1 is None or synset_2 is None:
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        # calculate the shortest path between the words within each synset
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])        
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    
    #print('length_dist: {0} {1} {2}'.format(synset_1, synset_2, math.exp(-ALPHA * l_dist)))

    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)


def hierarchy_dist(synset_1, synset_2):
    """
    Return a measure of depth in the ontology to model the fact that 
    nodes closer to the root are broader and have less semantic similarity
    than nodes further away from the root.
    """
    h_dist = sys.maxsize
    if synset_1 is None or synset_2 is None: 
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                #if hypernyms_1.has_key(lcs_candidate):
                if lcs_candidate in hypernyms_1:
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                #if hypernyms_2.has_key(lcs_candidate):
                if lcs_candidate in hypernyms_2:
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0

    #print('hierarchy_dist: {0} {1} {2}'.format(synset_1, synset_2, ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / 
    #    (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))))

    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / 
        (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))



###############################################################################
# calculate word_similarity for nouns, verbs, adjectives, and adverbs
###############################################################################
def word_similarity(word_1, word_2):
#    if (word_1 == word_2):
#        print('similar word encountered: {0} {1}'.format(word_1, word_2))
#        return 1.0
#    else:
#        synset_pair = get_best_synset_pair(word_1, word_2)
#        return (length_dist(synset_pair[0], synset_pair[1]) * 
#            hierarchy_dist(synset_pair[0], synset_pair[1]))
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) * 
            hierarchy_dist(synset_pair[0], synset_pair[1]))



##############################################################################
# Semantic sentence similartiy
##############################################################################


def most_similar_word(word, word_set):
    """
    Find the word in the joint word set that is most similar to the word
    passed in. We use the algorithm above to compute word similarity between
    the word and each word in the joint word set, and return the most similar
    word and the actual similarity value.
    """
#==============================================================================
#     max_sim = -1.0
#     sim_word = ""
#     
#     #taking current time as starting time
#     start_time=time.time()
# 
#     for ref_word in word_set:                
#         sim = word_similarity(word, ref_word)                
#         if sim > max_sim:
#             max_sim = sim
#             sim_word = ref_word
# 
#     #again taking current time - starting time 
#     print('for loop {0} {1} {2}'.format(time.time()-start_time, sim_word, max_sim))
# 
#     return sim_word, max_sim
#==============================================================================

    #taking current time as starting time
    #start_time=time.time()

    sim = [[ref_word, word_similarity(word, ref_word)] for ref_word in word_set]
    sim.sort(key=lambda tup: float(0 if tup[1] is None else tup[1]), reverse=True)
    #sim_word, max_sim = sim[0][0], sim[0][1]

    #again taking current time - starting time 
    #print('list comprehensions {0} {1} {2}'.format(time.time()-start_time, sim[0][0], sim[0][1]))

    return sim[0][0], sim[0][1]


def info_content(lookup_word):
    """
    Uses the Brown corpus available in NLTK to calculate a Laplace
    smoothed frequency distribution of words, then uses this information
    to compute the information content of the lookup_word.
    """
    global N
    if N == 0:
        # poor man's lazy evaluation
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                #if not brown_freqs.has_key(word):
                if not(word in brown_freqs):
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
    #n = 0 if not brown_freqs.has_key(lookup_word) else brown_freqs[lookup_word]
    n = 0 if not lookup_word in brown_freqs else brown_freqs[lookup_word]
    return 1.0 - (math.log(n + 1) / math.log(N + 1))
    

def semantic_vector(words, joint_words, info_content_norm):
    """
    Computes the semantic vector of a sentence. The sentence is passed in as
    a collection of words. The size of the semantic vector is the same as the
    size of the joint word set. The elements are 1 if a word in the sentence
    already exists in the joint word set, or the similarity of the word to the
    most similar word in the joint word set if it doesn't. Both values are 
    further normalized by the word's (and similar word's) information content
    if info_content_norm is True.
    """
    sent_set = set(words)
    semvec = np.zeros(len(joint_words))
    i = 0
    
    for joint_word in joint_words:
        
        #taking current time as starting time
        #start_time=time.time()

        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            if info_content_norm:
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = PHI if max_sim > PHI else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1

        #again taking current time - starting time 
        #print('semantic_vector time {0}'.format(time.time()-start_time))
        
    return semvec                
            

###############################################################################
# calculate semantic_similarity for the entire sentence
###############################################################################

def semantic_similarity(sentence_1, sentence_2, info_content_norm):
    """
    Computes the semantic similarity between two sentences as the cosine
    similarity between the semantic vectors computed for each sentence.
    """

    #if (utils.is_nan(sentence_1) or utils.is_nan(sentence_2)):
    #    return 0

    # use try except block instead of if to catch exception cases - more efficient
    try:
        #words_1 = nltk.word_tokenize(sentence_1)
        #words_2 = nltk.word_tokenize(sentence_2)
        
        # remove all stopwords so that we won't have to evaluate them for semantic similarity
        # hopefully reduce computing time
        words_1 = list(filter(is_stopword_numbers,  nltk.word_tokenize(sentence_1)))
        words_2 = list(filter(is_stopword_numbers,  nltk.word_tokenize(sentence_2)))
        
        if (len(words_1) == 0 or len(words_2) == 0):
            return 0
        else:
            joint_words = set(words_1).union(set(words_2))
            
            #taking current time as starting time
            #start_time=time.time()
            
            vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
            vec_2 = semantic_vector(words_2, joint_words, info_content_norm)
            
            #again taking current time - starting time 
            #print('semantic_similarity time {0}'.format(time.time()-start_time))
            
            # cosine similarity of the semantic vectors - this includes all the words
            # not just nouns, verbs, adjectives, and adverbs
            return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    except:
        return 0



######################### word order similarity ##########################

def word_order_vector(words, joint_words, windex):
    """
    Computes the word order vector for a sentence. The sentence is passed
    in as a collection of words. The size of the word order vector is the
    same as the size of the joint word set. The elements of the word order
    vector are the position mapping (from the windex dictionary) of the 
    word in the joint set if the word exists in the sentence. If the word
    does not exist in the sentence, then the value of the element is the 
    position of the most similar word in the sentence as long as the similarity
    is above the threshold ETA.
    """
    
    #print('word_order_vector: words: {0} joint_words: {1}'.format(words, joint_words))

    wovec = np.zeros(len(joint_words))
    i = 0
    wordset = set(words)
    for joint_word in joint_words:

        #start_time=time.time()

#        if joint_word in wordset:
#            # word in joint_words found in sentence, just populate the index
#            wovec[i] = windex[joint_word]
#        else:
#            # word not in joint_words, find most similar word and populate
#            # word_vector with the thresholded similarity
#            
#            sim_word, max_sim = most_similar_word(joint_word, wordset)
#            if max_sim > ETA:
#                wovec[i] = windex[sim_word]
#            else:
#                wovec[i] = 0

        if joint_word in wordset:
            # word in joint_words found in sentence, just populate the index
            wovec[i] = windex[joint_word]
            #print('joint_word in wordset: {0}'.format(joint_word))
        else:
            if (joint_word not in stpwrd):
                # word not in joint_words, find most similar word and populate
                # word_vector with the thresholded similarity

                #print('joint_word for similarity: {0}'.format(joint_word))

                sim_word, max_sim = most_similar_word(joint_word, wordset)
                if max_sim > ETA:
                    wovec[i] = windex[sim_word]
                else:
                    wovec[i] = 0
            else:
                wovec[i] = 0
                #print('joint_word is stopword: {0}'.format(joint_word))

        i = i + 1
    return wovec



def word_order_similarity(sentence_1, sentence_2):
    """
    Computes the word-order similarity between two sentences as the normalized
    difference of word order between the two sentences.
    """

    try:
        #taking current time as starting time
        #start_time=time.time()
        
        # since we are using try except - don't need to have this if statement for
        # checking - more efficient to use try except
        #if (utils.is_nan(sentence_1) or utils.is_nan(sentence_2)):
        #    return 0
        
        words_1 = nltk.word_tokenize(sentence_1)
        words_2 = nltk.word_tokenize(sentence_2)
        
        
        joint_words = list(set(words_1).union(set(words_2)))
        windex = {x[1]: x[0] for x in enumerate(joint_words)}
        
        #start_time=time.time()
        r1 = word_order_vector(words_1, joint_words, windex)
        r2 = word_order_vector(words_2, joint_words, windex)
        #print('word_order_vector words_1: {0} words_2: {1} ts:{2}'.format(len(words_1), len(words_2), time.time()-start_time))

        return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))
    except:
        return 0


######################### overall similarity ##########################

def similarity(sentence_1, sentence_2, info_content_norm):
    """
    Calculate the semantic similarity between two sentences. The last 
    parameter is True or False depending on whether information content
    normalization is desired or not.
    """
    return DELTA * semantic_similarity(sentence_1, sentence_2, info_content_norm) + \
        (1.0 - DELTA) * word_order_similarity(sentence_1, sentence_2)


# wrapper function for parallel apply
def gen_semantic_similarity(df, dest_col_ind, dest_col_name, col1, col2, info_content_norm):
    df.insert(dest_col_ind, dest_col_name, df.apply(lambda x: semantic_similarity(x[col1], x[col2], info_content_norm), axis=1, raw=True))
    return df


# wrapper function for parallel apply
def gen_word_order_similarity(df, dest_col_ind, dest_col_name, col1, col2):
    df.insert(dest_col_ind, dest_col_name, df.apply(lambda x: word_order_similarity(x[col1], x[col2]), axis=1, raw=True))
    return df
