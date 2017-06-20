#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 22:38:30 2017

@author: ianlo
"""

import pandas as pd
from multiprocessing import Pool, cpu_count


class WithExtraArgs(object):
  def __init__(self, func, **args):
    self.func = func
    self.args = args
  def __call__(self, df):
    return self.func(df, **self.args)


def applyParallel(dfGrouped, func, kwargs, cpuCount=cpu_count()):
    with Pool(cpuCount) as p:
        ret_list = p.map(WithExtraArgs(func, **kwargs), [group for name, group in dfGrouped])
    return pd.concat(ret_list)


#if __name__ == '__main__':
#    df = pd.DataFrame({'a': [6, 2, 2], 'b': [4, 5, 6]},index= ['g1', 'g1', 'g2'])
#    print ('parallel version: ')
#    print (applyParallel(df.groupby(df.index), tmpFunc))
#
#    print ('regular version: ')
#    print (df.groupby(df.index).apply(tmpFunc))
#
#    print ('ideal version (does not work): ')
#    print (df.groupby(df.index).applyParallel(tmpFunc))
    

#df = pd.DataFrame({'a': [1, 1, 1, 2, 2, 2], 'b': [1, 2, 1, 1, 2, 1]})
#groups = df.groupby(['a', 'b'])
#groups.grouper.group_info[0]
#
#
#_number_of_groups = 4
#
#df_train = pd.read_csv('./train.csv')
#df_train.insert(0,'grpId',df_train.apply(lambda row: row.name % _number_of_groups, axis=1, raw=True))
#df_train.head()
#
#print (applyParallel(df_train.groupby(df_train.grpId), tmpFunc2))

