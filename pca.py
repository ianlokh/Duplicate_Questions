#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:04:32 2017

@author: ianlo
"""

# Apply PCA and t-sne to reduce the 784 features
##############################################################################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



def pca(df, cols, pca, X_scaler, is_scaling = True):

    #pcaDF = df.loc[:, cols].values.copy()
    pcaDF = df.loc[:, cols].copy()

    if (is_scaling):
        #X_scaler = StandardScaler()
        standardisedX = X_scaler.fit_transform(pcaDF)
        pcaDF = pd.DataFrame(standardisedX)
        
    pca_result = pd.DataFrame(pca.fit_transform(pcaDF))
    pca_result.columns = ['pca_' + str(i) for i in pca_result.columns]

    #df = pd.concat([df, pca_result], axis=1)    
    for i in pca_result.columns: df[i] = pca_result[i].values
    
    del pca_result, pcaDF

    return df



def screeplot(pca, standardised_values):
    y = np.std(pca.transform(standardised_values), axis=0)**2
    x = np.arange(len(y)) + 1
    plt.plot(x, y, "o-")
    plt.xticks(x, ["Comp."+str(i) for i in x], rotation=60)
    plt.ylabel("Variance")
    plt.show()



def pca_scatter(pca, standardised_values, classifs):
    foo = pca.transform(standardised_values)
    bar = pd.DataFrame(list(zip(foo[:, 0], foo[:, 1], classifs)), columns=["PC1", "PC2", "Class"])
    sns.lmplot("PC1", "PC2", bar, hue="Class", fit_reg=False)
    


#==============================================================================
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# 
# 
# #X_scaler = StandardScaler()
# #standardisedX = X_scaler.fit_transform(trn.loc[:,'0':'783'].values)
# #standardisedX = pd.DataFrame(standardisedX)
# 
# pca = PCA(n_components=10)
# pca_result = pca.fit_transform(trn.loc[:,'tr_q1WrdCnt':'783'].values)
# #pca_result = pca.fit_transform(standardisedX)
# 
# 
# df = pd.DataFrame();
# df['PC1'] = pca_result[:,0]
# df['PC2'] = pca_result[:,1]
# df['PC3'] = pca_result[:,2]
# df['Class'] = trn.is_duplicate.values
# 
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
# 
# sum(pca.explained_variance_ratio_)
# 
# #pca.screeplot(pca, trn.loc[:,'tr_q1WrdCnt':'783'])
# #pca.pca_scatter(pca, trn.loc[:,'0':'783'].values, trn.is_duplicate)
# 
# interactive(utils.plot_3d_scatter, A=fixed(df), elevation=20, azimuth=240)
# 
# del df, pca
#==============================================================================
