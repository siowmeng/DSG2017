# -*- coding: utf-8 -*-
"""
Created on Sat May  6 22:52:47 2017

@author: siowmeng
"""

import pandas as pd
import numpy as np

trainData = pd.read_csv('train.csv')

uniqueGenres = np.unique(trainData['genre_id'].as_matrix())
uniqueUsers = np.unique(trainData['user_id'].as_matrix())

genreMatrix = np.empty((len(uniqueUsers), len(uniqueGenres)))
genreMatrix[:] = np.NaN
genreDF = pd.DataFrame(genreMatrix, index = uniqueUsers, columns = uniqueGenres)

# Populate Genre-User Table with 0, 1 and NaN
trainDataMatrix = trainData.loc[:, ['genre_id', 'user_id', 'is_listened']].as_matrix()

for i in range(trainDataMatrix.shape[0]):
    print(i)
    genre = trainDataMatrix[i, 0]
    user = trainDataMatrix[i, 1]
    listen = trainDataMatrix[i, 2]
    if np.isnan(genreDF.loc[user, genre]):
        genreDF.loc[user, genre] = listen
    else:
        if (listen == 1) and (genreDF.loc[user, genre] == 0):
            genreDF.loc[user, genre] = listen


# Write Matrix to CSV for future reading
genreDF.to_csv('genreDF.csv')