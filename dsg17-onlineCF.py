# -*- coding: utf-8 -*-
"""
Created on Sat May 27 17:49:35 2017

@author: siowmeng

"""

import pandas as pd
import numpy as np
from datetime import datetime

genreDF = pd.read_csv('genreDF.csv', index_col = 0)
genreDF.columns = pd.to_numeric(genreDF.columns)
genreList = list(genreDF.columns.values)

# Average rating of each genre
genreAvg = np.nanmean(genreDF.as_matrix(), axis = 0)
# Average rating given by each user
userAvg = np.nanmean(genreDF.as_matrix(), axis = 1)

# 0-1 matrix indicates the presence of user ratings
hasRatingMatrix = ~np.isnan(genreDF.as_matrix())

# User ratings, centered using user average
usrRatCent = genreDF.as_matrix().copy() - userAvg[:, np.newaxis]
usrRatCent = np.nan_to_num(usrRatCent)

numerator = np.dot(usrRatCent, usrRatCent.transpose())
denominator = np.sqrt(np.dot(usrRatCent**2, hasRatingMatrix.transpose())) * np.sqrt(np.dot(hasRatingMatrix, usrRatCent.transpose()**2))
denominator[np.where(denominator == 0)] = 1
distMatrix = numerator / denominator

# Pearson Correlation (average calculated using the whole user average)
#distMatrix = np.dot(usrRatCent, usrRatCent.transpose()) / (np.sqrt(np.dot(usrRatCent**2, hasRatingMatrix.transpose())) * np.sqrt(np.dot(hasRatingMatrix, usrRatCent.transpose()**2)))
np.fill_diagonal(distMatrix, 0) # Set diagonal to zero

distMatrix[abs(distMatrix) < 0.25] = 0
absWeightMatrix = np.dot(abs(distMatrix), hasRatingMatrix)

testData = pd.read_csv('test.csv', index_col = 'sample_id')

# Perform prediction
listPrediction = []
starttime = datetime.now()
for (genre, user) in testData[['genre_id', 'user_id']].values:
    
    predictGenre = genreList.index(genre)
    predictUser = user
    
    prediction = userAvg[predictUser]
    if absWeightMatrix[predictUser, predictGenre] != 0:
        prediction += np.dot(distMatrix[predictUser, :], usrRatCent[:, predictGenre]) / absWeightMatrix[predictUser, predictGenre]
        if prediction > 1:
            prediction = 1.0
        elif prediction < 0:
            prediction = 0
        
    listPrediction.append(prediction)

endtime = datetime.now()

print("Start Time: ", starttime)
print("End Time: ", endtime)

# Write to submission CSV file
listSampleID = [x for x in range(testData.shape[0])]
resultsArray = np.array([listSampleID, listPrediction]).transpose()
resultDF = pd.DataFrame(resultsArray, columns = ['sample_id', 'is_listened'])
resultDF['sample_id'] = resultDF['sample_id'].astype('int')
resultDF.to_csv('submissionCF.csv', index = False)

