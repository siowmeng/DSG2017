# -*- coding: utf-8 -*-
"""
Created on Sun May 28 00:35:17 2017

@author: siowmeng

"""

import pandas as pd
import numpy as np
from datetime import datetime
import random

genreDF = pd.read_csv('genreDF.csv', index_col = 0)
genreDF.columns = pd.to_numeric(genreDF.columns)
genreList = list(genreDF.columns.values)

elementIdx = np.where(~np.isnan(genreDF.as_matrix()))
numElements = len(elementIdx[0])
validationSize = round(0.1 * numElements)
trainingSize = numElements - validationSize

valListIdx = random.sample(range(numElements), validationSize)
valIdx = (elementIdx[0][valListIdx], elementIdx[1][valListIdx])

trainingMatrix = genreDF.as_matrix().copy()
trainingMatrix[valIdx] = np.nan
validationMatrix = genreDF.as_matrix().copy()
validationMatrix[:, :] = np.nan
validationMatrix[valIdx] = genreDF.as_matrix()[valIdx]

mu = np.nanmean(trainingMatrix)
centRatings1 = trainingMatrix - mu
centRatingsVal1 = validationMatrix - mu
bu = np.nanmean(centRatings1, axis = 1)
bu[np.isnan(bu)] = 0
centRatings2 = centRatings1 - bu[:, np.newaxis]
centRatingsVal2 = centRatingsVal1 - bu[:, np.newaxis]
bi = np.nanmean(centRatings2, axis = 0)
bi[np.isnan(bi)] = 0
centRatings3 = centRatings2 - bi[np.newaxis, :]
centRatingsVal3 = centRatingsVal2 - bi[np.newaxis, :]

def get_error(R, PQ):
    return np.nansum((R - PQ)**2)

R = centRatings3.copy()
RVal = centRatingsVal3
m, n = trainingMatrix.shape
n_iterations = 100
n_factors = 150
prediction = []
starttime = datetime.now()
i = 0
for lambda_ in [15.5]:
    np.random.seed(1)
    P = np.random.rand(m, n_factors) / n_factors
    Q = np.random.rand(n_factors, n) / n_factors
    # Use Alternating Least Squares for Latent Factor estimation
    errors = []
    bestValError = np.inf
    bestPQ = np.dot(P, Q)
    for ii in range(n_iterations):
        print("Iteration", ii)
        P = np.linalg.solve(np.dot(Q, Q.T) + lambda_ * np.eye(n_factors), 
                            np.dot(Q, np.nan_to_num(R.T))).T
        Q = np.linalg.solve(np.dot(P.T, P) + lambda_ * np.eye(n_factors), 
                            np.dot(P.T, np.nan_to_num(R)))
        PQ = np.dot(P, Q)
        inSampleError = get_error(R, PQ)
        valError = get_error(RVal, PQ)
        if valError < bestValError:
            bestValError = valError
            bestPQ = PQ
        errors.append(valError)
        print("In-Sample SSE:", inSampleError)
        print("Out-of-Sample SSE:", valError)

endtime = datetime.now()

print("Start Time: ", starttime)
print("End Time: ", endtime)

testData = pd.read_csv('test.csv', index_col = 'sample_id')

listPrediction = []
for (genre, user) in testData[['genre_id', 'user_id']].values:
    
    predictGenre = genreList.index(genre)
    predictUser = user
    
    prediction = bestPQ[predictUser, predictGenre] + bi[predictGenre] + bu[predictUser] + mu
    if prediction > 1:
        prediction = 1.0
    elif prediction < 0:
        prediction = 0
    listPrediction.append(prediction)

# Write to submission CSV file
listSampleID = [x for x in range(testData.shape[0])]
resultsArray = np.array([listSampleID, listPrediction]).transpose()
resultDF = pd.DataFrame(resultsArray, columns = ['sample_id', 'is_listened'])
resultDF['sample_id'] = resultDF['sample_id'].astype('int')
resultDF.to_csv('submissionALS.csv', index = False)
