#!/usr/bin/env python
# pca.py

import pandas as pd
import glob
import argparse
import os
import csv
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

allStats = pd.read_pickle('~/MATH189/project/Data/player_data_cleaned.pkl')

def scaleData(playerDf):
	# Input: cleaned data pandas df
	# Output: scaled X data matrix with unecessary features removed, label vector, as numpy arrays
	X = playerDf.drop(['ID','Year','Player','Tm', 'MVP'], axis=1, errors='ignore')
	y = playerDf.loc[:, allStats.columns == 'MVP'].values
	X_scaled = StandardScaler().fit_transform(X)
	return X_scaled, y

def performPca(scaledData, n_components=.95):
	#Inputs: scaled data matrix, number of components for pca (starts off as just trying to get 95% of variance)
	#Outputs: sklearn pca class
	pca = PCA(n_components=n_components)
	principalComp = pca.fit_transform(X_scaled)
	principalDf = pd.DataFrame(data=principalComp)
	return pca, principalDf