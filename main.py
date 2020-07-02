#!/usr/bin/env python
# main.py

import pandas as pd
import glob
import argparse
import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import datasets
from sklearn import svm

# def get_player_data(player, year):
#     all_data = player_data.loc[(player_data["Player"] == player) & (player_data["Year"] == year)]
#     if (all_data.shape[0] != 1):
#         return all_data.loc[(all_data["Tm"] == "TOT")]
#     return all_data

# def get_mvps():
#     return player_data.loc[player_data["MVP"] == 1]

def getFeatureAndLabelMatrix(playerDataDf):
	mvpLabels = playerDataDf.MVP
	playerData = playerDataDf.loc[:, playerDataDf.columns != 'MVP']
	mvpLabelsArray = mvpLabels.to_numpy()
	playerDataArray = playerData.to_numpy()
	return playerDataArray[:, 7:], mvpLabelsArray

def extractTestAndTrainData(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
	return X_train, X_test, y_train, y_test	

def main(args):
	scoring = ['precision_macro', 'recall_macro']
	playerDataDf = pd.read_csv(args.dataPath)
	X, y = getFeatureAndLabelMatrix(playerDataDf)
	return X, y
	# clf = svm.SVC(kernel='linear', C=1, random_state=0)
	# scores = cross_validate(clf, X, y, scoring=scoring)
	# return scores

if __name__ == "__main__":
	parser = argparse.ArgumentParser(add_help=True)
	parser.add_argument('-d', '--dataPath', help='Path to cleaned player data', required=True)
	args = parser.parse_args()
	X, y = main(args)