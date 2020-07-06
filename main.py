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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

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
	return playerDataArray[:, 4:], mvpLabelsArray

def extractTestAndTrainData(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	return X_train, X_test, y_train, y_test	

def runDTree(X, y, depth=None):
	X_train, X_test, y_train, y_test = extractTestAndTrainData(X,y)
	clf = DecisionTreeClassifier(random_state=0, max_depth=depth, max_features=None, splitter="best")
	clf = clf.fit(X_train, y_train)
	# Get feature importance
	importances = clf.feature_importances_
	print(importances)
	y_pred = clf.predict(X_test)
	tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
	return tn, fp, fn, tp

def runSVM(X, y, kernel="rbf", C=1.0):
	X_train, X_test, y_train, y_test = extractTestAndTrainData(X,y)
	clf = svm.SVC(kernel=kernel, C=C)
	clf = clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
	return tn, fp, fn, tp

def runCV(clf, X, y, k, scoring):
	# Run total cv
	y_pred = cross_val_predict(clf, X, y, cv=k)
	unique, counts = np.unique(y_pred, return_counts=True)
	conf_mat = confusion_matrix(y, y_pred)
	print(dict(zip(unique, counts)))
	print(conf_mat)

	# Run individual cv's
	scores = cross_validate(clf, X, y, cv=k, scoring=scoring)
	print(scores)

def main(args):
	# Get the X and y matrices
	scoring = ['precision_macro', 'recall_macro', 'f1']
	playerDataDf = pd.read_csv(args.dataPath)
	X, y = getFeatureAndLabelMatrix(playerDataDf)
	# print(runDTree(X, y))
	# for i in range(-2,2):
	# 	C = 2.0**i
	# 	print("C parameter of: " + str(C))
	# 	print(runSVM(X,y,C=C,kernel="linear"))

	for i in range(-2,2):
		C = 2.0**i
		print("C parameter of: " + str(C) + " and linear kernel for cross validation")
		clf = svm.SVC(kernel="linear", C=C, random_state=0)
		runCV(clf,X,y,k=5,scoring=scoring)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(add_help=True)
	parser.add_argument('-d', '--dataPath', help='Path to cleaned player data', required=True)
	args = parser.parse_args()
	main(args)