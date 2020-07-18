#!/usr/bin/env python
# plotData.py

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

nbins = 50

allStats = pd.read_csv('./Data/cluster_data.csv')

# mvpStats = allStats[allStats.MVP == 1]
# regStats = allStats[allStats.MVP == 0]

cluster0 = allStats[allStats.Cluster == 0]
cluster12 = allStats[allStats.Cluster == 12]
clusterRest = allStats[(allStats.Cluster != 0) & (allStats.Cluster != 12)]

def plotHistogram(feature, title, nbins, logscale=True):
	c0 = cluster0[feature].to_numpy()
	c12 = cluster12[feature].to_numpy()
	rest = clusterRest[feature].to_numpy()
	plt.hist(rest, bins=nbins, log=logscale, label="rest")
	plt.hist(c12, bins=nbins, log=logscale, label='c12')
	plt.hist(c0, bins=nbins, log=logscale, label='c0')
	plt.legend()
	if logscale:
		title = title + ' (Log Scale)'
	plt.title(title)
	plt.show()

def plotScatter(feature1, feature2, title):
	regData1 = regStats[feature1].to_numpy()
	regData2 = regStats[feature2].to_numpy()
	mvpData1 = mvpStats[feature1].to_numpy()
	mvpData2 = mvpStats[feature2].to_numpy()
	plt.scatter(regData1, regData2)
	plt.scatter(mvpData1, mvpData2, label='MVP')
	plt.legend()
	plt.xlabel(feature1)
	plt.ylabel(feature2)
	plt.title(title)
	plt.show()

for (columnName, columnData) in allStats.iteritems():
	plotHistogram(columnName, columnName, nbins)