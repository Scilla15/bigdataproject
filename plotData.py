#!/usr/bin/env python
# plotData.py

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

nbins = 50

allStats = pd.read_pickle('./Data/player_data_cleaned.pkl')

mvpStats = allStats[allStats.MVP == 1]
regStats = allStats[allStats.MVP == 0]

def plotHistogram(feature, title, nbins, logscale=True):
	regData = regStats[feature].to_numpy()
	mvpData = mvpStats[feature].to_numpy()
	plt.hist(regData, bins=nbins, log=logscale)
	plt.hist(mvpData, bins=nbins, log=logscale, label='MVP')
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
