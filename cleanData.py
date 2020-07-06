#!/usr/bin/env python
# cleanData.py

import pandas as pd
import glob
import argparse
import os
import csv

def removeRowsWithoutPlayers(df):
	# if mvp dataset
	if 'PLAYER' in df:
		dfName = 'MVP Data'
		origNumRows = df.shape[0]
		numRowsNoPlayers = sum(df.PLAYER.isna())
		df = df[df.PLAYER.notna()]
	else:
		dfName = 'Season Stats Data'
		origNumRows = df.shape[0]
		numRowsNoPlayers = sum(df.Player.isna())
		df = df[df.Player.notna()]
	print(dfName, ', original num rows:', origNumRows, ', rows without players:', numRowsNoPlayers, ', new num rows:', df.shape[0])
	return df

def removeRepeatedPlayersInYear(statsDf):
	print('Removing repeated players from season stats')
	origNumRows = statsDf.shape[0]
	repeatedPlayers = statsDf[statsDf.Tm == 'TOT']
	countRowsRemoved = 0
	for i, row in repeatedPlayers.iterrows():
		indexToDrop = statsDf[(statsDf.Player == row.Player) & (statsDf.Year == row.Year) & (statsDf.Tm != 'TOT')].index
		statsDf.drop(indexToDrop, inplace=True)
		countRowsRemoved += indexToDrop.size
	print('Count repeat players:', countRowsRemoved, ', count rows before:', origNumRows, ', count rows after:', statsDf.shape[0])
	return statsDf

def removeStarChar(statsDf):
	statsDf.Player = statsDf.Player.str.strip('*')
	return statsDf

def addLabelToStats(mvpDf, statsDf):
	statsDf['MVP'] = 0
	mvps = mvpDf[mvpDf.MVP == 1]
	for i, mvp in mvps.iterrows():
		index = statsDf[(statsDf.Year == mvp.YEAR+1) & (statsDf.Player == mvp.PLAYER)].index
		statsDf.loc[index, 'MVP'] = 1
		print('Found MVP, Player:', mvp.PLAYER, ', Year:', mvp.YEAR)
	return statsDf

def resolveNans(statsDf):
	# For now, remove Games Started, 3 point Ar, and 3 point % features
	statsDf = statsDf.drop(['GS', '3PAr', '3P%'], axis=1)
	# Remove any features that are redundant info (like percentages)
	statsDf = statsDf.drop(['FT%', 'eFG%', '2P%', 'FG%'], axis=1)
	# Fill nans in 3P, 3PA, FTr, TS%, and TOV% as 0
	statsDf['3P'] = statsDf['3P'].fillna(0)
	statsDf['3PA'] = statsDf['3PA'].fillna(0)
	statsDf['FTr'] = statsDf['FTr'].fillna(0)
	statsDf['TS%'] = statsDf['TS%'].fillna(0)
	statsDf['TOV%'] = statsDf['TOV%'].fillna(0)
	# Remove Alex Scales, JamesOn Curry, and Damion James from data frame (PER = nan)
	statsDf = statsDf[statsDf.PER.notna()]
	return statsDf

def mapPositions(statsDf):
	statsDf.Pos = statsDf.Pos.str.split('-').str[0]
	posMapping = {'C': 0, 'PF': 1, 'PG': 2, 'SF': 3, 'SG': 4}
	statsDf = statsDf.replace({'Pos': posMapping})
	return statsDf

def main(args):
	outputCsv = os.path.join(args.outputPath, 'player_data_cleaned.csv')
	outputDf = os.path.join(args.outputPath, 'player_data_cleaned.pkl')
	mvpDf = removeRowsWithoutPlayers(pd.read_csv(args.mvpDataPath))
	mvpDf = mvpDf[mvpDf.YEAR != 2017]
	statsDf = removeRowsWithoutPlayers(pd.read_csv(args.seasonStatsPath))
	statsDf = removeRepeatedPlayersInYear(statsDf)
	statsDf = removeStarChar(statsDf)
	statsDf = addLabelToStats(mvpDf, statsDf)
	statsDf = resolveNans(statsDf)
	statsDf = mapPositions(statsDf)
	statsDf.ID = range(statsDf.shape[0])
	statsDf.to_pickle(outputDf)
	statsDf.to_csv(outputCsv, index=False)
	return mvpDf, statsDf

if __name__ == "__main__":
	parser = argparse.ArgumentParser(add_help=True)
	parser.add_argument('-m', '--mvpDataPath', help='Path to MVP data for players', required=True)
	parser.add_argument('-s', '--seasonStatsPath', help='Path to season stats for players', required=True)
	parser.add_argument('-o', '--outputPath', help='Output directory for cleaned data to go into',required=True)
	args = parser.parse_args()
	mvpDf, statsDf = main(args)
