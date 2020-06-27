import pandas as pd

mvp_data = pd.read_csv(r'data/MVP.csv')
player_data = pd.read_csv(r'data/Seasons_Stats.csv')

print(mvp_data)
print(player_data)

def get_player_data(player, year):
    return player_data.loc[(player_data["Player"] == player) & (player_data["Year"] == year)]

def get_mvps():
    return mvp_data.loc[mvp_data["MVP"] == 1]

print(get_player_data("James Harden", 2010))
print(get_mvps())