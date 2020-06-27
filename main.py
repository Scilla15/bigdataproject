import pandas as pd

player_data = pd.read_csv(r'data/player_data.csv')

def get_player_data(player, year):
    all_data = player_data.loc[(player_data["Player"] == player) & (player_data["Year"] == year)]
    if (all_data.shape[0] != 1):
        return all_data.loc[(all_data["Tm"] == "TOT")]
    return all_data

def get_mvps():
    return player_data.loc[player_data["MVP"] == 1]

def main():
    x = player_data[player_data.Player.notna()]
    x.to_csv(r'data/new_data.csv')

if __name__ == "__main__":
    main()