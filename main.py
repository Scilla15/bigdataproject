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
    print(get_player_data("Dennis Awtrey", 1979))

if __name__ == "__main__":
    main()