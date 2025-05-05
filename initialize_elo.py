# Let's adjust the provided code based on the user's CSV file structure
import pandas as pd
import random

# Load the standings data from the provided CSV file
standings_df = pd.read_csv('mlb_standings.csv')

# Initialize Elo ratings using team names from the CSV
elo_ratings = {team: 1500 for team in standings_df['Team Name']}

# Function to update Elo ratings after each game
def update_elo(elo_ratings, team1, team2, winner, k=32):
    rating1 = elo_ratings[team1]
    rating2 = elo_ratings[team2]
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))

    if winner == team1:
        score1, score2 = 1, 0
    else:
        score1, score2 = 0, 1

    elo_ratings[team1] += k * (score1 - expected1)
    elo_ratings[team2] += k * (score2 - (1 - expected1))

# Simulate games based on wins and losses from standings
def simulate_games(standings_df, elo_ratings, k=32):
    teams = standings_df['Team Name'].tolist()
    for _, row in standings_df.iterrows():
        team = row['Team Name']
        wins = row['Wins']
        losses = row['Losses']

        # Simulate wins
        for _ in range(wins):
            opponent = random.choice([t for t in teams if t != team])
            update_elo(elo_ratings, team, opponent, winner=team, k=k)

        # Simulate losses
        for _ in range(losses):
            opponent = random.choice([t for t in teams if t != team])
            update_elo(elo_ratings, team, opponent, winner=opponent, k=k)

# Predict matchup outcomes using the current Elo ratings
def predict_matchup(team1, team2, elo_ratings):
    rating1 = elo_ratings[team1]
    rating2 = elo_ratings[team2]
    expected1 = 1 / (1 + 10 ** ((rating2 - rating1) / 400))
    expected2 = 1 - expected1

    print(f"{team1} Elo: {round(rating1, 2)}")
    print(f"{team2} Elo: {round(rating2, 2)}")
    print(f"Probability {team1} wins: {round(expected1 * 100, 2)}%")
    print(f"Probability {team2} wins: {round(expected2 * 100, 2)}%")

# Run the simulation and prediction
simulate_games(standings_df, elo_ratings)
predict_matchup('Los Angeles Dodgers', 'New York Mets', elo_ratings)
