import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from pathlib import Path

# Load standings data
file_path = Path("mlb_standings.csv")
standings_df = pd.read_csv(file_path)

# Team index mapping
teams = standings_df['Team Name'].tolist()
team_to_idx = {team: idx for idx, team in enumerate(teams)}
num_teams = len(teams)

# Initialize trainable Elo ratings
elo_ratings = torch.nn.Embedding(num_teams, 1)
torch.nn.init.constant_(elo_ratings.weight, 1500.0)

# Optimizer
optimizer = optim.Adam(elo_ratings.parameters(), lr=0.1)

# Elo probability function
def elo_probability(rating1, rating2):
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

# Generate synthetic match data from standings
synthetic_matches = []
for _, row in standings_df.iterrows():
    team_idx = team_to_idx[row['Team Name']]
    wins = int(row['Wins'])
    losses = int(row['Losses'])
    opponents = [i for i in range(num_teams) if i != team_idx]

    for _ in range(wins):
        opponent = random.choice(opponents)
        synthetic_matches.append((team_idx, opponent, team_idx))  # team won

    for _ in range(losses):
        opponent = random.choice(opponents)
        synthetic_matches.append((team_idx, opponent, opponent))  # team lost

# Training function
def train_elo(matches, epochs=300):
    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(matches)
        optimizer.zero_grad()

        for team1_idx, team2_idx, winner_idx in matches:
            rating1 = elo_ratings(torch.tensor(team1_idx))
            rating2 = elo_ratings(torch.tensor(team2_idx))
            pred = elo_probability(rating1, rating2)

            target = torch.tensor([1.0]) if winner_idx == team1_idx else torch.tensor([0.0])
            loss = F.binary_cross_entropy(pred, target)

            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")

# Run training
train_elo(synthetic_matches)

# Get trained Elo ratings
trained_ratings = {team: elo_ratings(torch.tensor(idx)).item() for team, idx in team_to_idx.items()}

import ace_tools_open as tools; tools.display_dataframe_to_user(name="Trained Elo Ratings", dataframe=pd.DataFrame.from_dict(trained_ratings, orient='index', columns=['Elo Rating']).sort_values(by='Elo Rating', ascending=False))
