import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import requests
import csv
from pathlib import Path


url = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2025"
response = requests.get(url)
data = response.json()

#retrieve stats from api
standings_data = []
for record in data['records']:
    for team_record in record['teamRecords']:
        try:
            team_name = team_record['team']['name']
            wins = team_record['wins']
            losses = team_record['losses']
            division = team_record['divisionRank']
            standings_data.append([team_name, wins, losses, division])
        except KeyError:
            continue

#create mlb_standings from api
with open("mlb_standings.csv", mode="w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Team", "Wins", "Losses", "Division Rank"])
    writer.writerows(standings_data)

#read csv
file_path = Path("mlb_standings.csv")
standings_df = pd.read_csv(file_path)
teams = standings_df['Team'].tolist()
team_to_idx = {team: idx for idx, team in enumerate(teams)}
idx_to_team = {idx: team for team, idx in team_to_idx.items()}
num_teams = len(teams)

#Initialize PyTorch Elo model
elo_ratings = torch.nn.Embedding(num_teams, 1)
torch.nn.init.constant_(elo_ratings.weight, 1500.0)
optimizer = optim.SGD(elo_ratings.parameters(), lr=0.01)


def elo_probability(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))


def generate_synthetic_matches(matches_per_pair=10):
    matches = []
    for team1 in range(num_teams):
        for team2 in range(num_teams):
            if team1 == team2:
                continue
            for _ in range(matches_per_pair):
                rating1 = elo_ratings(torch.tensor(team1)).item()
                rating2 = elo_ratings(torch.tensor(team2)).item()
                prob1 = elo_probability(rating1, rating2)
                winner = team1 if random.random() < prob1 else team2
                matches.append((team1, team2, winner))
    return matches


fixed_matches = generate_synthetic_matches(matches_per_pair=10)

def train_on_fixed_matches(matches, epochs=1000):
    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(matches)
        optimizer.zero_grad()
        for team1_idx, team2_idx, winner_idx in matches:
            rating1 = elo_ratings(torch.tensor(team1_idx))
            rating2 = elo_ratings(torch.tensor(team2_idx))
            pred = elo_probability(rating1, rating2)

            target = torch.tensor([1.0]) if winner_idx == team1_idx else torch.tensor([0.0])

            # Add Elo gap weighting
            elo_gap = abs(rating1.item() - rating2.item())
            weight = 1.0 + (elo_gap / 400)  

            loss = F.binary_cross_entropy(pred, target, weight=torch.tensor([weight]))
            loss.backward()
            total_loss += loss.item()
          
        optimizer.step()
        optimizer.zero_grad()
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}")


train_on_fixed_matches(fixed_matches)


trained_ratings = {team: elo_ratings(torch.tensor(idx)).item() for team, idx in team_to_idx.items()}
final_df = pd.DataFrame.from_dict(trained_ratings, orient='index', columns=['Elo Rating']).sort_values(by='Elo Rating', ascending=False)

final_df.to_csv("final_elo_ratings.csv", index=True)