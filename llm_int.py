import csv
import random
from pathlib import Path

import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torch.optim as optim

GAME_OUTCOME_URL = (
    "https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2024&gameType=R"
)
outcomeResponse = requests.get(GAME_OUTCOME_URL)
outcomeData = outcomeResponse.json()
url = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2024"
response = requests.get(url)
data = response.json()

# retrieve stats from api [STANDINGS]
standings_data = []
for record in data["records"]:
    for team_record in record["teamRecords"]:
        try:
            team_name = team_record["team"]["name"]
            wins = team_record["wins"]
            losses = team_record["losses"]
            division = team_record["divisionRank"]
            standings_data.append([team_name, wins, losses, division])
        except KeyError:
            continue
# create mlb_standings from api
with open("mlb_standings.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Team", "Wins", "Losses", "Division Rank"])
    writer.writerows(standings_data)

# retrieve stats from the outcome of the games from the api (GAME_OUTCOME_URL)
games = []
for date_info in outcomeData.get("dates", []):
    for game in date_info.get("games", []):
        game_date = game.get("gameDate")
        home_team = game.get("teams", {}).get("home", {}).get("team", {}).get("name")
        away_team = game.get("teams", {}).get("away", {}).get("team", {}).get("name")
        home_score = game.get("teams", {}).get("home", {}).get("score")
        away_score = game.get("teams", {}).get("away", {}).get("score")
        status = game.get("status", {}).get("detailedState")

        # only include final games (incase using api from present as opposed to old data)
        if status == "Final":
            games.append(
                {
                    "Date": game_date,
                    "Home Team": home_team,
                    "Away Team": away_team,
                    "Home Score": home_score,
                    "Away Score": away_score,
                }
            )

# write real game outcomes to CSV
with open("mlb_game_results.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file, fieldnames=["Date", "Home Team", "Away Team", "Home Score", "Away Score"]
    )
    writer.writeheader()
    writer.writerows(games)


# read csv
file_path = Path("mlb_standings.csv")
standings_df = pd.read_csv(file_path)
teams = standings_df["Team"].tolist()
team_to_idx = {team: idx for idx, team in enumerate(teams)}
idx_to_team = {idx: team for team, idx in team_to_idx.items()}
num_teams = len(teams)

# Initialize PyTorch Elo model
elo_ratings = torch.nn.Embedding(num_teams, 1)
torch.nn.init.constant_(elo_ratings.weight, 1500.0)
optimizer = optim.SGD(elo_ratings.parameters(), lr=0.01)


def elo_probability(r1, r2):
    """elo formula"""
    return 1 / (1 + 10 ** ((r2 - r1) / 400))


def generate_synthetic_matches(matches_per_pair=10):
    """'Pair each team against each team (atleast 10 times through the loop) to help calculate a more accurate elo based
    on initial elo rating"""
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


def generate_real_matches(game_data):
    matches = []
    for game in game_data:
        home = game["Home Team"]
        away = game["Away Team"]
        home_score = game["Home Score"]
        away_score = game["Away Score"]

        if home not in team_to_idx or away not in team_to_idx:
            continue  # Skip teams not in standings

        team1 = team_to_idx[home]
        team2 = team_to_idx[away]

        winner = team1 if home_score > away_score else team2

        matches.append((team1, team2, winner))
    return matches


real_matches = generate_real_matches(games)


fixed_matches = generate_synthetic_matches(matches_per_pair=10)


def train_on_synthetic_matches(matches, epochs=1000):
    """Train on the matches that we created in the fn generate_synthetic_matches"""
    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(matches)
        optimizer.zero_grad()
        for team1_idx, team2_idx, winner_idx in matches:
            rating1 = elo_ratings(torch.tensor(team1_idx))
            rating2 = elo_ratings(torch.tensor(team2_idx))
            pred = elo_probability(rating1, rating2)

            target = (
                torch.tensor([1.0]) if winner_idx == team1_idx else torch.tensor([0.0])
            )

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


def train_on_real_matches(matches, epochs=1000):
    """Train Elo model using real match outcomes (historical games)"""
    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(matches)
        optimizer.zero_grad()

        for team1_idx, team2_idx, winner_idx in matches:
            rating1 = elo_ratings(torch.tensor(team1_idx))
            rating2 = elo_ratings(torch.tensor(team2_idx))

            pred = elo_probability(rating1, rating2)
            target = (
                torch.tensor([1.0]) if winner_idx == team1_idx else torch.tensor([0.0])
            )

            # Optional: use lower weight than synthetic since real data is trusted
            elo_gap = abs(rating1.item() - rating2.item())
            weight = 1.0 + (elo_gap / 400)

            loss = F.binary_cross_entropy(pred, target, weight=torch.tensor([weight]))
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"[Real Matches] Epoch {epoch + 1}: Loss = {total_loss:.4f}")


def generate_realistic_synthetic_matches(matchups, matches_per_pair=10):
    """
    Generate synthetic matches from prior matchups (e.g., frequency of real games),
    making Elo learning more aligned with observed scheduling.
    """
    matches = []
    for team1, team2, freq in matchups:
        for _ in range(freq if freq > 0 else matches_per_pair):
            if team1 == team2:
                continue
            rating1 = elo_ratings(torch.tensor(team1)).item()
            rating2 = elo_ratings(torch.tensor(team2)).item()
            prob1 = elo_probability(rating1, rating2)
            winner = team1 if random.random() < prob1 else team2
            matches.append((team1, team2, winner))
    return matches


# Phase 1: Real-world calibration
train_on_real_matches(real_matches, epochs=1000)

# Save Elo ratings based ONLY on real match training
real_only_ratings = {
    team: elo_ratings(torch.tensor(idx)).item() for team, idx in team_to_idx.items()
}
real_only_df = pd.DataFrame(real_only_ratings.items(), columns=["Team", "Elo Rating"])
real_only_df = real_only_df.sort_values(by="Elo Rating", ascending=False)
real_only_df.to_csv("elo_after_real_training.csv", index=False)

# Phase 2: Generate synthetic matches with realistic Elo base
synthetic_matches = generate_synthetic_matches(matches_per_pair=10)

# Phase 3: Further train using synthetic matches
train_on_synthetic_matches(synthetic_matches, epochs=500)

# Save Elo ratings after full training (real + synthetic)
final_ratings = {
    team: elo_ratings(torch.tensor(idx)).item() for team, idx in team_to_idx.items()
}
final_df = pd.DataFrame(final_ratings.items(), columns=["Team", "Elo Rating"])
final_df = final_df.sort_values(by="Elo Rating", ascending=False)
final_df.to_csv("final_elo_ratings.csv", index=False)
