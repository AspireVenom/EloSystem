import csv
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torch.optim as optim

RETRAIN = False  # Set to True to force retraining

# --- Fetch API Data ---
GAME_OUTCOME_URL = (
    "https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2025&gameType=R"
)
STANDINGS_URL = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2025"

# --- DEFAULT DIVISIONS FOR TEAMS
TEAM_TO_DIVISION = {
    "New York Yankees": "AL East",
    "Boston Red Sox": "AL East",
    "Toronto Blue Jays": "AL East",
    "Tampa Bay Rays": "AL East",
    "Baltimore Orioles": "AL East",
    "Chicago White Sox": "AL Central",
    "Cleveland Guardians": "AL Central",
    "Detroit Tigers": "AL Central",
    "Kansas City Royals": "AL Central",
    "Minnesota Twins": "AL Central",
    "Houston Astros": "AL West",
    "Los Angeles Angels": "AL West",
    "Oakland Athletics": "AL West",
    "Seattle Mariners": "AL West",
    "Texas Rangers": "AL West",
    "Atlanta Braves": "NL East",
    "Miami Marlins": "NL East",
    "New York Mets": "NL East",
    "Philadelphia Phillies": "NL East",
    "Washington Nationals": "NL East",
    "Chicago Cubs": "NL Central",
    "Cincinnati Reds": "NL Central",
    "Milwaukee Brewers": "NL Central",
    "Pittsburgh Pirates": "NL Central",
    "St. Louis Cardinals": "NL Central",
    "Arizona Diamondbacks": "NL West",
    "Colorado Rockies": "NL West",
    "Los Angeles Dodgers": "NL West",
    "San Diego Padres": "NL West",
    "San Francisco Giants": "NL West",
}


outcome_data = requests.get(GAME_OUTCOME_URL).json()
standings_data_api = requests.get(STANDINGS_URL).json()

# --- Parse Standings Data ---
standings_data = []
for record in standings_data_api["records"]:
    for team_record in record["teamRecords"]:
        try:
            team_name = team_record["team"]["name"]
            wins = team_record["wins"]
            losses = team_record["losses"]
            division_rank = team_record["divisionRank"]
            division_name = team_record.get("division", {}).get(
                "name", TEAM_TO_DIVISION.get(team_name, "Unknown Division")
            )
            standings_data.append(
                [team_name, wins, losses, division_rank, division_name]
            )
        except KeyError:
            continue
    # Always overwrite with fresh data to avoid mismatches
with open("mlb_standings.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Team", "Wins", "Losses", "Division Rank", "Division Name"])
    writer.writerows(standings_data)
# --- Parse Game Outcomes ---
games = []
for date_info in outcome_data.get("dates", []):
    for game in date_info.get("games", []):
        if game.get("status", {}).get("detailedState") != "Final":
            continue
        games.append(
            {
                "Date": game.get("gameDate"),
                "Home Team": game.get("teams", {})
                .get("home", {})
                .get("team", {})
                .get("name"),
                "Away Team": game.get("teams", {})
                .get("away", {})
                .get("team", {})
                .get("name"),
                "Home Score": game.get("teams", {}).get("home", {}).get("score"),
                "Away Score": game.get("teams", {}).get("away", {}).get("score"),
            }
        )

if not os.path.exists("mlb_game_results.csv"):
    with open("mlb_game_results.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Date", "Home Team", "Away Team", "Home Score", "Away Score"],
        )
        writer.writeheader()
        writer.writerows(games)

# --- Setup for Elo Model ---
standings_df = pd.read_csv("mlb_standings.csv")
teams = standings_df["Team"].tolist()
team_to_idx = {team: idx for idx, team in enumerate(teams)}
idx_to_team = {idx: team for team, idx in team_to_idx.items()}
num_teams = len(teams)

elo_ratings = torch.nn.Embedding(num_teams, 1)
optimizer = optim.Adam(elo_ratings.parameters(), lr=0.01)

if os.path.exists("elo_model.pt") and not RETRAIN:
    print("Loading saved Elo model...")
    elo_ratings.load_state_dict(torch.load("elo_model.pt"))
else:
    print("Initializing Elo ratings to 1500...")
    torch.nn.init.constant_(elo_ratings.weight, 1500.0)


# --- Elo Probability ---
def elo_probability(r1, r2):
    return 1 / (1 + torch.exp((r2 - r1) * torch.log(torch.tensor(10.0)) / 400))


# --- Match Generators ---
def generate_synthetic_matches(matches_per_pair=10):
    matches = []

    # Group teams by division name
    division_groups = standings_df.groupby("Division Name")["Team"].apply(list)

    for division_name, team_names in division_groups.items():
        team_indices = [team_to_idx[team] for team in team_names if team in team_to_idx]

        for i in range(len(team_indices)):
            for j in range(len(team_indices)):
                if i == j:
                    continue

                team1 = team_indices[i]
                team2 = team_indices[j]

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
            continue
        team1 = team_to_idx[home]
        team2 = team_to_idx[away]
        winner = team1 if home_score > away_score else team2
        matches.append((team1, team2, winner))
    return matches


# --- Trainers ---
def train_on_matches(matches, epochs, label):
    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(matches)
        optimizer.zero_grad()
        for team1_idx, team2_idx, winner_idx in matches:
            team1_tensor = torch.tensor([team1_idx], dtype=torch.long)
            team2_tensor = torch.tensor([team2_idx], dtype=torch.long)
            rating1 = elo_ratings(team1_tensor)
            rating2 = elo_ratings(team2_tensor)
            pred = elo_probability(rating1, rating2)
            target = (
                torch.tensor([[1.0]])
                if winner_idx == team1_idx
                else torch.tensor([[0.0]])
            )
            elo_gap = torch.abs(rating1 - rating2)
            weight = (1.0 + (elo_gap / 50)).detach()
            loss = F.binary_cross_entropy(pred, target, weight=weight)
            loss.backward()
            total_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"[{label}] Epoch {epoch + 1}: Loss = {total_loss:.4f}")


# --- Run Training ---
if not os.path.exists("elo_model.pt") or RETRAIN:
    print("Training Elo model...")
    real_matches = generate_real_matches(games)
    train_on_matches(real_matches, epochs=1000, label="Real Matches")
    synthetic_matches = generate_synthetic_matches(matches_per_pair=10)
    train_on_matches(synthetic_matches, epochs=500, label="Synthetic Matches")

    print("Saving Elo model...")
    torch.save(elo_ratings.state_dict(), "elo_model.pt")

    final_ratings = {
        team: elo_ratings(torch.tensor(idx)).item() for team, idx in team_to_idx.items()
    }
    final_df = pd.DataFrame(final_ratings.items(), columns=["Team", "Elo Rating"])
    final_df = final_df.sort_values(by="Elo Rating", ascending=False)
    final_df.to_csv("final_elo_ratings.csv", index=False)

else:
    print("Skipping training — Elo model already exists.")
