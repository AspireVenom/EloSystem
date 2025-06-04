import csv
import os
import random

import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torch.optim as optim

RETRAIN = True  # Set to True to force retraining

# --- Fetch API Data ---
GAME_OUTCOME_URL = (
    "https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2025&gameType=R"
)
STANDINGS_URL = "https://statsapi.mlb.com/api/v0/standings?leagueId=103,104&season=2025"

SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=2025-03-28&endDate=2025-09-28"

# -- HOME TEAM ADVANTAGE --
HOME_ADVANTAGE = 28.2

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
    "Athletics": "AL West",
}

outcome_data = requests.get(GAME_OUTCOME_URL, timeout=10).json()
standings_data_api = requests.get(STANDINGS_URL, timeout=10).json()


# --- Parse games from the schedule
def fetch_future_schedule():
    url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=2025-03-28&endDate=2025-09-28"
    response = requests.get(url, timeout=10).json()

    future_games = []
    for date_info in response.get("dates", []):
        for game in date_info.get("games", []):
            home = game.get("teams", {}).get("home", {}).get("team", {}).get("name")
            away = game.get("teams", {}).get("away", {}).get("team", {}).get("name")

            if home and away and home in team_to_idx and away in team_to_idx:
                future_games.append(
                    (team_to_idx[home], team_to_idx[away])
                )  # return indices

    return future_games


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
optimizer = optim.Adam(elo_ratings.parameters(), lr=0.05)

if os.path.exists("elo_model.pt") and not RETRAIN:
    print("Loading saved Elo model...")
    elo_ratings.load_state_dict(torch.load("elo_model.pt"))
else:
    print("Initializing Elo ratings to 1500...")
    torch.nn.init.constant_(elo_ratings.weight, 1500.0)


# --- Elo Probability ---
def elo_probability(r1, r2):
    return 1 / (1 + torch.exp((r2 - r1) * torch.log(torch.tensor(10.0)) / 50))


# --- Match Generators ---


def train_on_matches(matches, epochs, label, batch_size=64):
    match_tensor = torch.tensor(matches, dtype=torch.long)
    for epoch in range(epochs):
        total_loss = 0.0
        match_tensor = match_tensor[torch.randperm(len(match_tensor))]  # shuffle

        for batch_start in range(0, len(match_tensor), batch_size):
            batch = match_tensor[batch_start : batch_start + batch_size]

            team1_idx = batch[:, 0]
            team2_idx = batch[:, 1]
            winner_idx = batch[:, 2]
            is_home_team1 = batch[:, 3].bool()

            rating1 = elo_ratings(team1_idx)
            rating2 = elo_ratings(team2_idx)

            # Apply home advantage
            rating1 += HOME_ADVANTAGE * is_home_team1.unsqueeze(1)
            rating2 += HOME_ADVANTAGE * (~is_home_team1).unsqueeze(1)

            pred = elo_probability(rating1, rating2)

            # Create target: 1 if winner is team1 else 0
            target = (winner_idx == team1_idx).float().unsqueeze(1)

            elo_gap = torch.abs(rating1 - rating2)
            weight = (1.0 + (elo_gap / 50)).detach()

            loss = F.binary_cross_entropy(pred, target, weight=weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"[{label}] Epoch {epoch + 1}: Loss = {total_loss:.4f}")


def generate_synthetic_matches_from_schedule():
    scheduled_games = fetch_future_schedule()
    synthetic_matches = []

    for home_idx, away_idx in scheduled_games:
        # Get Elo ratings
        r_home = elo_ratings(torch.tensor([home_idx])).item()
        r_away = elo_ratings(torch.tensor([away_idx])).item()

        # Apply home field advantage
        r_home += HOME_ADVANTAGE

        # Calculate probability that home team wins
        prob_home_win = elo_probability(r_home, r_away).item()

        # Simulate a winner based on probability
        winner = home_idx if random.random() < prob_home_win else away_idx

        # Append match: (home_team_idx, away_team_idx, winner_idx, is_home_team1=True)
        synthetic_matches.append((home_idx, away_idx, winner, True))

    return synthetic_matches


# --- SAVE data from the synthetic matches predicted post elo
def save_synthetic_matches_to_csv(matches, filename="simulated_future_matches.csv"):
    rows = []
    for team1, team2, winner, is_home_team1 in matches:
        rows.append(
            {
                "Home Team": idx_to_team[team1],
                "Away Team": idx_to_team[team2],
                "Winner": idx_to_team[winner],
                "Home Win": idx_to_team[winner] == idx_to_team[team1],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(matches)} simulated matches to {filename}")


def generate_standings_from_simulated_matches(matches):
    team_records = {team: {"Wins": 0, "Losses": 0} for team in teams}

    for team1, team2, winner, _ in matches:
        team1_name = idx_to_team[team1]
        team2_name = idx_to_team[team2]
        winner_name = idx_to_team[winner]

        if winner_name == team1_name:
            team_records[team1_name]["Wins"] += 1
            team_records[team2_name]["Losses"] += 1
        else:
            team_records[team2_name]["Wins"] += 1
            team_records[team1_name]["Losses"] += 1

    # Add division info
    rows = []
    for team in teams:
        division = TEAM_TO_DIVISION.get(team, "Unknown")
        wins = team_records[team]["Wins"]
        losses = team_records[team]["Losses"]
        rows.append([team, wins, losses, division])

    standings_df = pd.DataFrame(
        rows, columns=["Team", "Wins", "Losses", "Division Name"]
    )

    # Assign division ranks
    standings_df["Division Rank"] = (
        standings_df.groupby("Division Name")["Wins"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    # Reorder columns
    standings_df = standings_df[
        ["Team", "Wins", "Losses", "Division Rank", "Division Name"]
    ]

    return standings_df


def generate_real_matches(game_data):
    matches = []
    for game in game_data:
        home = game["Home Team"]
        away = game["Away Team"]
        home_score = game["Home Score"]
        away_score = game["Away Score"]

        if home not in team_to_idx or away not in team_to_idx:
            continue

        team1 = team_to_idx[home]  # home
        team2 = team_to_idx[away]  # away
        winner = team1 if home_score > away_score else team2

        matches.append((team1, team2, winner, True))  # last value = is_home_team1
    return matches


# --- Run Training ---
if not os.path.exists("elo_model.pt") or RETRAIN:
    print("Training Elo model...")
    real_matches = generate_real_matches(games)
    train_on_matches(real_matches, epochs=10000, label="Real Matches")
    print("Saving Elo model...")
    torch.save(elo_ratings.state_dict(), "elo_model.pt")

    final_ratings = {
        team: elo_ratings(torch.tensor(idx)).item() for team, idx in team_to_idx.items()
    }
    final_df = pd.DataFrame(final_ratings.items(), columns=["Team", "Elo Rating"])
    final_df = final_df.sort_values(by="Elo Rating", ascending=False)
    final_df.to_csv("final_elo_ratings.csv", index=False)

#  Generate synthetic outcomes for future real schedule
print("Simulating real MLB 2025 scheduled games with Elo-based winners...")

standings_from_sim = generate_standings_from_simulated_matches(
    generate_synthetic_matches_from_schedule()
)
standings_from_sim.to_csv("simulated_standings.csv", index=False)
print("Saved simulated standings to simulated_standings.csv")
