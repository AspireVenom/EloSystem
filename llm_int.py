"""
llm_int.py

Simulates MLB game outcomes and standings using an Elo-based model. Fetches real and future schedules, trains an Elo model on real results, and simulates future games.
"""

# --- Imports ---
import csv
import os
import random
import json
from typing import List, Tuple, Dict, Any

import pandas as pd
import requests
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import optuna
import optuna.visualization as vis





# --- Configuration & Constants ---
RETRAIN = True  # Set to True to force retraining

GAME_OUTCOME_URL = (
    "https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2025&gameType=R"
)
STANDINGS_URL = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2025"
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=2025-03-28&endDate=2025-09-28"

# --- Optimized Hyperparameters (from Optuna Bayesian optimization) ---
HOME_ADVANTAGE = 10.24  # optimized
DIVISION_RATING_SCALE = 71.15  # optimized
LEARNING_RATE = 0.00125  # optimized
BATCH_SIZE = 32  # optimized

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

# --- Utility Functions ---
def fetch_json(url: str) -> dict:
    """Fetch JSON data from a URL with error handling."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return {}

def fetch_future_schedule(team_to_idx: Dict[str, int]) -> List[Tuple[int, int]]:
    """Fetches the future MLB schedule and returns a list of (home_idx, away_idx) tuples."""
    response = fetch_json(SCHEDULE_URL)
    future_games = []
    for date_info in response.get("dates", []):
        for game in date_info.get("games", []):
            home = game.get("teams", {}).get("home", {}).get("team", {}).get("name")
            away = game.get("teams", {}).get("away", {}).get("team", {}).get("name")
            if home and away and home in team_to_idx and away in team_to_idx:
                future_games.append((team_to_idx[home], team_to_idx[away]))
    return future_games

def parse_standings_data(api_data: dict) -> List[list]:
    """Parse standings data from API response."""
    standings_data = []
    if "records" not in api_data:
        print("API response missing 'records'. Here is the actual response:")
        print(json.dumps(api_data, indent=2))
        raise SystemExit("Aborting due to invalid standings API response.")
    for record in api_data["records"]:
        team_records = record.get("teamRecords")
        if not team_records:
            print("'teamRecords' missing or empty in a record. Skipping...")
            continue
        for team_record in team_records:
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
            except KeyError as e:
                print(f"Missing key in team_record: {e}. Skipping this record.")
                continue
    if not standings_data:
        raise ValueError("No valid standings data parsed. Check team names and API structure.")
    return standings_data

def save_csv(filename: str, header: List[str], rows: List[Any]) -> None:
    """Save rows to a CSV file with a header."""
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)

def parse_game_outcomes(api_data: dict) -> List[dict]:
    """Parse completed game outcomes from API response."""
    games = []
    for date_info in api_data.get("dates", []):
        for game in date_info.get("games", []):
            if game.get("status", {}).get("detailedState") != "Final":
                continue
            games.append(
                {
                    "Date": game.get("gameDate"),
                    "Home Team": game.get("teams", {}).get("home", {}).get("team", {}).get("name"),
                    "Away Team": game.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                    "Home Score": game.get("teams", {}).get("home", {}).get("score"),
                    "Away Score": game.get("teams", {}).get("away", {}).get("score"),
                }
            )
    return games

def elo_probability(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Calculate the probability that r1 beats r2 using the Elo formula."""
    return 1 / (1 + torch.exp((r2 - r1) * torch.log(torch.tensor(10.0)) / 50))

def train_on_matches(
    matches: List[Tuple[int, int, int, bool]],
    epochs: int,
    label: str,
    batch_size: int,
    elo_ratings: torch.nn.Embedding,
    division_bonus_tensor: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Train the Elo model on match data."""
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
            # Apply division rating
            rating1 += DIVISION_RATING_SCALE * division_bonus_tensor[team1_idx]
            rating2 += DIVISION_RATING_SCALE * division_bonus_tensor[team2_idx]
            pred = elo_probability(rating1, rating2)
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

def generate_synthetic_matches_from_schedule(
    scheduled_games: List[Tuple[int, int]],
    elo_ratings: torch.nn.Embedding,
    division_bonus_tensor: torch.Tensor,
) -> List[Tuple[int, int, int, bool]]:
    """Simulate future matches using the current Elo ratings."""
    synthetic_matches = []
    for home_idx, away_idx in scheduled_games:
        r_home = elo_ratings(torch.tensor([home_idx])).item()
        r_away = elo_ratings(torch.tensor([away_idx])).item()
        r_home += HOME_ADVANTAGE
        r_home += DIVISION_RATING_SCALE * division_bonus_tensor[home_idx].item()
        r_away += DIVISION_RATING_SCALE * division_bonus_tensor[away_idx].item()
        prob_home_win = elo_probability(torch.tensor(r_home), torch.tensor(r_away)).item()
        winner = home_idx if random.random() < prob_home_win else away_idx
        synthetic_matches.append((home_idx, away_idx, winner, True))
    return synthetic_matches

def save_synthetic_matches_to_csv(
    matches: List[Tuple[int, int, int, bool]],
    idx_to_team: Dict[int, str],
    filename: str = "simulated_future_matches.csv",
) -> None:
    """Save simulated match results to a CSV file."""
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
    print(f"Saved {len(matches)} simulated matches to {filename}")

def generate_standings_from_simulated_matches(
    matches: List[Tuple[int, int, int, bool]],
    teams: List[str],
    idx_to_team: Dict[int, str],
) -> pd.DataFrame:
    """Generate standings from simulated matches."""
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
    rows = []
    for team in teams:
        division = TEAM_TO_DIVISION.get(team, "Unknown")
        wins = team_records[team]["Wins"]
        losses = team_records[team]["Losses"]
        rows.append([team, wins, losses, division])
    standings_df = pd.DataFrame(
        rows, columns=["Team", "Wins", "Losses", "Division Name"]
    )
    standings_df["Division Rank"] = (
        standings_df.groupby("Division Name")["Wins"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    standings_df = standings_df[
        ["Team", "Wins", "Losses", "Division Rank", "Division Name"]
    ]
    return standings_df

def generate_real_matches(
    game_data: List[dict], team_to_idx: Dict[str, int]
) -> List[Tuple[int, int, int, bool]]:
    """Convert real game data to match tuples for training."""
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

def fetch_completed_games_for_season(season: int) -> list:
    """
    Fetch all completed games for a given MLB season using the MLB API.
    Returns a list of dicts with date, home_team, away_team, home_score, away_score.
    """
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}&gameType=R"
    data = fetch_json(url)
    games = []
    for date_info in data.get("dates", []):
        for game in date_info.get("games", []):
            if game.get("status", {}).get("detailedState") != "Final":
                continue  # Only completed games
            games.append({
                "date": game["gameDate"],
                "home_team": game["teams"]["home"]["team"]["name"],
                "away_team": game["teams"]["away"]["team"]["name"],
                "home_score": game["teams"]["home"].get("score", 0),
                "away_score": game["teams"]["away"].get("score", 0),
            })
    return games

def backtest_bayesian_elo_on_season(season: int, K=0.1, T=1.0, initial_elo: float = 1500.0, initial_sigma2: float = 2000.0**2):
    """
    Run Bayesian Elo backtesting on a given MLB season using real game results from the MLB API.
    Each team's rating is a normal distribution (mean, variance).
    Saves elo_history_bayes.csv (with mean and stddev) and pred_vs_actual_bayes.csv.
    """
    print(f"\nRunning Bayesian Elo backtest for season {season}...")
    games = fetch_completed_games_for_season(season)
    if not games:
        print("No completed games found for this season.")
        return
    teams = sorted(set(g["home_team"] for g in games).union(g["away_team"] for g in games))
    team_mu = {team: initial_elo for team in teams}
    team_sigma2 = {team: initial_sigma2 for team in teams}
    predictions = []
    actuals = []
    elo_history = []
    initial_sigma2 = 2000.0**2
    min_sigma2 = 1000.0

    for g in sorted(games, key=lambda x: x["date"]):
        home, away = g["home_team"], g["away_team"]
        mu_diff = team_mu[home] + HOME_ADVANTAGE - team_mu[away]
        sigma2_sum = team_sigma2[home] + team_sigma2[away]
        temp = 2.5
        prob = norm.cdf((mu_diff / np.sqrt(sigma2_sum)) / T)
        epsilon = 1e-6
        clamped_prob = min(max(prob, epsilon), 1 - epsilon)
        predictions.append(clamped_prob)
        actuals.append(1 if g["home_score"] > g["away_score"] else 0)
        result = 1 if g["home_score"] > g["away_score"] else 0
        expected = clamped_prob
        delta = K * (result - expected)
        # Update means
        team_mu[home] += delta * team_sigma2[home] / sigma2_sum
        team_mu[away] -= delta * team_sigma2[away] / sigma2_sum
        # Update variances (uncertainty decreases)
        team_sigma2[home] = max(1 / (1/team_sigma2[home] + 1/sigma2_sum), min_sigma2)
        team_sigma2[away] = max(1 / (1/team_sigma2[away] + 1/sigma2_sum), min_sigma2)
        # Save Elo means and stddevs for both teams after the game
        elo_history.append({"date": g["date"], "team": home, "elo_mu": team_mu[home], "elo_std": np.sqrt(team_sigma2[home])})
        elo_history.append({"date": g["date"], "team": away, "elo_mu": team_mu[away], "elo_std": np.sqrt(team_sigma2[away])})

    # Evaluate
    print(f"Bayesian Elo Backtest results for {season}:")
    print("Log loss:", log_loss(actuals, predictions))
    print("Accuracy:", accuracy_score(actuals, [p > 0.5 for p in predictions]))
    print("Brier score:", brier_score_loss(actuals, predictions))
    # Save for dashboard
    pd.DataFrame(elo_history).to_csv("elo_history_bayes.csv", index=False)
    pd.DataFrame({"prob": predictions, "actual": actuals}).to_csv("pred_vs_actual_bayes.csv", index=False)
    print("Saved elo_history_bayes.csv and pred_vs_actual_bayes.csv for dashboard visualization.")
    return log_loss(actuals, predictions)

def main():
    # --- Fetch and save standings data ---
    standings_data_api = fetch_json(STANDINGS_URL)
    standings_data = parse_standings_data(standings_data_api)
    save_csv(
        "mlb_standings.csv",
        ["Team", "Wins", "Losses", "Division Rank", "Division Name"],
        standings_data,
    )
    # --- Parse teams and indices ---
    standings_df = pd.read_csv("mlb_standings.csv")
    teams = standings_df["Team"].tolist()
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    idx_to_team = {idx: team for team, idx in team_to_idx.items()}
    num_teams = len(teams)
    # --- Division ratings based on win percentages ---
    division_totals = standings_df.groupby("Division Name")[["Wins", "Losses"]].sum()
    division_strength = pd.Series(
        division_totals["Wins"] / (division_totals["Wins"] + division_totals["Losses"]) - 0.5
    ).to_dict()
    team_division_bonus = {
        team_to_idx[team]: division_strength.get(TEAM_TO_DIVISION.get(team, ""), 0.0)
        for team in teams
    }
    division_bonus_tensor = torch.tensor(
        [team_division_bonus[i] for i in range(num_teams)], dtype=torch.float32
    ).view(num_teams, 1)
    # --- Setup Elo model ---
    elo_ratings = torch.nn.Embedding(num_teams, 1)
    optimizer = optim.Adam(elo_ratings.parameters(), lr=LEARNING_RATE)
    if os.path.exists("elo_model.pt") and not RETRAIN:
        print("Loading saved Elo model...")
        elo_ratings.load_state_dict(torch.load("elo_model.pt"))
    else:
        print("Initializing Elo ratings to 1500...")
        torch.nn.init.constant_(elo_ratings.weight, 1500.0)
    # --- Parse and save real game outcomes ---
    outcome_data = fetch_json(GAME_OUTCOME_URL)
    games = parse_game_outcomes(outcome_data)
    if not os.path.exists("mlb_game_results.csv"):
        with open("mlb_game_results.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["Date", "Home Team", "Away Team", "Home Score", "Away Score"],
            )
            writer.writeheader()
            writer.writerows(games)
    # --- Train Elo model ---
    if not os.path.exists("elo_model.pt") or RETRAIN:
        print("Training Elo model...")
        real_matches = generate_real_matches(games, team_to_idx)
        train_on_matches(
            real_matches,
            epochs=10000,
            label="Real Matches",
            batch_size=BATCH_SIZE,
            elo_ratings=elo_ratings,
            division_bonus_tensor=division_bonus_tensor,
            optimizer=optimizer,
        )
        print("Saving Elo model...")
        torch.save(elo_ratings.state_dict(), "elo_model.pt")
        final_ratings = {
            team: elo_ratings(torch.tensor(idx)).item() for team, idx in team_to_idx.items()
        }
        final_df = pd.DataFrame(final_ratings.items(), columns=["Team", "Elo Rating"])
        final_df = final_df.sort_values(by="Elo Rating", ascending=False)
        final_df.to_csv("final_elo_ratings.csv", index=False)
    # --- Simulate future schedule ---
    print("Simulating real MLB 2025 scheduled games with Elo-based winners...")
    scheduled_games = fetch_future_schedule(team_to_idx)
    synthetic_matches = generate_synthetic_matches_from_schedule(
        scheduled_games, elo_ratings, division_bonus_tensor
    )
    save_synthetic_matches_to_csv(synthetic_matches, idx_to_team, "simulated_real_schedule_outcomes.csv")
    standings_from_sim = generate_standings_from_simulated_matches(
        synthetic_matches, teams, idx_to_team
    )
    standings_from_sim.to_csv("simulated_standings.csv", index=False)
    print("Saved simulated standings to simulated_standings.csv")

    # --- Bayesian Elo Backtest Optimization ---
    best_logloss = float('inf')
    best_K = None
    best_T = None

    for K in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        for T in [0.5, 1.0, 1.5, 2.0]:
            # Run your Bayesian Elo backtest here, passing K and T
            logloss = backtest_bayesian_elo_on_season(season=2025, K=K, T=T)
            print(f"K={K}, T={T}, Log Loss={logloss}")
            if logloss < best_logloss:
                best_logloss = logloss
                best_K = K
                best_T = T

    print(f"Best K: {best_K}, Best T: {best_T}, Best Log Loss: {best_logloss}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backtest', type=int, help='Run Elo backtest for a given season (e.g., 2023)')
    parser.add_argument('--bayes-backtest', type=int, help='Run Bayesian Elo backtest for a given season (e.g., 2023)')
    args = parser.parse_args()
    if args.bayes_backtest:
        backtest_bayesian_elo_on_season(args.bayes_backtest)
    elif args.backtest:
        backtest_elo_on_season(args.backtest)
    else:
        main()
