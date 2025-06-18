"""
optimize_hyperparams.py

Bayesian optimization of Elo model hyperparameters for MLB simulation using Optuna.
"""
import os
import random
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import optuna
from sklearn.model_selection import train_test_split
import optuna.visualization as vis

# --- Constants and Data Loading ---
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

def load_game_data():
    df = pd.read_csv("mlb_game_results.csv")
    return df

def load_standings():
    df = pd.read_csv("mlb_standings.csv")
    teams = df["Team"].tolist()
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    idx_to_team = {idx: team for team, idx in team_to_idx.items()}
    return teams, team_to_idx, idx_to_team

def generate_real_matches(game_df, team_to_idx):
    matches = []
    for _, game in game_df.iterrows():
        home = game["Home Team"]
        away = game["Away Team"]
        home_score = game["Home Score"]
        away_score = game["Away Score"]
        if home not in team_to_idx or away not in team_to_idx:
            continue
        team1 = team_to_idx[home]  # home
        team2 = team_to_idx[away]  # away
        winner = team1 if home_score > away_score else team2
        matches.append((team1, team2, winner, True))
    return matches

def get_division_bonus_tensor(teams, team_to_idx, standings_df, division_scale):
    division_totals = standings_df.groupby("Division Name")[["Wins", "Losses"]].sum()
    division_strength = pd.Series(
        division_totals["Wins"] / (division_totals["Wins"] + division_totals["Losses"]) - 0.5
    ).to_dict()
    team_division_bonus = {
        team_to_idx[team]: division_strength.get(TEAM_TO_DIVISION.get(team, ""), 0.0)
        for team in teams
    }
    bonus_tensor = torch.tensor(
        [team_division_bonus[i] for i in range(len(teams))], dtype=torch.float32
    ).view(len(teams), 1)
    return bonus_tensor

def elo_probability(r1, r2):
    return 1 / (1 + torch.exp((r2 - r1) * torch.log(torch.tensor(10.0)) / 50))

def train_and_evaluate(home_advantage, division_scale, learning_rate, batch_size):
    # Load data
    standings_df = pd.read_csv("mlb_standings.csv")
    teams, team_to_idx, idx_to_team = load_standings()
    num_teams = len(teams)
    game_df = load_game_data()
    matches = generate_real_matches(game_df, team_to_idx)
    # Split into train/val
    train_matches, val_matches = train_test_split(matches, test_size=0.2, random_state=42)
    # Division bonus tensor
    division_bonus_tensor = get_division_bonus_tensor(teams, team_to_idx, standings_df, division_scale)
    # Model
    elo_ratings = torch.nn.Embedding(num_teams, 1)
    torch.nn.init.constant_(elo_ratings.weight, 1500.0)
    optimizer = optim.Adam(elo_ratings.parameters(), lr=learning_rate)
    # Training
    match_tensor = torch.tensor(train_matches, dtype=torch.long)
    for epoch in range(2000):
        match_tensor = match_tensor[torch.randperm(len(match_tensor))]
        for batch_start in range(0, len(match_tensor), batch_size):
            batch = match_tensor[batch_start : batch_start + batch_size]
            team1_idx = batch[:, 0]
            team2_idx = batch[:, 1]
            winner_idx = batch[:, 2]
            is_home_team1 = batch[:, 3].bool()
            rating1 = elo_ratings(team1_idx)
            rating2 = elo_ratings(team2_idx)
            rating1 += home_advantage * is_home_team1.unsqueeze(1)
            rating2 += home_advantage * (~is_home_team1).unsqueeze(1)
            rating1 += division_scale * division_bonus_tensor[team1_idx]
            rating2 += division_scale * division_bonus_tensor[team2_idx]
            pred = elo_probability(rating1, rating2)
            target = (winner_idx == team1_idx).float().unsqueeze(1)
            elo_gap = torch.abs(rating1 - rating2)
            weight = (1.0 + (elo_gap / 50)).detach()
            loss = F.binary_cross_entropy(pred, target, weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Validation
    val_tensor = torch.tensor(val_matches, dtype=torch.long)
    team1_idx = val_tensor[:, 0]
    team2_idx = val_tensor[:, 1]
    winner_idx = val_tensor[:, 2]
    is_home_team1 = val_tensor[:, 3].bool()
    rating1 = elo_ratings(team1_idx)
    rating2 = elo_ratings(team2_idx)
    rating1 += home_advantage * is_home_team1.unsqueeze(1)
    rating2 += home_advantage * (~is_home_team1).unsqueeze(1)
    rating1 += division_scale * division_bonus_tensor[team1_idx]
    rating2 += division_scale * division_bonus_tensor[team2_idx]
    pred = elo_probability(rating1, rating2)
    target = (winner_idx == team1_idx).float().unsqueeze(1)
    val_loss = F.binary_cross_entropy(pred, target).item()
    return val_loss

def objective(trial):
    home_advantage = trial.suggest_uniform('home_advantage', 10, 50)
    division_scale = trial.suggest_uniform('division_scale', 10, 100)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    return train_and_evaluate(home_advantage, division_scale, learning_rate, batch_size)

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)
    vis.plot_optimization_history(study)
    vis.plot_parallel_coordinate(study)
    vis.plot_slice(study)
    vis.plot_contour(study)
    vis.plot_param_importances(study)
    plt.show()

if __name__ == "__main__":
    main() 