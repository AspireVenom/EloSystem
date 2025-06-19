import optuna
import numpy as np
from scipy.stats import norm
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
import pandas as pd
from llm_int import fetch_completed_games_for_season, HOME_ADVANTAGE

def bayesian_elo_logloss(season, K, initial_elo=1500.0, initial_sigma2=2000.0**2, min_sigma2=1000.0, T=1.0):
    games = fetch_completed_games_for_season(season)
    if not games:
        print("No completed games found for this season.")
        return 10.0  # Return a high log loss if no games
    teams = sorted(set(g["home_team"] for g in games).union(g["away_team"] for g in games))
    team_mu = {team: initial_elo for team in teams}
    team_sigma2 = {team: initial_sigma2 for team in teams}
    predictions = []
    actuals = []
    for g in sorted(games, key=lambda x: x["date"]):
        home, away = g["home_team"], g["away_team"]
        mu_diff = team_mu[home] + HOME_ADVANTAGE - team_mu[away]
        sigma2_sum = team_sigma2[home] + team_sigma2[away]
        prob_home_win = norm.cdf((mu_diff / np.sqrt(sigma2_sum)) / T)
        epsilon = 1e-6
        clamped_prob = min(max(prob_home_win, epsilon), 1 - epsilon)
        predictions.append(clamped_prob)
        actuals.append(1 if g["home_score"] > g["away_score"] else 0)
        result = 1 if g["home_score"] > g["away_score"] else 0
        expected = prob_home_win
        delta = K * (result - expected)
        team_mu[home] += delta * team_sigma2[home] / sigma2_sum
        team_mu[away] -= delta * team_sigma2[away] / sigma2_sum
        team_sigma2[home] = max(1 / (1/team_sigma2[home] + 1/sigma2_sum), min_sigma2)
        team_sigma2[away] = max(1 / (1/team_sigma2[away] + 1/sigma2_sum), min_sigma2)
    return log_loss(actuals, predictions)

def objective(trial):
    K = trial.suggest_float('K', 0.01, 2.0, log=True)
    logloss = bayesian_elo_logloss(season=2024, K=K)
    print(f"Trial K={K:.4f}, Log Loss={logloss:.4f}")
    return logloss

def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    print("Best K:", study.best_params['K'])
    print("Best log loss:", study.best_value)

if __name__ == "__main__":
    main() 