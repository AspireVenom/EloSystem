# ⚾ MLB Elo Rating System — Full Documentation

## Overview

This program builds a dynamic **Elo-based rating system** for Major League Baseball (MLB) teams using real 2025 game outcomes from the [MLB Stats API](https://statsapi.mlb.com), along with a post-training match simulation engine.

The system now focuses exclusively on **training using real match outcomes**, ensuring stable and interpretable Elo values. Synthetic matches are generated **after training**, using Elo probabilities to simulate full-season outcomes.

---

## 🔍 Key Changes from Previous Version

- 🔁 **Real-only training**: Elo is trained _only_ on actual 2025 game results.
- ⚖️ **Batch learning with PyTorch**: Uses embedding layers + gradient-based updates with gap-weighted loss.
- 🎯 **Home-field advantage** built into probability: +28.2 Elo points added to home teams.
- 🔢 **Post-training synthetic season**: Simulates outcomes for all scheduled 2025 games.
- 📈 **Dash-based visualization**: Interactive web UI comparing Elo ratings and simulated wins.

---

## Features

- Fetches real 2025 MLB results & standings via API.
- Initializes all Elo ratings to 1500.
- Trains using only real, final game outcomes.
- Applies gap-weighted binary cross-entropy loss.
- Simulates full 2025 season schedule based on trained Elo probabilities.
- Saves results:

  - Per-match synthetic outcomes
  - Team standings from simulations
  - Final Elo ratings

- Visualizes Elo vs Wins using Dash + Plotly.

---

## Dependencies

- Python 3.8+
- `requests`
- `pandas`
- `torch`
- `dash`
- `plotly`

Install:

```bash
pip install requests pandas torch dash plotly
```

---

## Data Sources

- **Game Results:**
  `https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2025&gameType=R`

- **Future Schedule for Simulation:**
  `https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=2025-03-28&endDate=2025-09-28`

- **Team Standings:**
  `https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2025`

---

## File Outputs

| File                           | Description                                  |
| ------------------------------ | -------------------------------------------- |
| `mlb_standings.csv`            | Actual team standings data from API          |
| `mlb_game_results.csv`         | Game-by-game results from real 2025 season   |
| `final_elo_ratings.csv`        | Trained Elo ratings after real-only training |
| `simulated_future_matches.csv` | Simulated outcomes of full 2025 season       |
| `simulated_standings.csv`      | Team-level wins/losses from simulation       |

---

## Program Flow

### 1. Load Data

- Pulls real results + standings from MLB Stats API
- Initializes team rating embeddings

### 2. Train Elo Ratings

```python
train_on_matches(real_matches, epochs=10000)
```

- Uses batch training with PyTorch
- Applies home-field advantage as Elo offset
- Uses gap-weighted binary cross-entropy loss
- Saves ratings to `final_elo_ratings.csv`

### 3. Simulate 2025 Season

```python
generate_synthetic_matches_from_schedule()
```

- Uses real MLB 2025 schedule
- Applies trained Elo ratings to calculate win probabilities
- Simulates winner via `random.random() < P_win`
- Saves full match outcomes to `simulated_future_matches.csv`

### 4. Generate Simulated Standings

```python
generate_standings_from_simulated_matches()
```

- Tallies win/loss records per team
- Sorts and ranks by division
- Outputs to `simulated_standings.csv`

---

## Elo Model Design

### `elo_probability(r1, r2)`

```python
1 / (1 + 10^((r2 - r1) / 400))
```

- Predicts win probability for team 1
- Elo difference includes +28.2 for home advantage

### Loss Function

```python
F.binary_cross_entropy(pred, target, weight=...)
```

- `target = 1.0` if team 1 wins, else `0.0`
- `weight = 1 + (|r1 - r2| / 50)` to emphasize confident errors

---

## Visualization

### 📊 Dash App

Run the interactive dashboard:

```bash
python app.py
```

It displays:

- Bar chart of **Relative Elo Ratings (Elo − 1500)**
- Bar chart of **Simulated Wins**
- Division-based color grouping
- Hover tooltips with full team info

---

## Known Limitations

- Elo does not yet incorporate time decay or prior season context.
- Schedule simulation assumes all games play out (no cancellations).
- Only team Elo is used — no player-level or stat-based inputs.

---

## Future Improvements

- Add rolling Elo tracking over time
- Integrate starter pitcher or lineup data
- Visualize match-level predictions with confidence intervals
- Add time-decayed training and seasonal resets
- Build playoff simulation using simulated standings

---

## Conclusion

This project models MLB team performance using real match data and an Elo-based probabilistic engine.
With PyTorch embeddings and simulation logic built-in, it creates a flexible platform for exploring team dynamics, match predictions, and analytics visualization.

Feel free to explore or fork the repo to test new ideas in ranking systems, probability modeling, or sports analytics.
