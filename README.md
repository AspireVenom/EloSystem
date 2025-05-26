# ⚾ MLB Elo Rating System — Full Documentation

## Overview

This program builds an Elo-based rating system for Major League Baseball (MLB) teams using real 2025 game outcomes from the [MLB Stats API](https://statsapi.mlb.com) and synthetic match simulations. The goal is to estimate team strength dynamically and visualize the evolution of ratings.

It follows a **three-phase training strategy**:

1. **Real match training** from actual game outcomes.
2. **Synthetic match simulation** based on Elo probabilities.
3. **Post-training visualization** comparing Elo after each phase.

---

## Features

- Retrieves real match outcomes via MLB API.
- Downloads current team standings.
- Initializes all team Elo ratings at 1500.
- Trains Elo ratings on actual games (2025 season).
- Generates synthetic matches based on current Elo for further training.
- Saves CSV snapshots of Elo after each training phase.
- Visualizes Elo changes with side-by-side bar charts.

---

## Dependencies

- Python 3.8+
- `requests`
- `pandas`
- `torch`
- `matplotlib`

Install dependencies via:

```bash
pip install requests pandas torch matplotlib
```

---

## Data Sources

- **Team Standings (2024):**

  ```text
  https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2024
  ```

- **Real Match Results (2025):**

  ```text
  https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2025&gameType=R
  ```

---

## File Outputs

- `mlb_standings.csv`: Team names, win/loss records, and division rank.
- `mlb_game_results.csv`: Historical game results with scores.
- `elo_after_real_training.csv`: Elo ratings after only real-match training.
- `final_elo_ratings.csv`: Final Elo ratings after both real and synthetic training.
- (Optional) `elo_rating_comparison.png`: Side-by-side bar chart of rating shifts.

---

## Program Flow

### 1. Data Collection

- **Standings:** Fetches team stats and stores them in `mlb_standings.csv`.
- **Game Outcomes:** Iterates over daily games and stores final game results in `mlb_game_results.csv`.

### 2. Preprocessing

- Builds a team name-to-index mapping for Elo matrix.
- Initializes all Elo scores to 1500.
- Uses PyTorch `Embedding` to represent team ratings.

### 3. Elo Training

#### A. Real Matches Training

```python
train_on_real_matches(real_matches, epochs=1000)
```

- Each match outcome updates the predicted win probability via the Elo formula.
- Uses weighted binary cross-entropy loss where weight is based on Elo gap.
- Trained ratings saved to `elo_after_real_training.csv`.

#### B. Synthetic Match Simulation

```python
generate_synthetic_matches(matches_per_pair=10)
train_on_synthetic_matches(synthetic_matches, epochs=500)
```

- Generates N synthetic matches per team-pair using their current Elo ratings.
- Repeats training using these synthetic match outcomes to refine scores.

### 4. Output

- Saves post-training Elo ratings to `final_elo_ratings.csv`.

### 5. Visualization (Optional)

```python
import matplotlib.pyplot as plt
```

- Compares Elo ratings **after real match training** vs **after full training** with a grouped bar chart.

---

## Elo Model Design

### `elo_probability(r1, r2)`

```python
1 / (1 + exp((r2 - r1) * ln(10) / 400))
```

Predicts the probability that team 1 beats team 2 based on current ratings.

### Loss Function

```python
F.binary_cross_entropy(pred, target, weight=...)
```

- Targets are 1.0 if team1 wins, else 0.0.
- `weight` increases proportionally with Elo rating gap.

---

## Known Limitations

- Synthetic matches are randomly generated and may not fully reflect real-world scheduling.
- No use of home-field advantage, starting pitchers, or player-level features yet.
- Real 2025 standings data is not used to initialize Elo — all teams start at 1500.
- Doesn't persist model state across sessions (you re-train from scratch).

---

## Potential Improvements

- Add home-field advantage modifier.
- Integrate starting pitcher WAR or ERA as additional factors.
- Use historical team ratings as priors instead of flat 1500.
- Add seasonal decay to prevent overfitting to early-season games.
- Incorporate team-level stats like run differential or batting average.

---

## Conclusion

This program is a scalable foundation for rating MLB teams using Elo logic. It intelligently combines real and synthetic match data to dynamically adjust team strengths and outputs visualizations to monitor rating changes.
