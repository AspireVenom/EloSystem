# ⚾ MLB Elo Rating System — Advanced Analytics Platform

## Overview

This project implements a comprehensive **Elo-based rating system** for Major League Baseball (MLB) with both classic and Bayesian approaches. It features real-time data fetching, advanced model training, historical backtesting, hyperparameter optimization, and interactive visualizations.

The system now includes:
- **Classic Elo**: Traditional Elo ratings with PyTorch-based training
- **Bayesian Elo**: Probabilistic team strength modeling with uncertainty quantification (mean, variance, stddev)
- **Historical Backtesting**: Model evaluation on past seasons (2024, 2025)
- **Hyperparameter Optimization**: Automated tuning using Optuna and grid search (K-factor, temperature)
- **Interactive Dashboard**: Dash-based visualization with Elo uncertainty, reliability diagrams, and more

---

## 🔍 Key Features

### 🎯 **Dual Elo Models**
- **Classic Elo**: Traditional rating system with gradient-based updates
- **Bayesian Elo**: Team strengths modeled as distributions (mean + variance)
- **Uncertainty Quantification**: Confidence intervals and Elo stddev visualized in dashboard
- **Probability Clamping**: Prevents overconfident predictions for better calibration
- **Minimum Variance**: Ensures uncertainty does not collapse unrealistically

### 📊 **Advanced Analytics**
- **Historical Backtesting**: Evaluate models on 2024 and 2025 season data
- **Model Calibration**: Reliability diagrams and calibration curves
- **Performance Metrics**: Log loss, accuracy, Brier score
- **Elo Trajectories**: Track team rating evolution over time
- **Uncertainty Visualization**: Elo stddev shown in dashboard plots

### 🤖 **Automated Optimization**
- **Optuna Integration**: Bayesian hyperparameter optimization
- **Grid Search**: For K-factor and temperature (T) in Bayesian Elo
- **K-factor & Temperature Tuning**: Optimize learning rate and probability scaling for both models

### 📈 **Interactive Visualizations**
- **Elo Rating Comparisons**: Current vs. historical ratings
- **Simulated Standings**: Win/loss projections
- **Calibration Plots**: Model reliability assessment
- **Trajectory Analysis**: Team rating evolution over time
- **Uncertainty Bands**: Elo stddev shown as shaded regions or error bars

---

## Dependencies

```bash
pip install requests pandas torch dash plotly scikit-learn scipy numpy matplotlib optuna
```

**Core Dependencies:**
- `torch` - PyTorch for neural network training
- `dash` & `plotly` - Interactive web dashboard
- `scikit-learn` - Model evaluation metrics
- `optuna` - Hyperparameter optimization
- `scipy` - Statistical functions for Bayesian Elo

---

## Quick Start

### 1. **Setup Environment**
```bash
# Clone and setup
git clone https://github.com/AspireVenom/EloSystem.git
cd win_calculator
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. **Run Main Simulation**
```bash
python llm_int.py
```
This will:
- Fetch current MLB data
- Train both classic and Bayesian Elo models
- Generate predictions and simulations
- Save results to CSV files

### 3. **Launch Dashboard**
```bash
python app.py
```
Open http://127.0.0.1:8050/ in your browser

### 4. **Optimize Hyperparameters**
```bash
python optimize_bayes_k.py  # Bayesian Elo optimization
python optimize_hyperparams.py  # Classic Elo optimization
```

### 5. **Backtest or Bayesian Backtest**
```bash
python llm_int.py --backtest 2024         # Classic Elo backtest for 2024
python llm_int.py --bayes-backtest 2024   # Bayesian Elo backtest for 2024
```

---

## Data Sources

- **Current Season**: MLB Stats API (2025 season)
- **Historical Data**: MLB Stats API (2024 season for backtesting)
- **Schedule Data**: Future games for simulation

**API Endpoints:**
- Game Results: `https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2025&gameType=R`
- Standings: `https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2025`
- Historical: `https://statsapi.mlb.com/api/v1/schedule?sportId=1&season=2024&gameType=R`

---

## File Structure

### **Core Application Files**
| File | Description |
|------|-------------|
| `llm_int.py` | Main application with both Elo models and CLI for backtesting |
| `app.py` | Interactive Dash dashboard |
| `optimize_bayes_k.py` | Bayesian Elo hyperparameter optimization (K, T) |
| `optimize_hyperparams.py` | Classic Elo hyperparameter optimization |

### **Output Files**
| File | Description |
|------|-------------|
| `final_elo_ratings.csv` | Trained classic Elo ratings |
| `elo_history.csv` | Historical Elo trajectories |
| `pred_vs_actual.csv` | Prediction vs. actual outcomes |
| `simulated_standings.csv` | Simulated season standings |
| `elo_history_bayes.csv` | Bayesian Elo trajectories (mean, stddev) |
| `pred_vs_actual_bayes.csv` | Bayesian predictions vs. actual |

### **Model Files**
| File | Description |
|------|-------------|
| `elo_model.pt` | Saved PyTorch model |
| `test_calibration.png` | Model calibration plot |
| `test_elo_trajectories.png` | Elo trajectory visualization |

---

## Model Architecture

### **Classic Elo Model**
```python
def elo_probability(r1, r2):
    return 1 / (1 + torch.exp((r2 - r1) * torch.log(torch.tensor(10.0)) / 50))
```

**Features:**
- PyTorch embeddings for team ratings
- Home field advantage (+10.24 Elo points)
- Division-based rating adjustments
- Gap-weighted binary cross-entropy loss

### **Bayesian Elo Model**
```python
def bayesian_elo_update(team_mu, team_sigma2, result, expected, K, T):
    delta = K * (result - expected)
    team_mu += delta * team_sigma2 / sigma2_sum
    team_sigma2 = max(1 / (1/team_sigma2 + 1/sigma2_sum), min_sigma2)
    # Probability clamping and minimum variance for calibration
```

**Features:**
- Team strength as normal distribution (μ, σ²)
- Uncertainty quantification (stddev visualized)
- Adaptive learning rates
- Probability calibration (temperature scaling, clamping)
- Minimum variance for stable uncertainty

---

## Dashboard Features

### **Current Season Analysis**
- **Relative Elo Ratings**: Team ratings centered at 1500
- **Simulated Wins**: Projected season outcomes
- **Division Grouping**: Color-coded by division
- **Elo Uncertainty**: Elo stddev shown as error bars or shaded bands

### **Historical Backtesting**
- **Elo Trajectories**: Interactive team rating evolution
- **Prediction Calibration**: Reliability diagrams
- **Performance Metrics**: Log loss, accuracy analysis
- **Uncertainty Over Time**: Elo stddev for each team

### **Interactive Controls**
- **Team Selection**: Multi-select dropdown for trajectory plots
- **Date Filtering**: Focus on specific time periods
- **Real-time Updates**: Dynamic plot generation

---

## Model Evaluation

### **Performance Metrics**
- **Log Loss**: Measures prediction quality (lower is better)
- **Accuracy**: Percentage of correct predictions
- **Brier Score**: Calibration quality assessment
- **Calibration Curve**: Reliability diagram

### **Current Results**
- **Classic Elo**: Log loss ~0.69, optimized K-factor
- **Bayesian Elo**: Log loss ~0.68, improved calibration and uncertainty quantification
- **Calibration**: Both models show good reliability; Bayesian Elo now visualizes uncertainty

---

## Advanced Features

### **Hyperparameter Optimization**
```bash
# Optimize Bayesian Elo K-factor and temperature
python optimize_bayes_k.py

# Optimize Classic Elo parameters
python optimize_hyperparams.py
```

**Optimized Parameters:**
- Home advantage: 10.24 Elo points
- Division rating scale: 71.15
- Learning rate: 0.00125
- Batch size: 32
- Bayesian K-factor and temperature (T): grid search and Optuna supported

### **Monte Carlo Simulation**
- Full season simulation using trained models
- Uncertainty propagation through predictions
- Multiple simulation runs for robust estimates

### **Data Pipeline**
- Automated data fetching from MLB API
- Real-time standings updates
- Historical data integration for backtesting

---

## Interpreting Elo Stddev (Uncertainty)

- **Elo stddev** represents the model's uncertainty about a team's true strength.
- In the dashboard, higher stddev means less confidence in the rating; lower stddev means more certainty.
- Use stddev to compare not just which team is rated higher, but how confident the model is in that rating.
- Uncertainty bands are especially useful early in the season or for teams with few games played.

---

## Usage Examples

### **Run Complete Analysis**
```python
from llm_int import main
main()  # Fetches data, trains models, generates predictions
```

### **Backtest on Historical Data**
```python
from llm_int import backtest_bayesian_elo_on_season
results = backtest_bayesian_elo_on_season(2024, K=0.1, T=1.0)
```

### **Optimize Hyperparameters**
```python
import optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
```

---

## Troubleshooting

- **Dash Port Conflicts**: If you see "Port 8050 is in use", either kill the process using that port or change the port in `app.py` (e.g., `app.run_server(port=8051)`).
- **Deprecated Dash API Usage**: If you see warnings about deprecated Dash features, update your Dash version and check the [Dash migration guide](https://dash.plotly.com/).
- **Python Environment Activation**: Ensure your virtual environment is activated before running scripts.
- **API Rate Limits**: If you hit MLB API rate limits, try again later or cache results locally.

---

## Future Enhancements

### **Planned Features**
- **Player-level Elo**: Individual player ratings
- **Time Decay**: Seasonal rating adjustments
- **Playoff Simulation**: Post-season predictions
- **Real-time Updates**: Live game integration

### **Advanced Modeling**
- **Glicko-2**: Alternative rating system
- **TrueSkill**: Microsoft's rating algorithm
- **Ensemble Methods**: Combine multiple models
- **Deep Learning**: Neural network extensions

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## License

This project is open source. Feel free to use, modify, and distribute according to your needs.

---

## Acknowledgments

- **MLB Stats API** for providing game data
- **PyTorch** for deep learning framework
- **Dash** for interactive visualizations
- **Optuna** for hyperparameter optimization

---

*Last updated: June 2025*
