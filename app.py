import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# --- Load Data ---
print("Loading data files...")

# Core data
elo_df = pd.read_csv("final_elo_ratings.csv")
standings_df = pd.read_csv("simulated_standings.csv")
mlb_standings_df = pd.read_csv("mlb_standings.csv")

# Simulation data
simulated_games_df = pd.read_csv("simulated_real_schedule_outcomes.csv")
synthetic_predictions_df = pd.read_csv("synthetic_match_predictions.csv")

# Historical/backtest data
try:
    elo_traj_df = pd.read_csv("elo_history.csv")
    pred_df = pd.read_csv("pred_vs_actual.csv")
    elo_traj_bayes_df = pd.read_csv("elo_history_bayes.csv")
    pred_bayes_df = pd.read_csv("pred_vs_actual_bayes.csv")
    print(f"Loaded historical data: {len(elo_traj_df)} classic, {len(elo_traj_bayes_df)} Bayesian records")
except Exception as e:
    print(f"Error loading historical data: {e}")
    elo_traj_df = pd.DataFrame(columns=["date", "team", "elo"])
    pred_df = pd.DataFrame(columns=["prob", "actual"])
    elo_traj_bayes_df = pd.DataFrame(columns=["date", "team", "mu", "sigma"])
    pred_bayes_df = pd.DataFrame(columns=["prob", "actual"])

# --- Data Processing ---
# Add relative Elo ratings
mean_elo = elo_df["Elo Rating"].mean()
elo_df["Relative Elo"] = elo_df["Elo Rating"] - mean_elo

# Merge standings data
merged_df = pd.merge(elo_df, standings_df, on="Team")
merged_df = merged_df.sort_values(by="Relative Elo", ascending=False)

# Process simulation data
if not simulated_games_df.empty:
    simulated_games_df['Game_Index'] = range(len(simulated_games_df))
    simulated_games_df['Cumulative_Home_Wins'] = simulated_games_df['Home Win'].cumsum()
    simulated_games_df['Cumulative_Away_Wins'] = (~simulated_games_df['Home Win']).cumsum()

# Process synthetic predictions
if not synthetic_predictions_df.empty:
    synthetic_predictions_df['Win_Probability_Diff'] = synthetic_predictions_df['Team 1 Win Prob'] - synthetic_predictions_df['Team 2 Win Prob']
    synthetic_predictions_df['Expected_Winner'] = synthetic_predictions_df['Team 1 Win Prob'] > synthetic_predictions_df['Team 2 Win Prob']

# --- Create Dash App ---
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "MLB 2025 — Advanced Elo Simulation Dashboard"

# Configure app to handle callback exceptions gracefully
app.config.suppress_callback_exceptions = True

# Get unique teams for dropdowns
unique_teams = sorted(elo_df["Team"].unique()) if not elo_df.empty else []
default_teams = unique_teams[:5] if len(unique_teams) >= 5 else unique_teams

# Helper functions for calibration metrics
def calculate_calibration_error(pred_df, threshold=0.7):
    """Calculate calibration error for predictions above threshold."""
    if pred_df.empty:
        return 0
    
    high_conf_mask = pred_df['prob'] > threshold
    if not np.any(high_conf_mask):
        return 0
    
    high_conf_preds = pred_df[high_conf_mask]
    return np.mean(np.abs(high_conf_preds['prob'] - high_conf_preds['actual']))

def calculate_calibrated_error(pred_df, threshold=0.7):
    """Calculate calibration error after isotonic calibration."""
    if pred_df.empty:
        return 0
    
    calibrated_probs = apply_isotonic_calibration(
        pred_df['prob'].values, 
        pred_df['actual'].values
    )
    
    high_conf_mask = calibrated_probs > threshold
    if not np.any(high_conf_mask):
        return 0
    
    return np.mean(np.abs(calibrated_probs[high_conf_mask] - pred_df['actual'].values[high_conf_mask]))

def calculate_improvement(pred_df, threshold=0.7):
    """Calculate percentage improvement in calibration error."""
    if pred_df.empty:
        return 0
    
    original_error = calculate_calibration_error(pred_df, threshold)
    calibrated_error = calculate_calibrated_error(pred_df, threshold)
    
    if original_error == 0:
        return 0
    
    return (original_error - calibrated_error) / original_error

def calculate_brier_score(pred_df):
    """Calculate Brier score for original predictions."""
    if pred_df.empty:
        return 0
    
    return np.mean((pred_df['prob'] - pred_df['actual'])**2)

def calculate_calibrated_brier_score(pred_df):
    """Calculate Brier score for calibrated predictions."""
    if pred_df.empty:
        return 0
    
    calibrated_probs = apply_isotonic_calibration(
        pred_df['prob'].values, 
        pred_df['actual'].values
    )
    
    return np.mean((calibrated_probs - pred_df['actual'].values)**2)

def calculate_brier_improvement(pred_df):
    """Calculate percentage improvement in Brier score."""
    if pred_df.empty:
        return 0
    
    original_brier = calculate_brier_score(pred_df)
    calibrated_brier = calculate_calibrated_brier_score(pred_df)
    
    if original_brier == 0:
        return 0
    
    return (original_brier - calibrated_brier) / original_brier

def apply_isotonic_calibration(probabilities, actuals, variance_constraint=0.01):
    """
    Apply isotonic regression with variance constraints to improve calibration.
    
    Args:
        probabilities: Predicted probabilities
        actuals: Actual outcomes (0 or 1)
        variance_constraint: Maximum allowed variance in calibrated probabilities
    
    Returns:
        Calibrated probabilities
    """
    if len(probabilities) == 0:
        return probabilities
    
    # Sort by predicted probability
    sorted_indices = np.argsort(probabilities)
    sorted_probs = probabilities[sorted_indices]
    sorted_actuals = actuals[sorted_indices]
    
    # Apply isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    calibrated_probs = iso_reg.fit_transform(sorted_probs, sorted_actuals)
    
    # Apply variance constraints to prevent overfitting
    # Smooth high-confidence predictions (>0.7) to reduce drop-off
    high_conf_mask = calibrated_probs > 0.7
    if np.any(high_conf_mask):
        # Calculate empirical variance in high-confidence region
        high_conf_actuals = sorted_actuals[high_conf_mask]
        high_conf_probs = calibrated_probs[high_conf_mask]
        
        if len(high_conf_actuals) > 1:
            empirical_variance = np.var(high_conf_actuals)
            
            # If variance is too low, add regularization
            if empirical_variance < variance_constraint:
                # Apply conservative smoothing to high-confidence predictions
                smoothing_factor = 0.1
                calibrated_probs[high_conf_mask] = (
                    calibrated_probs[high_conf_mask] * (1 - smoothing_factor) + 
                    np.mean(high_conf_actuals) * smoothing_factor
                )
    
    # Ensure monotonicity is preserved
    for i in range(1, len(calibrated_probs)):
        if calibrated_probs[i] < calibrated_probs[i-1]:
            calibrated_probs[i] = calibrated_probs[i-1]
    
    # Return to original order
    result = np.zeros_like(calibrated_probs)
    result[sorted_indices] = calibrated_probs
    
    return result

def create_calibrated_reliability_figure(pred_df, title, color='blue'):
    """
    Create reliability diagram with isotonic calibration applied.
    """
    if pred_df.empty:
        return px.scatter(title="No data available")
    
    # Apply isotonic calibration
    calibrated_probs = apply_isotonic_calibration(
        pred_df['prob'].values, 
        pred_df['actual'].values,
        variance_constraint=0.01
    )
    
    # Create bins for reliability diagram
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    actual_rates = []
    calibrated_rates = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = (pred_df['prob'] >= bins[i]) & (pred_df['prob'] < bins[i + 1])
        if mask.sum() > 0:
            actual_rates.append(pred_df[mask]['actual'].mean())
            calibrated_rates.append(np.mean(calibrated_probs[mask]))
            bin_counts.append(mask.sum())
        else:
            actual_rates.append(0)
            calibrated_rates.append(0)
            bin_counts.append(0)
    
    # Create figure with both original and calibrated predictions
    fig = go.Figure()
    
    # Original predictions
    fig.add_trace(go.Scatter(
        x=bin_centers, 
        y=actual_rates, 
        mode='lines+markers', 
        name='Original Predictions', 
        line=dict(color=color, width=2),
        marker=dict(size=8)
    ))
    
    # Calibrated predictions
    fig.add_trace(go.Scatter(
        x=bin_centers, 
        y=calibrated_rates, 
        mode='lines+markers', 
        name='Calibrated Predictions', 
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1], 
        y=[0, 1], 
        mode='lines', 
        name='Perfect Calibration', 
        line=dict(color='gray', dash='dot', width=1)
    ))
    
    # Add confidence intervals for high-confidence region
    high_conf_mask = np.array(bin_centers) > 0.7
    if np.any(high_conf_mask):
        high_conf_centers = np.array(bin_centers)[high_conf_mask]
        high_conf_actuals = np.array(actual_rates)[high_conf_mask]
        high_conf_counts = np.array(bin_counts)[high_conf_mask]
        
        # Calculate confidence intervals
        confidence_intervals = []
        for i, (center, actual, count) in enumerate(zip(high_conf_centers, high_conf_actuals, high_conf_counts)):
            if count > 0:
                # Wilson confidence interval
                z = 1.96  # 95% confidence
                denominator = 1 + z**2/count
                centre_adjusted = (actual + z*z/(2*count)) / denominator
                error_adjusted = z * np.sqrt((actual * (1-actual) + z*z/(4*count))/count) / denominator
                confidence_intervals.append((centre_adjusted - error_adjusted, centre_adjusted + error_adjusted))
            else:
                confidence_intervals.append((0, 0))
        
        # Add confidence intervals to plot
        fig.add_trace(go.Scatter(
            x=high_conf_centers,
            y=[ci[0] for ci in confidence_intervals],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=high_conf_centers,
            y=[ci[1] for ci in confidence_intervals],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            name='95% Confidence Interval',
            showlegend=True
        ))
    
    fig.update_layout(
        title=f"{title} (with Isotonic Calibration)",
        xaxis_title="Predicted Probability",
        yaxis_title="Actual Rate",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# --- Layout ---
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("⚾ MLB 2025 — Advanced Elo Simulation Dashboard", 
                className="dash-header"),
        html.P("Comprehensive Elo-based MLB team analysis with simulation insights", 
               className="dash-subtitle"),
    ], className="dash-header"),
    
    # Navigation Tabs
    dcc.Tabs([
        # Tab 1: Current Season Overview
        dcc.Tab(label="🏆 Current Season", children=[
            html.Div([
                html.Div([
                    html.H2("Current Season Analysis", className="section-header"),
                ], className="section-header"),
                
                # Elo Ratings vs Simulated Wins
                html.Div([
                    html.H3("Elo Ratings vs Simulated Performance"),
                    dcc.Graph(
                        id="elo-vs-wins-scatter",
                        figure=px.scatter(
                            merged_df, 
                            x="Relative Elo", 
                            y="Wins", 
                            color="Division Name",
                            size="Elo Rating",
                            hover_data=["Team", "Losses", "Division Rank"],
                            title="Elo Ratings vs Simulated Wins",
                            labels={"Relative Elo": "Relative Elo Rating", "Wins": "Simulated Wins"},
                            color_discrete_sequence=px.colors.qualitative.Set1,
                        ),
                        style={"height": "500px", "marginBottom": "30px"}
                    ),
                ], className="graph-container"),
                
                # Side-by-side bar charts
                html.Div([
                    html.Div([
                        html.H3("Relative Elo Ratings"),
                        dcc.Graph(
                            id="elo-bar",
                            figure=px.bar(
                                merged_df,
                                x="Team",
                                y="Relative Elo",
                                color="Division Name",
                                title="Relative Elo Ratings by Team",
                                hover_data=["Elo Rating", "Division Rank", "Wins", "Losses"],
                                color_discrete_sequence=px.colors.qualitative.Set2,
                            ),
                            style={"height": "400px"}
                        ),
                    ], className="graph-container", style={"width": "50%", "display": "inline-block"}),
                    
                    html.Div([
                        html.H3("Simulated Wins"),
                        dcc.Graph(
                            id="wins-bar",
                            figure=px.bar(
                                merged_df,
                                x="Team",
                                y="Wins",
                                color="Division Name",
                                title="Simulated Wins by Team",
                                hover_data=["Elo Rating", "Losses", "Division Rank"],
                                color_discrete_sequence=px.colors.qualitative.Set3,
                            ),
                            style={"height": "400px"}
                        ),
                    ], className="graph-container", style={"width": "50%", "display": "inline-block"}),
                ], className="side-by-side"),
                
                # Division Standings Comparison
                html.Div([
                    html.H3("Simulated vs Actual Standings Comparison"),
                    dcc.Graph(
                        id="standings-comparison",
                        figure=go.Figure(data=[
                            go.Bar(name='Simulated Wins', x=merged_df['Team'], y=merged_df['Wins'], 
                                   marker_color='lightblue'),
                            go.Bar(name='Actual Wins', x=mlb_standings_df['Team'], y=mlb_standings_df['Wins'], 
                                   marker_color='orange')
                        ]).update_layout(
                            title="Simulated vs Actual Wins",
                            barmode='group',
                            xaxis_tickangle=-45,
                            height=500
                        ),
                        style={"marginBottom": "30px"}
                    ),
                ], className="comparison-chart"),
            ], className="dash-tab-content")
        ]),
        
        # Tab 2: Simulation Details
        dcc.Tab(label="🎲 Simulation Details", children=[
            html.Div([
                html.Div([
                    html.H2("Simulation Analysis", className="section-header"),
                ], className="section-header"),
                
                # Key Metrics Cards
                html.Div([
                    html.Div([
                        html.H4("Total Simulated Games"),
                        html.Div(f"{len(simulated_games_df):,}", className="metric-value"),
                    ], className="metric-card"),
                    html.Div([
                        html.H4("Home Win Rate"),
                        html.Div(f"{simulated_games_df['Home Win'].mean():.1%}", className="metric-value"),
                    ], className="metric-card"),
                    html.Div([
                        html.H4("Average Win Probability"),
                        html.Div(f"{synthetic_predictions_df['Team 1 Win Prob'].mean():.1%}", className="metric-value"),
                    ], className="metric-card"),
                    html.Div([
                        html.H4("Teams in Simulation"),
                        html.Div(f"{len(unique_teams)}", className="metric-value"),
                    ], className="metric-card"),
                ], className="metrics-grid"),
                
                # Game-by-game simulation
                html.Div([
                    html.H3("Simulated Game Outcomes"),
                    html.P(f"Total simulated games: {len(simulated_games_df)}", style={"marginBottom": "10px"}),
                    dcc.Graph(
                        id="game-simulation",
                        figure=px.scatter(
                            simulated_games_df.head(100),  # Show first 100 games
                            x="Game_Index",
                            y="Cumulative_Home_Wins",
                            color="Home Team",
                            title="Cumulative Home Team Wins (First 100 Games)",
                            labels={"Game_Index": "Game Number", "Cumulative_Home_Wins": "Cumulative Home Wins"},
                        ),
                        style={"height": "400px", "marginBottom": "30px"}
                    ),
                ], className="graph-container"),
                
                # Win probability distribution
                html.Div([
                    html.H3("Win Probability Distribution"),
                    dcc.Graph(
                        id="win-prob-dist",
                        figure=px.histogram(
                            synthetic_predictions_df,
                            x="Team 1 Win Prob",
                            nbins=30,
                            title="Distribution of Win Probabilities",
                            labels={"Team 1 Win Prob": "Win Probability", "count": "Number of Games"},
                        ),
                        style={"height": "400px", "marginBottom": "30px"}
                    ),
                ], className="graph-container"),
                
                # Team performance in simulations
                html.Div([
                    html.H3("Team Performance in Simulations"),
                    dcc.Graph(
                        id="team-sim-performance",
                        figure=px.bar(
                            simulated_games_df.groupby('Winner').size().reset_index(name='Wins'),
                            x='Winner',
                            y='Wins',
                            title="Total Wins by Team in Simulation",
                            labels={"Winner": "Team", "Wins": "Simulated Wins"},
                        ).update_layout(xaxis_tickangle=-45),
                        style={"height": "500px", "marginBottom": "30px"}
                    ),
                ], className="graph-container"),
            ], className="dash-tab-content")
        ]),
        
        # Tab 3: Historical Backtesting
        dcc.Tab(label="📊 Historical Backtesting", children=[
            html.Div([
                html.Div([
                    html.H2("Model Performance on 2024 Season", className="section-header"),
                ], className="section-header"),
                
                # Team selection for trajectories
                html.Div([
                    html.H3("Elo Rating Trajectories"),
                    html.P(f"Available teams: {len(unique_teams)}", style={"marginBottom": "10px"}),
                    dcc.Dropdown(
                        id="team-dropdown",
                        options=[{"label": t, "value": t} for t in unique_teams],
                        value=default_teams,
                        multi=True,
                        placeholder="Select teams to plot Elo trajectory",
                        style={"marginBottom": "20px"}
                    ),
                    dcc.Graph(
                        id="elo-trajectory",
                        style={"height": "500px", "marginBottom": "30px"}
                    ),
                ], className="graph-container"),
                
                # Model comparison
                html.Div([
                    html.H3("Classic vs Bayesian Elo Performance"),
                    html.Div([
                        html.Div([
                            html.H4("Classic Elo"),
                            dcc.Graph(
                                id="classic-calibration",
                                figure=px.scatter(
                                    pred_df, x="prob", y="actual",
                                    labels={"prob": "Predicted Probability", "actual": "Actual Outcome"},
                                    title="Classic Elo: Predicted vs Actual",
                                    opacity=0.5,
                                ) if not pred_df.empty else px.scatter(title="No data available"),
                                style={"height": "400px"}
                            ),
                        ], className="graph-container", style={"width": "50%", "display": "inline-block"}),
                        
                        html.Div([
                            html.H4("Bayesian Elo"),
                            dcc.Graph(
                                id="bayes-calibration",
                                figure=px.scatter(
                                    pred_bayes_df, x="prob", y="actual",
                                    labels={"prob": "Predicted Probability", "actual": "Actual Outcome"},
                                    title="Bayesian Elo: Predicted vs Actual",
                                    opacity=0.5,
                                ) if not pred_bayes_df.empty else px.scatter(title="No data available"),
                                style={"height": "400px"}
                            ),
                        ], className="graph-container", style={"width": "50%", "display": "inline-block"}),
                    ], className="side-by-side"),
                ]),
                
                # Performance metrics
                html.Div([
                    html.H3("Model Performance Metrics"),
                    html.Div([
                        html.Div([
                            html.H4("Classic Elo Metrics"),
                            html.P(f"Log Loss: {pred_df['prob'].apply(lambda x: -np.log(x) if x > 0 else 10).mean():.4f}" if not pred_df.empty else "No data"),
                            html.P(f"Accuracy: {((pred_df['prob'] > 0.5) == pred_df['actual']).mean():.1%}" if not pred_df.empty else "No data"),
                        ], className="performance-metric"),
                        html.Div([
                            html.H4("Bayesian Elo Metrics"),
                            html.P(f"Log Loss: {pred_bayes_df['prob'].apply(lambda x: -np.log(x) if x > 0 else 10).mean():.4f}" if not pred_bayes_df.empty else "No data"),
                            html.P(f"Accuracy: {((pred_bayes_df['prob'] > 0.5) == pred_bayes_df['actual']).mean():.1%}" if not pred_bayes_df.empty else "No data"),
                        ], className="performance-metric"),
                    ], className="performance-metrics"),
                ]),
                
                # Calibration improvements
                html.Div([
                    html.H3("Calibration Improvements"),
                    html.Div([
                        html.Div([
                            html.H4("High Confidence Region (>0.7)"),
                            html.P(f"Original Calibration Error: {calculate_calibration_error(pred_df, 0.7):.4f}" if not pred_df.empty else "No data"),
                            html.P(f"Calibrated Error: {calculate_calibrated_error(pred_df, 0.7):.4f}" if not pred_df.empty else "No data"),
                            html.P(f"Improvement: {calculate_improvement(pred_df, 0.7):.1%}" if not pred_df.empty else "No data"),
                        ], className="performance-metric"),
                        html.Div([
                            html.H4("Overall Calibration"),
                            html.P(f"Original Brier Score: {calculate_brier_score(pred_df):.4f}" if not pred_df.empty else "No data"),
                            html.P(f"Calibrated Brier Score: {calculate_calibrated_brier_score(pred_df):.4f}" if not pred_df.empty else "No data"),
                            html.P(f"Improvement: {calculate_brier_improvement(pred_df):.1%}" if not pred_df.empty else "No data"),
                        ], className="performance-metric"),
                    ], className="performance-metrics"),
                ]),
            ], className="dash-tab-content")
        ]),
        
        # Tab 4: Advanced Analytics
        dcc.Tab(label="🔬 Advanced Analytics", children=[
            html.Div([
                html.Div([
                    html.H2("Advanced Model Analysis", className="section-header"),
                ], className="section-header"),
                
                # Reliability diagrams
                html.Div([
                    html.H3("Model Calibration (Reliability Diagrams)"),
                    html.Div([
                        html.Div([
                            html.H4("Classic Elo Calibration"),
                            dcc.Graph(
                                id="classic-reliability",
                                style={"height": "400px"}
                            ),
                        ], className="graph-container", style={"width": "50%", "display": "inline-block"}),
                        
                        html.Div([
                            html.H4("Bayesian Elo Calibration"),
                            dcc.Graph(
                                id="bayes-reliability",
                                style={"height": "400px"}
                            ),
                        ], className="graph-container", style={"width": "50%", "display": "inline-block"}),
                    ], className="side-by-side"),
                ]),
                
                # Elo rating distribution
                html.Div([
                    html.H3("Elo Rating Distribution"),
                    dcc.Graph(
                        id="elo-distribution",
                        figure=px.histogram(
                            elo_df,
                            x="Elo Rating",
                            nbins=20,
                            title="Distribution of Final Elo Ratings",
                            labels={"Elo Rating": "Elo Rating", "count": "Number of Teams"},
                        ),
                        style={"height": "400px", "marginBottom": "30px"}
                    ),
                ], className="graph-container"),
                
                # Division analysis
                html.Div([
                    html.H3("Division Performance Analysis"),
                    dcc.Graph(
                        id="division-analysis",
                        figure=px.box(
                            merged_df,
                            x="Division Name",
                            y="Relative Elo",
                            title="Elo Rating Distribution by Division",
                            labels={"Division Name": "Division", "Relative Elo": "Relative Elo Rating"},
                        ),
                        style={"height": "400px", "marginBottom": "30px"}
                    ),
                ], className="graph-container"),
            ], className="dash-tab-content")
        ]),
    ], className="dash-tabs"),
    
], className="dash-container", style={"padding": "0", "maxWidth": "100%", "margin": "0"})

# --- Callbacks ---
@app.callback(
    Output('elo-trajectory', 'figure'),
    Input('team-dropdown', 'value')
)
def update_elo_trajectory(selected_teams):
    if not selected_teams or not elo_traj_df.empty:
        return px.scatter(title="No data available")
    
    filtered = elo_traj_df[elo_traj_df["team"].isin(selected_teams)]
    if filtered.empty:
        return px.scatter(title="No data available for selected teams")
    
    return px.line(filtered, x="date", y="elo", color="team", 
                   title="Elo Rating Trajectories (2024 Season)",
                   labels={"date": "Date", "elo": "Elo Rating", "team": "Team"})

@app.callback(
    Output('classic-reliability', 'figure'),
    Input('team-dropdown', 'value')  # Dummy input to trigger callback
)
def update_classic_reliability(selected_teams):
    return create_calibrated_reliability_figure(pred_df, "Classic Elo Reliability Diagram", 'blue')

@app.callback(
    Output('bayes-reliability', 'figure'),
    Input('team-dropdown', 'value')  # Dummy input to trigger callback
)
def update_bayes_reliability(selected_teams):
    return create_calibrated_reliability_figure(pred_bayes_df, "Bayesian Elo Reliability Diagram", 'green')

if __name__ == '__main__':
    print("Starting Dash app...")
    app.run(debug=True, port=8051)
    print("Dash is running on http://127.0.0.1:8051/")
