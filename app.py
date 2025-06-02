import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html

# Load CSVs
elo_df = pd.read_csv("final_elo_ratings.csv")  # Columns: Team, Elo Rating
standings_df = pd.read_csv(
    "simulated_standings.csv"
)  # Team, Wins, Losses, Division Rank, Division Name

# Add relative Elo and round
mean_elo = elo_df["Elo Rating"].mean()
elo_df["Relative Elo"] = elo_df["Elo Rating"] - mean_elo

# Merge on Team
merged_df = pd.merge(elo_df, standings_df, on="Team")

# Sort by Relative Elo
merged_df = merged_df.sort_values(by="Relative Elo", ascending=False)

# Create Dash App
app = dash.Dash(__name__)
app.title = "MLB Elo & Simulated Standings"

app.layout = html.Div(
    [
        html.H1(
            "MLB 2025 — Elo Ratings vs Simulated Wins", style={"textAlign": "center"}
        ),
        dcc.Graph(
            id="elo-bar",
            figure=px.bar(
                merged_df,
                x="Team",
                y="Relative Elo",
                color="Division Name",
                title="Relative Elo Ratings by Team (Centered at 1500)",
                hover_data=["Elo Rating", "Division Rank", "Wins", "Losses"],
                color_discrete_sequence=px.colors.qualitative.Set2,
            ),
        ),
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
        ),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
