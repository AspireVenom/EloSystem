import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output

# Load your data
elo_df = pd.read_csv("final_elo_ratings.csv")
standings_df = pd.read_csv("mlb_standings.csv")
merged_df = pd.merge(elo_df, standings_df, on="Team")

# Add jitter to reduce overlap
np.random.seed(42)  # Reproducibility
merged_df["Wins_Jittered"] = merged_df["Wins"] + np.random.normal(
    0, 0.3, size=len(merged_df)
)

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Needed for deployment

app.layout = html.Div(
    [
        dcc.Graph(id="elo-scatter", className="full-graph"),  # ← add comma here
        dcc.Dropdown(
            id="team-selector",
            className="floating-dropdown",
            options=[{"label": team, "value": team} for team in merged_df["Team"]],
            placeholder="Select a team",
        ),
    ]
)


# Callback to update scatter plot
@app.callback(Output("elo-scatter", "figure"), Input("team-selector", "value"))
def update_figure(selected_team):
    fig = px.scatter(
        merged_df,
        x="Wins_Jittered",
        y="Elo Rating",
        text="Team",
        hover_data=["Wins", "Losses", "Division Rank"],
        title="Elo Rating vs Wins — Over/Underrated Teams",
    )

    fig.update_traces(
        textposition="top right",
        textfont=dict(size=10),
        texttemplate="%{text}",  # default
    )
    if selected_team:
        selected = merged_df[merged_df["Team"] == selected_team]
        fig.add_scatter(
            x=selected["Wins_Jittered"],
            y=selected["Elo Rating"],
            mode="markers+text",
            text=selected["Team"],
            marker=dict(color="red", size=16),
            name="Selected Team",
        )

    fig.update_layout(
        xaxis_title="Wins (2025)", yaxis_title="Elo Rating", template="plotly_white"
    )

    return fig


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
