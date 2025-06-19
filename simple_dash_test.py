import dash
import pandas as pd
import plotly.express as px
from dash import dcc, html
from dash.dependencies import Input, Output

# Load data
elo_traj_df = pd.read_csv("elo_history.csv")
pred_df = pd.read_csv("pred_vs_actual.csv")

print(f"Data loaded: {len(elo_traj_df)} elo records, {len(pred_df)} predictions")

# Create simple app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Simple Backtest Test"),
    
    html.H2("Elo Trajectories"),
    dcc.Dropdown(
        id='team-dropdown',
        options=[{'label': team, 'value': team} for team in sorted(elo_traj_df['team'].unique())],
        value=['New York Yankees'],
        multi=True
    ),
    dcc.Graph(id='elo-graph'),
    
    html.H2("Calibration Plot"),
    dcc.Graph(
        id='calibration-graph',
        figure=px.scatter(pred_df, x='prob', y='actual', title='Predicted vs Actual')
    )
])

@app.callback(
    Output('elo-graph', 'figure'),
    Input('team-dropdown', 'value')
)
def update_elo_graph(selected_teams):
    if not selected_teams:
        return px.scatter(title="No teams selected")
    
    filtered = elo_traj_df[elo_traj_df['team'].isin(selected_teams)]
    return px.line(filtered, x='date', y='elo', color='team', title='Elo Trajectories')

if __name__ == '__main__':
    print("Starting Dash app...")
    app.run(debug=True, port=8051) 