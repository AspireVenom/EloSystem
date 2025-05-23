import requests
import csv

url = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2024&standingsTypes=regularSeason&hydrate=team"

response = requests.get(url)
data = response.json()

standings_data = []

for record in data['records']:
    for team_record in record['teamRecords']:
        try:
            team_name = team_record['team']['name']
            wins = team_record['wins']
            losses = team_record['losses']
            division = team_record['divisionRank']  # Use division rank if division name is missing
            standings_data.append([team_name, wins, losses, division])
        except KeyError:
            continue

# Save to CSV
with open("mlb_standings.csv", mode="w", newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Team", "Wins", "Losses", "Division Rank"])
    writer.writerows(standings_data)

print("✅ Data saved to mlb_standings.csv")
