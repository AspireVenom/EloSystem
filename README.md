# EloSystem

<p>A ML algorithm which uses real data (from MLB API) to encapsulate all <br>
match results [CURRENTLY 2024 SEASON].
* writes the data acquired from the api into a csv **mlb_game_results** then iterates through the games <br>
and with pytorch learns the outcomes and correclty calculates the teams elo rating based on the season <br>
based on the real games, I created a function <em>generate_synthetic_matches</em> which will create <br>
<em>synthetic matches</em> and the elo calculated to determine the outcome of the matches, subsequently <br>
also training on the data of the synthetic matches for elo outputting the file **final_elo_ratings.csv** </p>

## Issues

<p> Currently the outputted elo <em>lineup</em> is very realistic to the mlb generated elo, however, <br>
you can see that the elo outputted is insignificant in the **final_elo_ratings.csv**. This is likely to <br>
an issue with the way I am calculating the actual elo based on matches and the training having insignificantg <br>
noise. </p>

