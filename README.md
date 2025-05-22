# EloSystem

An elo calculator for MLB teams using pytorch and epochs to train on potential matches (that have not occurred [synthetic matches]), 
and use the formula:
R_new = R_old + K * (S - E) 

Which then produces an output of their elo rating based on the synthetic matches.

And calls to the MLB API for updated stats and records.