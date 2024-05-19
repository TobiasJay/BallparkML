import pandas as pd
import numpy as np

# This version of preprocess will return columns for the home and away team elo, as well as the result of the game.
# X contains the elo columns and y contains the result of the game.
def preprocess(p_df, b_df):
    # Im not really sure how to start out so I will make each team have a starting elo of 500 and you gain or lose 10 points in an equal matchup.
    # The basic math:
    # On a win, ELO gained = 10 + (OpposingELO - TeamELO / 30)
    # On a loss, ELO lost = 10 - (OpposingELO - TeamELO / 30)
    # The principle is that teams are rewarded more for beating stronger teams and penalized more for losing to weaker teams. And vice versa.
    # Here, 10 is a constant that can be adjusted and is just arbitrary for now.
    # Also, the 30 is the constant that decides how much to penalize stronger teams and reward weaker teams.
    # A larger constant corresponds with a decreased importance on the difference of elo.
    # A smaller constant will conversely make elo gained more dependent on the difference of elo.
    # A team that is 3 games ahead of another (or 30 elo points) will gain 1 less elo point for a win and lose 1 extra for a loss.
    
    
    return X, y