import pandas as pd
import numpy as np
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def main():
    # Read the CSV file into a pandas DataFrame
    p_df = pd.read_csv('data/pitcherstats.csv')
    b_df = pd.read_csv('data/batterstats.csv')

    # Extract the required columns
    # Opp_R = Opponent Runs while pitcher was in the game
    pitcher_columns = ['Date', 'Player-additional', 'IP', 'Opp_R', 'ER', 'Team', 'Opp', 'Result'] # add more features later
    selected_pdata = p_df[pitcher_columns]
    batter_columns = ['Date', 'BA', 'Team', 'Opp', 'Result'] # add more features later (HR, RBI, etc.)
    selected_bdata = b_df[batter_columns]


    # Create seasonal avg BA column
    # Create Opp Pitcher ERA column
    # Create Score column

    # Create seasonal avg BA column
    # Doesn't include current BA value in the average
    X1 = pd.DataFrame(selected_bdata['Date'])
    X1['BA_AvgToDate'] = (selected_bdata.groupby('Team')['BA'].cumsum() - selected_bdata['BA']) / (selected_bdata.groupby('Team').cumcount())
    # adding column for average of BA over last 5 games
    window_size = 5

    
    # Calculate the rolling mean over the last five games (Change window function in x.rolling(window= ... )) Check documentation
    # Find x.rolling documentation at this URL: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
    # closed='left' means that the window will not include the current game
    X1['BA_AvgLast5'] = selected_bdata.groupby('Team')['BA'].transform(lambda x: x.rolling(window=window_size, min_periods=1,closed='left').mean())
    X1['Team'] = selected_bdata['Team']
    # Map IP to 0.0, 0.3333, 0.6667 for the ERA math to work (original values have 4.1 represent 4 innings and 1 out)
    selected_pdata['IP'] = selected_pdata['IP'].apply(lambda x: int(x) + 1/3 if round(x % 1, 1) == 0.1 else int(x) + 2/3 if round(x % 1, 1) == 0.2 else x)

    X2 = pd.DataFrame(selected_pdata['Date'])

    # ERA is today's ER / today's IP, IP total is all IP seasonally, ERA_cum is ERA up to, but not including today
    X2['IP TOTAL'] = selected_pdata.groupby(['Player-additional'])['IP'].cumsum()
    X2['ERA'] = 9 * selected_pdata['ER'] / selected_pdata['IP']
    X2['ERA_cum'] = 9 * (selected_pdata.groupby(['Player-additional'])['ER'].cumsum() - selected_pdata['ER']) / (selected_pdata.groupby(['Player-additional'])['IP'].cumsum() - selected_pdata['IP'])
    X2['Player-additional'] = selected_pdata['Player-additional']


    # Calculate the rolling mean over the last five games
    # does not include today's ERA in calcuation
    X2['IP_Last5'] = selected_pdata.groupby(['Player-additional'])['IP'].transform(lambda x: x.rolling(window=window_size, min_periods=1,closed='left').sum())
    X2['ER_Last5'] = selected_pdata.groupby(['Player-additional'])['ER'].transform(lambda x: x.rolling(window=window_size, min_periods=1,closed='left').sum())
    X2['ERA_Last5'] = 9 * X2['ER_Last5'] / X2['IP_Last5']
    # Using a trick here to link the opposing pitchers and the batters together
    X2['Team'] = selected_pdata['Opp']
    print(X2.head(200))
    # Batting and pitching datasets are not lined up, so we need to merge them, but merging wasn't working earlier

    # Issue: there are double headers and so sometimes there are multiple games on the same day. Who pitches which game?
    


    X1.drop_duplicates(subset=['Date', 'Team'], keep='first', inplace=True)
    print(len(selected_bdata))
    matches = pd.merge(X1, X2, how='inner', on=['Date', 'Team'])
    matches.drop(['IP_Last5', 'ER_Last5'], axis=1, inplace=True)
    # check for duplicate rows in pd.merge
    # Team and result will match but pitcher will be from opposing team
    matches['Result'] = selected_bdata['Result']
    # Extract win/loss and scores
    matches[['Outcome', 'Scores']] = matches['Result'].str.extract(r'([WL]) (\d+-\d+)')

    # Create binary column for win (1) and loss (0)
    matches['Win'] = (matches['Outcome'] == 'W').astype(int)

    # Split the Scores column into two separate columns
    matches[['Score', 'Opp_Score']] = matches['Scores'].str.split('-', expand=True).astype(int)

    # Drop unnecessary columns
    matches = matches.drop(['Result', 'Outcome', 'Scores', 'Opp_Score'], axis=1)
    print(matches.head(60))

    '''
    # Line 4016 in dataset marks the start of september, the last month of the season
    training_games = df.head(4016)
    test_games = df.tail(len(df) - 4016)
    # Then we need to split into features and target (# of runs scored)
    X = df.drop(['Win', 'Score', 'Opp_R', 'Opp_Score','Player-additional'], axis=1)
    y = df['Score']
    '''


    '''
    # Merge Datasets
    # Currently bugged
    df = pd.merge(selected_bdata, selected_pdata, how='inner', on=['Date', 'Team', 'Opp'])

    # Chat GPT code for converting result to number feature "W 2 - 7" => Win column, Score column, Opp_Score column

    # Extract win/loss and scores
    df[['Outcome', 'Scores']] = df['Result'].str.extract(r'([WL]) (\d+-\d+)')

    # Create binary column for win (1) and loss (0)
    df['Win'] = (df['Outcome'] == 'W').astype(int)

    # Split the Scores column into two separate columns
    df[['Score', 'Opp_Score']] = df['Scores'].str.split('-', expand=True).astype(int)

    # Drop unnecessary columns
    df = df.drop(['Result', 'Outcome', 'Scores'], axis=1)

    
    # Line 4016 in dataset marks the start of september, the last month of the season
    training_games = df.head(4016)
    test_games = df.tail(len(df) - 4016)
    # Then we need to split into features and target (# of runs scored)
    X = df.drop(['Win', 'Score', 'Opp_R', 'Opp_Score','Player-additional'], axis=1)
    y = df['Score']


    # Split into training and test data
    
    X_train = X.head(4016)
    X_test = X.tail(len(X) - 4016)  
    y_train = y.head(4016)
    y_test = y.tail(len(y) - 4016)
    print(X_train)
    
    # X_train contains Date, BA, Team, Opp, IP, ER
    
    # Approach 1. Avg all season stats for each team
    # 1.1 Create dataset containing average stats to this point, with W - L as the target
    # 1.2 Repeat for each day of the season to create the rows of the dataset up to 80% of the season
    # 1.3 Train model on that dataset
    # 1.4 Test model on the remaining 20% of the season, following same averaging procedure.
    # 1.5 Improve model by using average of last five games or test on other seasons for more confidence.

    # 1.1 Create dataset containing average stats to this point, with W - L as the target
    # Use Team BA, Opp Pitcher ERA (use team ERA if this is pitcher's first game) and skip first game of season


    # Approach 2. Modify Approach 1 to evaluate only the last 5 games for each team

    stat_df = df.copy()
    # sets index to Date and Team so we can quickly reference stats for a given team on a given date



    # Map IP to 0.0, 0.3333, 0.6667 for the ERA math to work (original values have 4.1 represent 4 innings and 1 out)
    stat_df['IP'] = stat_df['IP'].apply(lambda x: int(x) + 1/3 if round(x % 1, 1) == 0.1 else int(x) + 2/3 if round(x % 1, 1) == 0.2 else x)
    stat_df['IP_Total'] = stat_df.groupby('Player-additional')['IP'].cumsum()
    #filtered_df = stat_df[stat_df['Player-additional'] == 'weavelu01']
    #print(filtered_df)

    # adding column for average of BA to date
    stat_df['BA_AvgToDate'] = stat_df.groupby('Team')['BA'].cumsum() / (stat_df.groupby('Team').cumcount() + 1)
    # adding column for average of ERA to date
    stat_df['IP_Total'] = stat_df.groupby(['Team', 'Player-additional'])['IP'].cumsum()
    stat_df['ERA_toDate'] = 9 * stat_df.groupby(['Team', 'Player-additional'])['ER'].cumsum() / stat_df.groupby(['Team', 'Player-additional'])['IP'].cumsum()

    #stat_df['PitcherERA_AvgToDate'] = stat_df.groupby(['Team', 'Player-additional'])['ER'].cumsum() / (stat_df.groupby(['Team', 'Player-additional']).cumcount() + 1)
    #stat_df['PitcherERA_AvgToDate'] = stat_df['PitcherERA_AvgToDate'] * 9 / stat_df['IP']


    # adding column for average of BA over last 5 games
    column_to_average = 'BA'
    window_size = 5

    # Calculate the rolling mean over the last five games
    stat_df['BA_AvgLast5'] = stat_df.groupby('Team')[column_to_average].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())

    # adding column for average of ERA over last 5 games
    stat_df['ERA_Last5'] = stat_df.groupby(['Team', 'Player-additional'])['ER'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    stat_df['ERA_Last5'] = stat_df['ERA_Last5'] * 9 / stat_df['IP']

    # There are 383 unique starting pitchers in the dataset. Onehot encoding them would add that many features
    # print(stat_df['Player-additional'].nunique()) 
    '''



if __name__ == '__main__':
    main()