import pandas as pd
import numpy as np
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



def main():
    # Read the CSV file into a pandas DataFrame
    p_df = pd.read_csv('data/2023_pstats.csv')
    b_df = pd.read_csv('data/2023_bstats.csv')

    # Extract the required columns
    # Opp_R = Opponent Runs while pitcher was in the game
    pitcher_columns = ['Date', 'Player-additional', 'IP', 'Opp_R', 'ER', 'Team', 'Opp', 'Result', 'Unnamed: 5'] # add more features later
    selected_pdata = p_df[pitcher_columns]
    batter_columns = ['Date', 'BA', 'Team', 'Opp', 'Result', 'Unnamed: 3', 'OPS'] # add more features later (HR, RBI, etc.)
    selected_bdata = b_df[batter_columns]

    # Create seasonal avg BA column
    # Create Opp Pitcher ERA column
    # Create Score column

    # Create seasonal avg BA column
    # Doesn't include current BA value in the average
    X1 = pd.DataFrame(selected_bdata['Date'])
    X1['Opp'] = selected_bdata['Opp']
    X1['HorA'] = selected_bdata['Unnamed: 3']
    
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
    X2['Team'] = selected_pdata['Team']
    X2['Opp'] = selected_pdata['Opp']

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
    X2.drop(['IP_Last5', 'ER_Last5'], axis=1, inplace=True)

    # Using a trick here to link the opposing pitchers and the batters together
    X1.drop_duplicates(subset=['Date', 'Team'], keep='first', inplace=True)
    # Merge pitchers and batters together into one dataframe
    X1['Result'] = selected_bdata['Result']
    matches = pd.merge(X1, X2, how='inner', on=['Date', 'Team','Opp'])
    # Assuming 'matches' is your DataFrame
    # Create an empty DataFrame to store the pairs of games
    game_pairs = pd.DataFrame()

    # Iterate through unique dates in the original DataFrame
    for date in matches['Date'].unique():
        # Subset the DataFrame for the current date
        subset_df = matches[matches['Date'] == date]
        # Iterate through rows to isolate pairs of games
        for index, row in subset_df.iterrows():
            # Find the corresponding row with the opposing team and only keep @ teams

            opposing_row = subset_df[(subset_df['Team'] == row['Opp']) & (row['HorA'] == '@')]
            # Check if an opposing row is found
            if not opposing_row.empty:
                # Away = row, Home = opposing_row
                combined_row = {
                    'Date': date,
                    'AwayTeam': row['Team'],
                    'HomeTeam': opposing_row['Team'].values[0],
                    'HomeResult': opposing_row['Result'].values[0],
                    'H_BA_AvgToDate': opposing_row['BA_AvgToDate'].values[0],
                    'A_BA_AvgToDate': row['BA_AvgToDate'],
                    'H_BA_AvgLast5': opposing_row['BA_AvgLast5'].values[0],
                    'A_BA_AvgLast5': row['BA_AvgLast5'],
                    'H_ERA_cum': opposing_row['ERA_cum'].values[0],
                    'A_ERA_cum': row['ERA_cum'],
                    'H_ERA_Last5': opposing_row['ERA_Last5'].values[0],
                    'A_ERA_Last5': row['ERA_Last5'],
                    # Add other columns as needed
                }                
                # Append the combined row to the DataFrame
                game_pairs = game_pairs._append(combined_row, ignore_index=True)
                # Save BA 
                # The pair of games is isolated here
                # 'row' contains one team, 'opposing_row' contains the other
                # Add your code here to process the pair of games if needed
                # ...
                
    # Extract win/loss and scores
    game_pairs[['Outcome', 'Scores']] = game_pairs['HomeResult'].str.extract(r'([WL]) (\d+-\d+)')

    # Create binary column for win (1) and loss (0)
    game_pairs['H_Win'] = (game_pairs['Outcome'] == 'W').astype(int)

    # Split the Scores column into two separate columns
    game_pairs[['H_Score', 'A_Score']] = game_pairs['Scores'].str.split('-', expand=True).astype(int)
    
    
    # Drop unnecessary columns
    game_pairs.drop(['HomeResult', 'Outcome', 'Scores'], axis=1, inplace=True)

    # Drop rows with NaN values
    game_pairs.dropna(inplace=True)
    # drop target from X and save to y
    y = game_pairs['H_Win']
    X = game_pairs.drop(['H_Score', 'A_Score', 'Date', 'AwayTeam','HomeTeam','H_Win'], axis=1)

    
    # Create Training and test sets
    # Line 4016 in dataset marks the start of september, the last month of the season
    # Originally 4016 was the start of september, but we removed rows with NaN values, so the split is now at 3611
    # 1693 is now our magic number where september starts and also 81.3% of the data
    X_train = X.head(1693)
    X_test = X.tail(len(X) - 1693)
    y_train = y.head(1693)
    y_test = y.tail(len(y) - 1693)
    # Then we need to split into features and target (# of runs scored)
    adaboost = AdaBoostClassifier(n_estimators=10)

    # 3. Train the AdaBoost model
    adaboost.fit(X_train, y_train)

    # 4. Predict on the test set
    y_pred_adaboost = adaboost.predict(X_test)

    # 5. Evaluate the model's performance
    accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
    print("Testing accuracy using AdaBoost:", accuracy_adaboost)

    # Evaluate the training accuracy
    y_pred_adaboost_train = adaboost.predict(X_train)
    accuracy_adaboost_train = accuracy_score(y_train, y_pred_adaboost_train)
    print("Training accuracy using AdaBoost:", accuracy_adaboost_train)
    

    # Create SVM model
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("Testing accuracy using SVM:", accuracy_svm)
    training_accuracy_svm = accuracy_score(y_train, svm.predict(X_train))
    print("Training accuracy using SVM:", training_accuracy_svm)


if __name__ == '__main__':
    main()