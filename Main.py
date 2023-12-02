import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def main():
    # Read the CSV file into a pandas DataFrame
    p_df = pd.read_csv('data/pitcherstats.csv')
    b_df = pd.read_csv('data/batterstats.csv')

    # Extract the required columns
    pitcher_columns = ['Date', 'Player-additional', 'IP', 'R', 'Team', 'Opp', 'Result'] # add more features later
    selected_pdata = p_df[pitcher_columns]
    batter_columns = ['Date', 'BA', 'Team', 'Opp'] # add more features later (HR, RBI, etc.)
    selected_bdata = b_df[batter_columns]

    # Merge Datasets

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

    # Train Model on 20 games at the end of the season, so 
    # Split into Training and test data (Train on first 142 games, test on last 20 games)
    training_games = df.head(142)
    test_games = df.tail(20)

    # Create a dictionary with tuples (Team, Date) as keys and corresponding rows as values for train and test data
    TrainDict = {}
    for index, row in training_games.iterrows():
        key = (row['Team'], row['Date'])
        TrainDict[key] = row

    TestDict = {}
    for index, row in test_games.iterrows():
        key = (row['Team'], row['Date'])
        TestDict[key] = row


    # 2. Train model on training data


    # Example: Accessing data for ('SEA', '2023-03-30')
    example_key = ('SEA', '2023-05-03')
    print(DataDict.get(example_key))
    print(DataDict.get(example_key))

if __name__ == '__main__':
    main()