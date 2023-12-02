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

    FullData = pd.merge(selected_bdata, selected_pdata, how='inner', on=['Date', 'Team', 'Opp'])

    # Chat GPT code for converting result to number feature "W 2 - 7" => Win column, Score column, Opp_Score column

    # Extract win/loss and scores
    df[['Outcome', 'Scores']] = df['Result'].str.extract(r'([WL]) (\d+-\d+)')

    # Create binary column for win (1) and loss (0)
    df['Win'] = (df['Outcome'] == 'W').astype(int)

    # Split the Scores column into two separate columns
    df[['Score', 'Opp_Score']] = df['Scores'].str.split('-', expand=True).astype(int)

    # Drop unnecessary columns
    df = df.drop(['Result', 'Outcome', 'Scores'], axis=1)

    # Display the modified dataframe
    print(df)



    # Create a dictionary with tuples (Team, Date) as keys and corresponding rows as values
    DataIndex = {}
    for index, row in FullData.iterrows():
        key = (row['Team'], row['Date'])
        DataIndex[key] = row



    df = pd.DataFrame()
    # Train Model on 20 games at the end of the season

    # 1. Seperate data into training and testing data: First 142 games are training, last 20 are testing
    training_games = FullData.head(142)
    test_games = FullData.tail(20)

    # We want to train on features and predict the result. However right now our result is a string, so we need to convert it to a number
    # 0 = Loss, 1 = Win or design a differntial metric (aka team 1 scored 3 more runs than team 2)
    # 1.5 Convert result to number




    # 2. Train model on training data


    # Example: Accessing data for ('SEA', '2023-03-30')
    example_key = ('SEA', '2023-05-03')
    print(FullData.get(example_key))
    print(FullData.get(example_key))

if __name__ == '__main__':
    main()