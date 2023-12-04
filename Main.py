import pandas as pd
import numpy as np
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

'''
class Data:

    def __init__(self):
        """
        Data class.

        Attributes
        --------------------
            X -- numpy array of shape (n,d), features
            y -- numpy array of shape (n,), targets
        """

        # n = number of examples, d = dimensionality
        self.X = None
        self.y = None

        self.Xnames = None
        self.yname = None

    def load(self, filename, header=0, predict_col=-1):
        """Load csv file into X array of features and y array of labels."""

        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, '..', 'data', filename)

        # load data
        with open(f, 'r') as fid:
            data = np.loadtxt(fid, delimiter=",", skiprows=header)

        # separate features and labels
        if predict_col is None:
            self.X = data[:,:]
            self.y = None
        else:
            if data.ndim > 1:
                self.X = np.delete(data, predict_col, axis=1)
                self.y = data[:,predict_col]
            else:
                self.X = None
                self.y = data[:]

        # load feature and label names
        if header != 0:
            with open(f, 'r') as fid:
                header = fid.readline().rstrip().split(",")

            if predict_col is None:
                self.Xnames = header[:]
                self.yname = None
            else:
                if len(header) > 1:
                    self.Xnames = np.delete(header, predict_col)
                    self.yname = header[predict_col]
                else:
                    self.Xnames = None
                    self.yname = header[0]
        else:
            self.Xnames = None
            self.yname = None

def load_data(filename, header=0, predict_col=-1):
    """Load csv file into Data class."""
    data = Data()
    data.load(filename, header=header, predict_col=predict_col)
    return data
'''

def main():
    # Read the CSV file into a pandas DataFrame
    p_df = pd.read_csv('data/pitcherstats.csv')
    b_df = pd.read_csv('data/batterstats.csv')

    # Extract the required columns
    # Opp_R = Opponent Runs while pitcher was in the game
    pitcher_columns = ['Date', 'Player-additional', 'IP', 'Opp_R', 'ER', 'Team', 'Opp', 'Result'] # add more features later
    selected_pdata = p_df[pitcher_columns]
    batter_columns = ['Date', 'BA', 'Team', 'Opp'] # add more features later (HR, RBI, etc.)
    selected_bdata = b_df[batter_columns]


    # Create seasonal avg BA column
    # Create Opp Pitcher ERA column
    # Create Score column

    # Create seasonal avg BA column
    X = pd.DataFrame(selected_bdata.groupby('Team')['BA'].cumsum() / (selected_bdata.groupby('Team').cumcount() + 1), columns=['BA_AvgToDate'])
    
    # Create Opp Pitcher ERA column

    # Map IP to 0.0, 0.3333, 0.6667 for the ERA math to work (original values have 4.1 represent 4 innings and 1 out)
    selected_pdata['IP'] = selected_pdata['IP'].apply(lambda x: int(x) + 1/3 if round(x % 1, 1) == 0.1 else int(x) + 2/3 if round(x % 1, 1) == 0.2 else x)

    X['IP TOTAL'] = selected_pdata.groupby(['Player-additional'])['IP'].cumsum()
    X['ERA'] = 9 * selected_pdata['ER'] / selected_pdata['IP']
    X['ERA_cum'] = 9 * selected_pdata.groupby(['Player-additional'])['ER'].cumsum() / selected_pdata.groupby(['Player-additional'])['IP'].cumsum()
    X['Player-additional'] = selected_pdata['Player-additional']
    # adding column for average of BA over last 5 games
    column_to_average = 'BA'
    window_size = 5

    # Calculate the rolling mean over the last five games
    X['BA_AvgLast5'] = selected_bdata.groupby('Team')[column_to_average].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    X['IP_Last5'] = selected_pdata.groupby(['Player-additional'])['IP'].transform(lambda x: x.rolling(window=window_size, min_periods=1).sum())
    X['ER_Last5'] = selected_pdata.groupby(['Player-additional'])['ER'].transform(lambda x: x.rolling(window=window_size, min_periods=1).sum())
    X['ERA_Last5'] = 9 * X['ER_Last5'] / X['IP_Last5']
    #X['ERA_Last5'] = selected_pdata.groupby(['Player-additional'])['ERA'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    print(X.tail(60))
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