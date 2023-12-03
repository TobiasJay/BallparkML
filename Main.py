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

    '''
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
    '''
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


    # sets index to Date and Team so we can quickly reference stats for a given team on a given date
    df.set_index(['Date', 'Team'], inplace=True)

    # adding column for average of BA to date
    df['BA_AvgToDate'] = df.groupby('Team')['BA'].cumsum() / (df.groupby('Team').cumcount() + 1)
    df['PitcherERA_AvgToDate'] = df.groupby(['Team', 'Player-additional'])['IP'].cumsum() / (df.groupby(['Team', 'Player-additional']).cumcount() + 1)

    # adding column for average of BA over last 5 games
    column_to_average = 'BA'
    window_size = 5

    # Calculate the rolling mean over the last five games
    df['BA_AvgLast5'] = df.groupby('Team')[column_to_average].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())

    print(df.loc['2023-03-30', ...]) 



if __name__ == '__main__':
    main()