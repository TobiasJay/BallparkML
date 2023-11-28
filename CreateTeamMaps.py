import pandas as pd

# Read the CSV file into a pandas DataFrame
p_df = pd.read_csv('data/pitcherstats.csv')
b_df = pd.read_csv('data/batterstats.csv')

# Extract the required columns
pitcher_columns = ['Date', 'Player-additional', 'IP', 'R', 'Team', 'Opp']
selected_pdata = p_df[pitcher_columns]
batter_columns = ['Date', 'BA', 'Team', 'Opp']
selected_bdata = b_df[batter_columns]

# Create a dictionary with tuples (Team, Date) as keys and corresponding rows as values
PitchingIndex = {}
for index, row in selected_pdata.iterrows():
    key = (row['Team'], row['Date'])
    PitchingIndex[key] = row

BattingIndex = {}
for index, row in selected_bdata.iterrows():
    key = (row['Team'], row['Date'])
    BattingIndex[key] = row

        
# Example: Accessing data for ('SEA', '2023-03-30')
example_key = ('SEA', '2023-05-03')
print(PitchingIndex.get(example_key))
print(BattingIndex.get(example_key))

