from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np


def calculate_batavg_average(data):
    """
    

    """
    if 'batavg' not in data or not data['batavg']:
        return 0  # it will return 0 if the batavg is not in the data.

    batavg_sum = sum(data['batavg'])
    batavg_count = len(data['batavg'])

    return batavg_sum / batavg_count


data = {
    "pitching": [3.5, 4.2, 3.8],
    "batavg": [0.250, 0.275, 0.260],
    "score": [5, 3, 4],
    "conceded": [2, 4, 1],
    "date": ['2023-05-01', '2023-05-02', '2023-05-03']
}

batavg_avg = calculate_batavg_average(data)
print("Average of batavg:", batavg_avg)

def predict_batavg_last_5_games(data, n_neighbors=10):
    """
    
    """
    if 'batavg' not in data or not data['batavg']:
        return []

    # Convert the data to a NumPy array for easier manipulation
    dataset = np.column_stack([data[feature] for feature in data if feature != 'date'])
    
    # Features (X) and target (y)
    X = dataset[:, :-1]  # All columns except 'batavg'
    y = dataset[:, -1]   # 'batavg' column

    # Splitting data: last 5 games for testing, the rest for training
    X_train, X_test = X[:-5], X[-5:]
    y_train = y[:-5]

    # Create and train KNN regressor
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Predict 'batavg' for the last 5 games
    predictions = knn.predict(X_test)

    return predictions

# Example usage with the provided data
data = {
    "pitching": [3.5, 4.2, 3.8, 4.1, 3.9, 4.0, 4.3, 4.4, 4.5, 4.6, 3.6, 4.1, 4.0, 3.7, 4.3, 4.2, 4.0, 3.9, 4.5, 4.7],
    "batavg": [0.250, 0.275, 0.260, 0.270, 0.280, 0.290, 0.300, 0.310, 0.320, 0.330, 0.240, 0.265, 0.285, 0.295, 0.305, 0.315, 0.325, 0.335, 0.345, 0.355],
    "score": [5, 3, 4, 6, 5, 7, 6, 8, 7, 9, 4, 2, 5, 7, 8, 6, 7, 5, 9, 10],
    "conceded": [2, 4, 1, 3, 2, 5, 4, 3, 6, 5, 3, 2, 4, 3, 2, 4, 3, 2, 4, 1],
    "date": ['2023-05-01', '2023-05-02', '2023-05-03', '2023-05-04', '2023-05-05', '2023-05-06', '2023-05-07', '2023-05-08', '2023-05-09', '2023-05-10', 
             '2023-05-11', '2023-05-12', '2023-05-13', '2023-05-14', '2023-05-15', '2023-05-16', '2023-05-17', '2023-05-18', '2023-05-19', '2023-05-20']
}


predicted_batavg = predict_batavg_last_5_games(data)
print("Predicted 'batavg' for the last 5 games:", predicted_batavg)
formatted_predictions = ["batavg={:.3f}".format(value) for value in predicted_batavg]

# Display the formatted predictions
for prediction in formatted_predictions:
    print(prediction)