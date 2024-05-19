import pandas as pd
import Preprocess
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import matplotlib.pyplot as plt


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def main():
    season = '2023'

    # Read the CSV file into a pandas DataFrame
    p_df = pd.read_csv('data/' + season + '_pstats.csv')
    b_df = pd.read_csv('data/' + season + '_bstats.csv')

    # ======================= Preprocess Data =======================
    X, y = Preprocess.preprocess(p_df, b_df)

    # Create Training and test sets
    # Line 4016 in dataset marks the start of september, the last month of the season
    # Originally 4016 was the start of september, but we removed rows with NaN values, so the split is now at 3611
    # 1693 is now our magic number where september starts and also 81.3% of the data after condensing the games pairs into single rows
    #X_train, X_test = Preprocess.split(X, 1693)
    #y_train, y_test = Preprocess.split(y, 1693)

    # truc split takes data, train_size, test_size and returns train, test, and remaining data
    # for every 100 games train the model on 90 games and test on 10 games

    # Batch Training Approach
    overall_accuracy = 0.0
    overall_f1 = 0.0
    #overall_recall = 0.0
    f1_rolling_avg = []
    accuracy_rolling_avg = []
    #recall_rolling_avg = []
    num_batches = 0
    batch_size = 1
    for i in range(batch_size + 1, len(X), batch_size):
        X_train, X_test = Preprocess.trunc_split(X, i - 1, 1)
        y_train, y_test = Preprocess.trunc_split(y, i - 1, 1)

        # Then we need to split into features and target (# of runs scored)
        adaboost = AdaBoostClassifier(n_estimators=10)

        # 3. Train the AdaBoost model
        adaboost.fit(X_train, y_train)

        # 4. Predict on the test set
        y_pred_adaboost = adaboost.predict(X_test)

        # 5. Evaluate the model's performance
        accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
        

        f1 = f1_score(y_test, y_pred_adaboost, average='weighted')
        #recall = recall_score(y_test, y_pred_adaboost)

        overall_f1 += f1
        overall_accuracy += accuracy_adaboost
        #overall_recall += recall
        num_batches += 1
        f1_rolling_avg.append(overall_f1 / num_batches)
        accuracy_rolling_avg.append(overall_accuracy / num_batches)

        # Train the model on the training set
        # Test the model on the test set
        # Evaluate the model's performance
        # Repeat for the next batch

    print("Overall F1 using AdaBoost:", overall_accuracy / num_batches)
    
    plt.plot(range(num_batches), f1_rolling_avg, label='F1 Score Rolling Average')
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.xlabel('Data Points Used for training')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Over Batches')
    plt.legend()
    plt.show()

    '''
    # ======================= Train Models =======================

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
    f1 = f1_score(y_test, y_pred_adaboost, average='weighted')
    print("Training accuracy using AdaBoost:", accuracy_adaboost_train)
    print("F1 score using AdaBoost:", f1)
    

    # SVM predictions are not relavent. It will predict all win or all loss for the entire test set

    # Create SVM model
    #svm = SVC(kernel='linear')
    #svm.fit(X_train, y_train)
    #y_pred_svm = svm.predict(X_test)
    #accuracy_svm = accuracy_score(y_test, y_pred_svm)
    #print("Testing accuracy using SVM:", accuracy_svm)
    #training_accuracy_svm = accuracy_score(y_train, svm.predict(X_train))
    #print("Training accuracy using SVM:", training_accuracy_svm)


    # Confusion matrix for AdaBoost model
    cm_adaboost = confusion_matrix(y_test, y_pred_adaboost)
    print("Confusion matrix for AdaBoost model:")
    print(cm_adaboost)

    # Confusion matrix for SVM model
    #cm_svm = confusion_matrix(y_test, y_pred_svm)
    #print("Confusion matrix for SVM model:")
    #print(cm_svm)
    '''


if __name__ == '__main__':
    main()
