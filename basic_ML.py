import pandas as pd 
import numpy as np 
import os 
import argparse

import mlflow
import mlflow.sklearn
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


# creating a function to get the data 

def get_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    # print(url)

    try:
    # read the data from the csv 
        df = pd.read_csv(url, sep=";")
        return df 
    except Exception as e:
        return e
    
def model_evaluate(y_test, pred, pred_prob):
    '''mae=mean_absolute_error(y_test, pred)
    mse=mean_squared_error(y_test, pred)
    rmse= np.sqrt(mean_squared_error(y_test, pred))
    r2= r2_score( y_test, pred)'''

    accuracy = accuracy_score(y_test, pred)

    # using the ovr for the multi_class 
    roc_auc= roc_auc_score(y_test,pred_prob, multi_class='ovr')

    return accuracy, roc_auc


def main(n_estimators, max_depth):
      
    df= get_data()
    # print(df)

    # splitting the training and test dataset 
    train,test= train_test_split(df)

    # segegating the training and test data

    X_train= train.drop(["quality"], axis=1)
    print(X_train)
    X_test= test.drop(["quality"], axis=1)
    print(X_test)
    y_train= train[["quality"]]
    y_test= test[["quality"]]

    # Training the dataset 

    '''lr = ElasticNet()
    lr.fit(X_train, y_train)
    pred= lr.predict(X_test)'''

    with mlflow.start_run():

    # Random Forest 
        rf=RandomForestClassifier(n_estimators= n_estimators, max_depth= max_depth)
        rf.fit(X_train, y_train)
        pred= rf.predict(X_test)

        pred_prob = rf.predict_proba(X_test)

        # Calling the function to get the evaluation metrics
        accuracy, roc_auc = model_evaluate(y_test, pred, pred_prob)

        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('roc_auc', roc_auc)

        # evaluating the model 
        # mae, mse, rmse, r2 = model_evaluate(y_test,pred)
        # print(f' Mean Absolute Error is {mae}, Mean Squared Error is {mse}, Root Mean Squared Error is {rmse}, R2 score is {r2}')

        print(f'Accuracy for the model : {accuracy}')
        print('\n')
        print (f'The ROC_AUC Score for the model : {roc_auc}')

if __name__== "__main__":

    # Taking the inputs of hyperparameters from the user end to test 
    args =argparse.ArgumentParser()
    args.add_argument("--n_estimators", "-n", default=100 , type= int)
    args.add_argument("--max_depth", "-m", default=10 , type= int)
    parse_args= args.parse_args()
    
    try:
        n_estimators= parse_args.n_estimators
        max_depth= parse_args.max_depth
        main(n_estimators, max_depth)
    except Exception as e:
        raise e