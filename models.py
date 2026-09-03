

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import data_loader
from preprocessing import preprocessing

import config

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(**config.LOGISTIC_REGRESSION_PARAMS)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":

    df = data_loader.load_data()
    X_train, X_test, y_train, y_test = preprocessing(df)
    model = train_logistic_regression(X_train, y_train)
    print("Model trained successfully.")

    y_pred = model.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["Genuine", "Fraud"]))