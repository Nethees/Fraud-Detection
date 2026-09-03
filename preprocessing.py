

import config

import data_loader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocessing(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE):

    pp_df = df.copy()
    X = pp_df.drop(columns=["Class"])
    y = pp_df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y)
    
    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()

    X_train["Amount"]= amount_scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"]= amount_scaler.transform(X_test[["Amount"]])

    X_train["Time"]= time_scaler.fit_transform(X_train[["Time"]])
    X_test["Time"]= time_scaler.transform(X_test[["Time"]])

    print(f"Training Fraud rate: {y_train.mean() * 100:.4f}%, Test Fraud rate: {y_test.mean() * 100:.4f}%")

    print(X_train.columns, X_train.dtypes)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = data_loader.load_data()
    X_train, X_test, y_train, y_test = preprocessing(df)