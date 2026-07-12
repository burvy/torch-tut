import pandas as pd
import numpy as np

def main():
    print("Hello World!")

    # loading dataset
    print("Loading Data...")
    X_data = pd.read_csv("../input_housing/X_HousePrice.csv").set_index(["Unnamed: 0", "Unnamed: 1"]).info()
    X_train, X_test = X_data.loc["train"], X_data.loc["test"]

    y_data = pd.read_csv("../input_housing/y_HousePrice.csv").set_index(["Unnamed: 0", "Id"]).info()
    y_train, y_test = y_data.loc["train"], y_data.loc["test"]

    y_train_log, y_test_log = np.log(y_train.copy()).squeeze(), np.log(y_test.copy()).squeeze()
    print("Data Loaded!")

if __name__ == "__main__":
    main()
