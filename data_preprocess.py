import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import numpy as np

def preprocess(train_file, test_file):
    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    old_size_train = train_data.shape[0]
    old_size_test = test_data.shape[0]

    # Remove nans
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    print("Removed {} rows from train data".format(old_size_train - train_data.shape[0]))
    print("Removed {} rows from test data".format(old_size_test - test_data.shape[0]))


    # Mapping dictionary
    mapping_dict = {}

    # Binary encoding
    for col in ["Gender", "Customer Type", "Type of Travel", "Class", "satisfaction"]:
        if col == "Class":
            class_mapping = {"Eco": 0, "Eco Plus": 1, "Business": 2}
            train_data[col] = train_data[col].map(class_mapping)
            test_data[col] = test_data[col].map(class_mapping)
            mapping_dict[col] = class_mapping
        else:
            train_data[col], mapping_dict[col] = pd.factorize(train_data[col])
            test_data[col] = test_data[col].map(dict(zip(mapping_dict[col], range(len(mapping_dict[col])))))

    # Drop first 2 columns
    train_data = train_data.iloc[:, 2:]
    test_data = test_data.iloc[:, 2:]

    # Separate labels
    train_y = torch.tensor(train_data.pop("satisfaction").values)
    test_y = torch.tensor(test_data.pop("satisfaction").values)

    # MinMax scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data[train_data.columns.difference(["Age", "Gender", "Customer Type", "Type of Travel", "Class"])] = scaler.fit_transform(train_data[train_data.columns.difference(["Age", "Gender", "Customer Type", "Type of Travel", "Class"])])
    test_data[test_data.columns.difference(["Age", "Gender", "Customer Type", "Type of Travel", "Class"])] = scaler.transform(test_data[test_data.columns.difference(["Age", "Gender", "Customer Type", "Type of Travel", "Class"])])

    def custom_scale(age, min_age, max_age):
        return (age - min_age) / (max_age - min_age)

    min_age = 0
    max_age = 120

    train_data["Age"] = custom_scale(train_data["Age"], min_age, max_age)
    test_data["Age"] = custom_scale(test_data["Age"], min_age, max_age)

    # Split the training data into train and validation sets
    train_x, val_x, train_y, val_y = train_test_split(train_data, train_y, test_size=0.2, random_state=42)
    train_noise = np.random.normal(0, 0.3, train_x.shape)
    val_noise = np.random.normal(0, 0.3, val_x.shape)
    test_noise = np.random.normal(0, 0.3, test_data.shape)

    train_x = train_x + train_noise
    val_x = val_x + val_noise
    test_data = test_data + test_noise

    # Convert dataframes to PyTorch Tensors
    train_x = torch.tensor(train_x.values, dtype=torch.float)
    val_x = torch.tensor(val_x.values, dtype=torch.float)
    test_x = torch.tensor(test_data.values, dtype=torch.float)
    train_y = train_y.float()
    val_y = val_y.float()
    test_y = test_y.float()

    return train_x, train_y, val_x, val_y, test_x, test_y, mapping_dict
