import math
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def convert_log_to_units(log_value):
    """Converts the predicted log value from the model to actual units.

    Args:
        log_value (float): The predicted log value of the model.

    Returns:
        units (int): The actual predicted units used for analysis.
    """
    return int(math.exp(log_value))


def train_model(X_train, y_train):
    """Trains the model using the training dataset. Default parameters are set, but can be adapted to requirements.

    Args:
        X_train (DataFrame): The training feature columns used to train the model: [StoreCount, ShelfCapacity, PromoShelfCapacity, IsPromo, ItemNumber, CategoryCode, GroupCode, month, weekday, UnitSales_-7, UnitSales_-14, UnitSales_-21]
        y_train (Series): The target column feature used to train the model: [UnitSales]

    Returns:
        model (RandomForestRegressor): The fitted RandomForestRegressor model to the training data.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_features=round(len(X_train.columns) / 3),
        max_depth=len(X_train.columns),
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, filename):
    """Saves the model in the correct folder

    Args:
        model (RandomForestRegressor): The fitted RandomForestRegressor model.
        filename (str): The location where the model should be saved.
    """
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def load_model():
    """Loads the trained model for use.

    Returns:
        model (RandomForestRegressor): The trained model.
    """
    # Ensure the path to the model file is correct.
    with open("./models/forecasting_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


def eval_model(y_pred, y_test):
    """Evaluates the model based on two metrics, Mean Squared Error and Mean Absolute Error.

    Args:
        y_pred (Series): The predicted amount of units by the model.
        y_test (Series): The actual amount of units, with which we compare our models prediction.
    """
    print("Model evaluation:")
    print("Mean Squared Error: ", mean_squared_error(y_pred, y_test, squared=False))
    print("Mean Absolute Error: ", mean_absolute_error(y_pred, y_test))
