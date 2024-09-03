import json
import math
import pickle

import pandas as pd


def load_model():
    """Loads the pretrained model using pickle

    Returns:
        model: RandomForestRegressor
    """
    # Ensure the path to the model file is correct.
    with open("./models/forecasting_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


def predict(model, data):
    """Uses the model to predict based on the input data

    Args:
        model (RandomForestRegressor): The model used for prediction
        data (List): _description_

    Returns:
        float: Log value output from the model
    """
    return model.predict(data)


def convert_log_to_units(log_value):
    """Converts the log value output from the model to actual units

    Args:
        log_value (float): Log value output

    Raises:
        ValueError: Raises an error if more than 1 log values are given

    Returns:
        int: Actualy units predicted
    """
    # Extract the single prediction value from the array
    if len(log_value) == 1:  # Ensure there is exactly one item in the array
        return int(math.exp(log_value[0]))
    else:
        raise ValueError("Expected exactly one log value.")


def get_user_input():
    """ Gives the user instructions on how to use the model for prediction

    Raises:
        ValueError: Raises an error if the list is not consistent with the necessary length of input

    Returns:
        DataFrame: Returns a Pandas Dataframe that the model can use to predict.
    """
    print("Please enter the input data as a list:")
    print(
        "[StoreCount, ShelfCapacity, PromoShelfCapacity, IsPromo, ItemNumber, CategoryCode, GroupCode, month, weekday, UnitSales_-7, UnitSales_-14, UnitSales_-21]"
    )
    print(
        "Example: [781, 12602.0, 4922.0, true, 8646, 7292, 5494, 11, 3, 6.190, 6.217, 6.075]"
    )
    user_input = input("Enter your data here: ")

    try:
        # Evaluate the input to convert it from string to actual list
        input_list = json.loads(user_input)

        # Check if the input is indeed a list and has the correct number of elements
        if not isinstance(input_list, list) or len(input_list) != 12:
            raise ValueError("Input must be a list with 12 elements.")

        # Convert boolean string to boolean type

        # Create DataFrame from list
        columns = [
            "StoreCount",
            "ShelfCapacity",
            "PromoShelfCapacity",
            "IsPromo",
            "ItemNumber",
            "CategoryCode",
            "GroupCode",
            "month",
            "weekday",
            "UnitSales_-7",
            "UnitSales_-14",
            "UnitSales_-21",
        ]
        return pd.DataFrame([input_list], columns=columns)

    except (SyntaxError, NameError, TypeError, ValueError) as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    """
    Running the script will load the model, collect the input from the user and print the acutal amount of predicted units.
    """
    model = load_model()
    new_data = get_user_input()  # Collects input and returns it in a DataFrame

    if new_data is not None:
        predictions = predict(model, new_data)  # Returns an array of predictions
        # Convert log-transformed prediction to actual units
        predicted_units = convert_log_to_units(predictions)
        print("Predicted units:", predicted_units)
