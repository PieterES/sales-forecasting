import datetime

import numpy as np
import pandas as pd


def load_and_prepare_data(path):
    """First step in data preprocessing, loads the data, transforms the target column and adds the month and weekdays.

    Args:
        path (str): The location of the dataset.

    Returns:
        DataFrame: Returns a cleaned version of the original dataset.
    """
    df_prep = pd.read_csv(path, sep=";", header=0)
    df_prep["UnitSales"] = np.log(df_prep["UnitSales"])
    df_prep["DateKey"] = pd.to_datetime(df_prep["DateKey"], format="%Y%m%d")
    df_prep["month"] = df_prep["DateKey"].dt.month
    df_prep["weekday"] = df_prep["DateKey"].dt.weekday
    # Drop null values
    df_prep_clean_0 = df_prep[df_prep["UnitSales"].notnull()].copy()
    df_prep_clean = df_prep_clean_0[df_prep_clean_0["ShelfCapacity"].notnull()].copy()

    df_prep_clean["month"] = df_prep_clean["month"].astype("category")
    df_prep_clean["weekday"] = df_prep_clean["weekday"].astype("category")

    return df_prep_clean


def train_test_split(df, split):
    """Splits the dataframe into training and test set.

    Args:
        df (DataFrame): The original, cleaned dataset.
        split (int): The ratio of training and test split, default is 80/20.

    Returns:
        training_dataset: The dataset used to train the model.
        test_datset: The datset used to evaluate the model.
    """
    df_to_split = df.copy()

    # We split the data in a train set and a test set, we do this, 80, 20 percent respectively.
    nr_of_unique_dates = len(df_to_split.DateKey.unique())
    train_split_delta = round(nr_of_unique_dates * split)
    train_split_date = df_to_split.DateKey.dt.date.min() + datetime.timedelta(
        days=train_split_delta
    )

    tr_df = df_to_split[df_to_split["DateKey"].dt.date <= train_split_date].copy()
    tst_df = df_to_split[df_to_split["DateKey"].dt.date > train_split_date].copy()

    tr_df["GroupCode"] = tr_df["GroupCode"].astype("category")
    tr_df["ItemNumber"] = tr_df["ItemNumber"].astype("category")
    tr_df["CategoryCode"] = tr_df["CategoryCode"].astype("category")

    # determine unique item numbers, and filter the validation and test on these
    items_we_train_on = tr_df["ItemNumber"].unique()
    test_df_filtered = tst_df[tst_df["ItemNumber"].isin(items_we_train_on)].copy()

    test_df_filtered["GroupCode"] = test_df_filtered["GroupCode"].astype("category")
    test_df_filtered["ItemNumber"] = test_df_filtered["ItemNumber"].astype("category")
    test_df_filtered["CategoryCode"] = test_df_filtered["CategoryCode"].astype(
        "category"
    )

    return tr_df, test_df_filtered


def add_lagged_features(input_df, lag_iterator, feature):
    """Adds the lagged features, -7, -14, -21, to the dataframe based on the feature.

    Args:
        input_df (DataFrame): The dataframe without lagged features.
        lag_iterator (list): The list of lag days old.
        feature (column): The feature column we want to use for the lags.

    Returns:
        DataFrame: Dataframe with the added lagged features, als dropping the original DateKey feature column.
    """
    output_df = input_df.copy()
    for lag in lag_iterator:
        df_to_lag = input_df[["DateKey", "ItemNumber", feature]].copy()
        # we add the nr of days equal to the lag we want
        df_to_lag["DateKey"] = df_to_lag["DateKey"] + datetime.timedelta(days=lag)

        # the resulting dataframe contains sales data that is lag days old for the date that is in that row
        df_to_lag = df_to_lag.rename(columns={feature: feature + "_-" + str(lag)})

        # we join this dataframe on the original dataframe to add the lagged variable as feature
        output_df = output_df.merge(df_to_lag, how="left", on=["DateKey", "ItemNumber"])
    # drop na rows that have been caused by these lags'
    output_df = output_df.drop(columns=["DateKey"])
    return output_df.dropna()


def create_model_format(df):
    """Splits the DataFrame into target column and feature columns.

    Args:
        df (DataFrame): Original DataFrame with the target and feature columns together.

    Returns:
        Series: Target feature column.
        DataFrame: DataFrame with all the feature columns used for training.
    """
    return df["UnitSales"], df.drop(columns=["UnitSales"])
