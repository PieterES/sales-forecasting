# Imports
from data_prep import load_and_prepare_data, train_test_split, add_lagged_features, create_model_format
from utils import train_model, save_model, eval_model

# Variables
path = "./data/dataset.csv"
split = 0.8
range_of_lags = [7, 14, 21] # 1 week ago, 2 weeks ago, 3 weeks ago
feature_to_lag = 'UnitSales'
save_location = "./models/forecasting_model.pkl"

# Data preprocessing steps
df_prep = load_and_prepare_data(path)
train_df, test_df_filtered = train_test_split(df_prep, split)

# make the lags per dataset (no data leakage) and also do the NaN filtering per set
train_df_lag = add_lagged_features(train_df, range_of_lags, feature_to_lag)
test_df_lag = add_lagged_features(test_df_filtered, range_of_lags, feature_to_lag)

# We convert the data in the required format for the model (label y and features x)
train_y, train_X = create_model_format(train_df_lag)
test_y, test_X = create_model_format(test_df_lag)

# Model steps
model = train_model(train_X, train_y)
y_pred = model.predict(test_X)

eval_model(y_pred, test_y)

save_model(model, save_location)