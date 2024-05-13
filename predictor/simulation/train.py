import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
import numpy as np

# Load the CSV data into pandas DataFrames
upload_data = pd.read_csv('upload_times.csv')
download_data = pd.read_csv('download_times.csv')

# Separate the features (memory size) and target variables (times)
X_upload = upload_data['MemSize'].values.reshape(-1, 1)
y_upload = upload_data['Time (microseconds)'].values
X_download = download_data['MemSize'].values.reshape(-1, 1)
y_download = download_data['Time (microseconds)'].values

# Split the data into train and test sets
X_train_upload, X_test_upload, y_train_upload, y_test_upload = train_test_split(X_upload, y_upload, test_size=0.3, random_state=42)
X_train_download, X_test_download, y_train_download, y_test_download = train_test_split(X_download, y_download, test_size=0.3, random_state=42)

# Hyperparameter tuning
param_grid = {
    'num_leaves': [31, 63, 127],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [-1, 5, 10],
    'min_data_in_leaf': [20, 50, 100],
    'bagging_fraction': [0.6, 0.8, 1.0],
    'feature_fraction': [0.6, 0.8, 1.0]
}

upload_model = RandomizedSearchCV(lgb.LGBMRegressor(), param_grid, n_iter=50, cv=5, scoring='neg_mean_squared_error', random_state=42)
upload_model.fit(X_train_upload, y_train_upload)

download_model = RandomizedSearchCV(lgb.LGBMRegressor(), param_grid, n_iter=50, cv=5, scoring='neg_mean_squared_error', random_state=42)
download_model.fit(X_train_download, y_train_download)

# Save the trained models to files
upload_model.best_estimator_.booster_.save_model('upload_model.txt')
download_model.best_estimator_.booster_.save_model('download_model.txt')

# Evaluate the models on the test set
upload_preds = upload_model.predict(X_test_upload)
download_preds = download_model.predict(X_test_download)

upload_mse = metrics.mean_squared_error(y_test_upload, upload_preds)
download_mse = metrics.mean_squared_error(y_test_download, download_preds)

print(f'Upload MSE: {upload_mse}')
print(f'Download MSE: {download_mse}')

# Print the best hyperparameters
print('Best hyperparameters for upload model:')
print(upload_model.best_params_)
print('Best hyperparameters for download model:')
print(download_model.best_params_)

# Predict the transfer time for a given input memory size
input_mem_size = 1111111  # Example: 1111111 bytes
predicted_upload_time = upload_model.predict(np.array([input_mem_size]).reshape(1, -1))
predicted_download_time = download_model.predict(np.array([input_mem_size]).reshape(1, -1))

print(f'Predicted upload time for {input_mem_size}: {predicted_upload_time[0]} μs')
print(f'Predicted download time for {input_mem_size}: {predicted_download_time[0]} μs')