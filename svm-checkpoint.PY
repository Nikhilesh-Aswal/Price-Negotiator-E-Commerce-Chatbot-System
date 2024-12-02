import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib

# Step 1: Create project folders
os.makedirs('price_negotiator_bot/data', exist_ok=True)
os.makedirs('price_negotiator_bot/model_training', exist_ok=True)
os.makedirs('price_negotiator_bot/notebooks', exist_ok=True)

# Step 2: Load the dataset
data_path = 'price_negotiator_bot/data/negotiation_data.csv'
data = pd.read_csv(data_path)

# Step 3: Calculate discount percentage and select features/target
data['discount'] = ((data['actual_price'] - data['negotiation_price']) / data['actual_price']) * 100
X = data[['actual_price']]
y = data['discount']

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train the SVM model
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Step 7: Save the model and scaler
joblib.dump(model, 'price_negotiator_bot/model_training/model_svm.joblib')
joblib.dump(scaler, 'price_negotiator_bot/model_training/scaler.joblib')

# Step 8: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
