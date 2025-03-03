import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Load test data
df = pd.read_csv("../../data/processed_data.csv")
X_test = df[['Temperature (°C)', 'Occupancy']]
y_test = df['Energy Consumption (kWh)']

# Load trained model
model = joblib.load("../../src/models/energy_model.pkl")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"Model Evaluation Completed! Mean Absolute Error: {mae:.2f} kWh")

