import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define paths
data_path = r"C:\Users\NICY\energy_consumption_prediction\data\updated_merged_data.csv"
model_path_electricity = r"C:\Users\NICY\energy_consumption_prediction\src\models\electricity_model.pkl"
model_path_energy = r"C:\Users\NICY\energy_consumption_prediction\src\models\total_energy_model.pkl"

# Load dataset
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Error: Dataset not found at {data_path}")

data = pd.read_csv(data_path)

# Validate dataset
required_features = ["Temperature (°C)", "Humidity (%)", "Occupancy (People)", "Wind Speed (km/h)", "Solar Radiation (W/m²)", "Precipitation (mm)"]
required_targets = ["Electricity Consumption (kWh)", "Total Energy Consumption (kWh)"]

# Check if required columns exist
missing_features = [col for col in required_features if col not in data.columns]
missing_targets = [col for col in required_targets if col not in data.columns]

if missing_features:
    raise KeyError(f"❌ Missing feature columns: {missing_features}")
if missing_targets:
    raise KeyError(f"❌ Missing target columns: {missing_targets}")

# Select relevant data
X = data[required_features]
y_electricity = data["Electricity Consumption (kWh)"]
y_total_energy = data["Total Energy Consumption (kWh)"]

# Split data
X_train, X_test, y_train_electricity, y_test_electricity = train_test_split(X, y_electricity, test_size=0.2, random_state=42)
X_train, X_test, y_train_total_energy, y_test_total_energy = train_test_split(X, y_total_energy, test_size=0.2, random_state=42)

# Train models
electricity_model = RandomForestRegressor(n_estimators=100, random_state=42)
total_energy_model = RandomForestRegressor(n_estimators=100, random_state=42)

electricity_model.fit(X_train, y_train_electricity)
total_energy_model.fit(X_train, y_train_total_energy)

# Evaluate models
y_pred_electricity = electricity_model.predict(X_test)
y_pred_total_energy = total_energy_model.predict(X_test)

mae_electricity = mean_absolute_error(y_test_electricity, y_pred_electricity)
mse_electricity = mean_squared_error(y_test_electricity, y_pred_electricity)

mae_total_energy = mean_absolute_error(y_test_total_energy, y_pred_total_energy)
mse_total_energy = mean_squared_error(y_test_total_energy, y_pred_total_energy)

print(f"\n📊 Electricity Model: MAE={mae_electricity:.2f}, MSE={mse_electricity:.2f}")
print(f"📊 Total Energy Model: MAE={mae_total_energy:.2f}, MSE={mse_total_energy:.2f}")

# Save trained models
joblib.dump(electricity_model, model_path_electricity)
joblib.dump(total_energy_model, model_path_energy)

print(f"\n✅ Electricity Model saved at: {model_path_electricity}")
print(f"✅ Total Energy Model saved at: {model_path_energy}")
