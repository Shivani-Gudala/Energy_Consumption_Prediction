import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, url_for

# Define base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Define correct template and static directories
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Initialize Flask app with corrected paths
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# Define model directory
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")  # Move up one directory

# Model paths
electricity_model_path = os.path.join(MODEL_DIR, "electricity_model.pkl")
total_energy_model_path = os.path.join(MODEL_DIR, "total_energy_model.pkl")

# Function to load models safely
def load_model(path):
    if os.path.exists(path):
        print(f"✅ Loading model: {path}")
        return joblib.load(path)
    print(f"⚠️ Model file missing: {path}")
    return None

# Load models
electricity_model = load_model(electricity_model_path)
total_energy_model = load_model(total_energy_model_path)

# Function to generate energy-saving suggestions
def generate_suggestions(total_energy, occupancy, solar_radiation):
    suggestions = []
    
    # Solar Radiation
    if solar_radiation > 200:
        suggestions.append("🌞 Consider installing solar panels to reduce grid dependency.")
    if solar_radiation < 50:
        suggestions.append("🌥️ Maximize energy efficiency by reducing artificial lighting usage during daylight hours.")

    # Total Energy Consumption
    if total_energy > 500:
        suggestions.append("🔄 Optimize HVAC system efficiency for energy savings.")
        suggestions.append("⚡ Upgrade to energy-efficient appliances to lower consumption.")
    if total_energy > 1000:
        suggestions.append("🏢 Implement a building energy management system for better control.")

    # Occupancy
    if occupancy > 100:
        suggestions.append("💡 Use automated lighting controls to reduce unnecessary usage.")
        suggestions.append("🛠️ Implement motion sensors in classrooms and offices.")
    if occupancy > 200:
        suggestions.append("🔋 Consider upgrading electrical infrastructure to handle peak loads efficiently.")


    # Optimal Condition
    if not suggestions:
        suggestions.append("✔️ Your energy consumption is within an optimal range.")

    return suggestions


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if electricity_model is None or total_energy_model is None:
            return jsonify({"error": "⚠️ Model files are missing. Please check the server logs."})

        features = ["temperature", "humidity", "precipitation", "wind_speed", "solar_radiation", "occupancy"]
        input_data = []

        for feature in features:
            value = request.form.get(feature)
            if value is None or value.strip() == "":
                return jsonify({"error": f"⚠️ Missing input: {feature}"})
            input_data.append(float(value))

        input_features = np.array([input_data])

        predicted_electricity = electricity_model.predict(input_features)[0]
        predicted_total_energy = total_energy_model.predict(input_features)[0]

        estimated_electricity_cost = predicted_electricity * 30 * 0.12
        estimated_total_energy_cost = predicted_total_energy * 30 * 0.12

        suggestions = generate_suggestions(predicted_total_energy, input_data[-1], input_data[-2])

        if not os.path.exists(STATIC_DIR):
            os.makedirs(STATIC_DIR)

        plt.figure(figsize=(6, 4))
        labels = ["Electricity", "Total Energy"]
        values = [predicted_electricity, predicted_total_energy]

        plt.bar(labels, values, color=['blue', 'green'])
        plt.xlabel("Energy Type")
        plt.ylabel("Consumption (kWh)")
        plt.title("Predicted Energy Consumption")

        graph_path = os.path.join(STATIC_DIR, "prediction_graph.png")
        plt.savefig(graph_path, format='png', dpi=100)
        plt.close()

        if not os.path.exists(graph_path):
            print(f"❌ Graph save failed! Check directory: {STATIC_DIR}")

        graph_url = url_for('static', filename='prediction_graph.png', _external=True)

        return render_template("prediction.html", 
                               electricity=f"{predicted_electricity:.2f} kWh", 
                               total_energy=f"{predicted_total_energy:.2f} kWh",
                               electricity_cost=f"${estimated_electricity_cost:.2f}",
                               total_energy_cost=f"${estimated_total_energy_cost:.2f}",
                               suggestions=suggestions,
                               graph_url=graph_url)

    except Exception as e:
        print(f"⚠️ Error: {str(e)}")
        return jsonify({"error": f"⚠️ An error occurred: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)
