import pandas as pd
import matplotlib.pyplot as plt

def generate_energy_graph():
    """Generates a graph of energy consumption trends."""
    df = pd.read_csv("data/energy_data.csv")
    
    plt.figure(figsize=(8, 4))
    plt.plot(df["date"], df["consumption"], marker="o", linestyle="-", color="blue", label="Energy Consumption")
    plt.xlabel("Date")
    plt.ylabel("Energy (kWh)")
    plt.title("Weekly Energy Consumption Trend")
    plt.legend()
    
    plt.savefig("src/web/static/energy_graph.png")
    plt.close()

