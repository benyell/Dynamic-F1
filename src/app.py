import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
from rl_agent import RLAgent
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.rl_environment import SupplyChainEnvironment

# File paths
processed_file = "data/final_processed_data.csv"
schedule_file = "data/f1_race_schedule.csv"
distances_file = "data/constructors_distances.csv"
model_file = "models/trained_rl_agent.pkl"

# Load datasets
try:
    race_schedule = pd.read_csv(schedule_file)
    processed_data = pd.read_csv(processed_file)
    processed_data["date"] = pd.to_datetime(processed_data["date"])
    st.sidebar.success("Data loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
    st.stop()

# Sidebar - Configurations
st.sidebar.title("F1 Supply Chain Management")
selected_race = st.sidebar.selectbox("Select a Race", race_schedule["EventName"])
max_inventory = st.sidebar.number_input(
    "Max Inventory (Normalized Scale)", 
    min_value=100, 
    max_value=1000, 
    value=500, 
    step=50
)
max_days = st.sidebar.number_input(
    "Simulation Duration (Days)", 
    min_value=10, 
    max_value=100, 
    value=50, 
    step=10
)
# Header
st.title("F1 Supply Chain Management Dashboard")
st.write("Analyze and optimize the F1 supply chain using telemetry and RL-based decisions.")

# Section 1: Race Schedule Viewer
st.header("Race Schedule")
st.dataframe(race_schedule)

# Section 2: Inventory, Transport Costs, and Demand Trends
st.header("Inventory, Transport Costs, and Demand Trends")

# Inventory Levels Over Time
fig_inventory = px.line(
    processed_data,
    x="date",
    y="inventory_level",
    title="Inventory Levels Over Time",
    labels={"date": "Date", "inventory_level": "Inventory Level (Normalized)"},
)
st.plotly_chart(fig_inventory)

# Transport Costs Over Time
fig_transport = px.line(
    processed_data,
    x="date",
    y="transport_cost",
    title="Transport Costs Over Time",
    labels={"date": "Date", "transport_cost": "Transport Cost (Normalized)"},
)
st.plotly_chart(fig_transport)

# Demand Over Time
fig_demand = px.line(
    processed_data,
    x="date",
    y="demand",
    title="Demand Trends Over Time",
    labels={"date": "Date", "demand": "Demand (Normalized)"},
)
st.plotly_chart(fig_demand)

# RL Agent Integration Section
st.header("RL Agent Decisions and Metrics")

# Load RL Agent
try:
    agent = RLAgent.load_model(model_file)
    st.success("Trained RL agent loaded successfully.")
except Exception as e:
    st.error(f"Error loading the RL agent: {e}")
    agent = None

# Function to calculate metrics
def calculate_metrics(results_df):
    total_reward = results_df["Reward"].sum()
    positive_reward_rate = (results_df["Reward"] > 0).mean() * 100
    mean_inventory_level = results_df["Inventory After Action"].mean()
    total_actions = len(results_df)
    avg_action = results_df["Action Taken (Inventory Adjustment)"].mean()
    return {
        "Total Reward": total_reward,
        "Positive Reward Rate (%)": positive_reward_rate,
        "Mean Inventory Level (Normalized)": mean_inventory_level,
        "Total Actions Taken": total_actions,
        "Average Action Adjustment (Normalized)": avg_action,
    }

# Display RL agent decisions and metrics if available
if agent:
    def simulate_rl(env):
        state = env.reset()
        results = []
        done = False

        while not done:
            inventory_before = env.inventory_level
            action = agent.choose_action(state)
            state, reward, done = env.step(action)
            inventory_after = env.inventory_level
            results.append({
                "Day": env.current_day,
                "Inventory Before Action": inventory_before,
                "Inventory After Action": inventory_after,
                "Action Taken (Inventory Adjustment)": action,
                "Reward": reward,
            })
        return pd.DataFrame(results)

    # Simulate Training
    env_train = SupplyChainEnvironment(
        data_file=processed_file,
        distances_file=distances_file,
        max_days=max_days,
        max_inventory=max_inventory,
    )
    train_results_df = simulate_rl(env_train)

    # Simulate Testing
    env_test = SupplyChainEnvironment(
        data_file=processed_file,
        distances_file=distances_file,
        max_days=max_days,
        max_inventory=max_inventory,
    )
    test_results_df = simulate_rl(env_test)

    # Calculate metrics
    train_metrics = calculate_metrics(train_results_df)
    test_metrics = calculate_metrics(test_results_df)

    # Display Metrics
    st.subheader("Training Metrics")
    for metric, value in train_metrics.items():
        st.write(f"**{metric}:** {value}")

    st.subheader("Testing Metrics")
    for metric, value in test_metrics.items():
        st.write(f"**{metric}:** {value}")

    # Visualize Inventory Changes
    st.subheader("Inventory Changes During Training")
    fig_train_inventory = px.line(
        train_results_df,
        x="Day",
        y=["Inventory Before Action", "Inventory After Action"],
        title="Inventory Levels Before and After RL Agent Actions (Training)",
        labels={"Day": "Day", "value": "Inventory Level", "variable": "Inventory State"},
    )
    st.plotly_chart(fig_train_inventory)

    st.subheader("Inventory Changes During Testing")
    fig_test_inventory = px.line(
        test_results_df,
        x="Day",
        y=["Inventory Before Action", "Inventory After Action"],
        title="Inventory Levels Before and After RL Agent Actions (Testing)",
        labels={"Day": "Day", "value": "Inventory Level", "variable": "Inventory State"},
    )
    st.plotly_chart(fig_test_inventory)

else:
    st.warning("RL agent decisions are not available. Train the agent and reload the app.")
