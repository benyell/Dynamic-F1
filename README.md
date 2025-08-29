# Dynamic Supply Chain Optimization for Formula 1 Teams Using Reinforcement Learning

## Aim of the Project
This project focuses on applying Reinforcement Learning (RL) to optimize the dynamic supply
chain operations of Formula 1 (F1) teams, which face unique logistical challenges due to tight
schedules, fluctuating demands and global race locations.

Traditional supply chain methods struggle with the real-time adaptability required in such high-stakes environments. 
By leveraging RL, a subfield of Artificial Intelligence that enables agents to learn optimal actions through
interaction and feedback, the project aims to create a model capable of minimizing costs,
improving efficiency, and adapting swiftly to disruptions. 

## Key Features

1. Markov Decision Process-based Supply Chain Modeling: Represents F1 logistics as a sequential decision-making problem
2. Q-Learning Agent: Learns optimal inventory management policies through trial and error
3. Real F1 Data Integration: Uses FastF1 dataset and F1DB repository for realistic simulations
4. Interactive Dashboard: Streamlit-based visualization for exploring model performance
5. Synthetic Data Generation: Creates realistic scenarios including demand spikes and delivery delays

## Technical Architecture

### Markov Decision Process Framework

The system models the supply chain using:
* States: Inventory levels, remaining stock, time until next race, lead times, stock status, and geographical constraints
* Actions: Order placement, supplier selection, shipment expediting, or maintaining current levels
* Rewards: Balance holding costs, stockout penalties, and transportation expenses
* Transitions: Probabilistic updates to inventory and system status

### Q-Learning Implementation

The agent uses the Bellman equation for Q-value updates:
> Q(s,a) = Q(s,a) + α [r + γ max Q(s′, a′) − Q(s,a)]

Where:\
α: Learning rate controlling information integration\
γ: Discount factor balancing immediate vs. future rewards\
r: Immediate reward from the action\
s′: Next state after action execution

## Data Pipeline

### Data Sources
* FastF1 Dataset: Race schedules, telemetry data, pit stop information
* F1DB Repository: Engine/tire manufacturers, constructor details
Synthetic Generation: 180-day simulation with realistic demand patterns

### Preprocessing Steps

Step 1: Telemetry aggregation and pit stop normalization using MinMaxScaler\

Step 2: Missing value imputation for races without telemetry data\

Step 3: Inventory adjustment to handle negative values\

Step 4: Feature normalization across all supply chain variables

## Installation Pipeline
Prerequisites
> pip install -r requirements.txt

Data Collection
* FastF1
  > python src/data_collection.py

* Generate Synthetic Data
  > python src/generate_dataset.py\
  > python src/generate_synthetic_data.py

Running the Model
* Data preprocessing
  > python data_preprocessing.py

* Train and Test the RL agent
  > src/train_rl_agent.py
  > python src/test_rl_agent.py

* Launch interactive dashboard
  > streamlit run src/app.py

## Results & Performance
### Training Phase

* Agent learns through 180 simulated days
* Dynamic epsilon-greedy strategy balances exploration/exploitation
* Q-table updates based on reward feedback and state transitions

![Alt text](images/dashboard-screenshot.png)

![Model Architecture](docs/images/model-architecture.png)

### Key Metrics

* Inventory Optimization: Reduced stockouts while minimizing holding costs
* Adaptive Decision-Making: Improved stability during testing phase
* Cost Minimization: Balanced transportation and inventory expenses

### Visualization Dashboard
The interactive Streamlit dashboard provides:

* Real-time inventory level tracking
* Demand trend analysis with Plotly charts
* Training vs testing comparison plots

## Conclusion
This project successfully demonstrates the viability of reinforcement learning for tackling the dynamic and complex logistical challenges faced by Formula 1 teams. The RL-based supply chain optimization model effectively adapts to real-world scenarios by learning to balance inventory levels, transportation costs, and fluctuating demand patterns through the integration of historical F1 data and realistic synthetic scenarios.

### Key Achievements
  * Adaptive Learning: The Q-learning agent successfully learned optimal inventory management policies through iterative experience
  * Real-world Applicability: Integration of actual F1 data provided realistic training scenarios and validated the model's practical relevance
  * Dynamic Response: Interactive dashboards and performance metrics demonstrate the model's ability to respond effectively to changing conditions
  * Foundational Framework: Established a solid foundation for applying reinforcement learning to high-pressure logistics environments




