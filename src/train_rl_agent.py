from environment.rl_environment import SupplyChainEnvironment
from rl_agent import RLAgent

def train_agent(env, agent, episodes):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        print(f"Episode {episode+1}/{episodes} - Total Reward: {total_reward}")

    print("Training completed.")
    return agent

if __name__ == "__main__":
    env = SupplyChainEnvironment(
        data_file="data/supply_chain_data_with_distances.csv",
        distances_file="data/constructors_distances.csv"
    )
    state_size = len(env.state)
    action_size = 10  # Example: Reorder quantities in steps of 10
    agent = RLAgent(state_size, action_size)

    trained_agent = train_agent(env, agent, episodes=1000)
