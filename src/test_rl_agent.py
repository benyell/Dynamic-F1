from environment.rl_environment import SupplyChainEnvironment
from rl_agent import RLAgent

def test_agent(env, agent):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward

        state = next_state

    print(f"Test Completed - Total Reward: {total_reward}")

if __name__ == "__main__":
    env = SupplyChainEnvironment(
        data_file="data/supply_chain_data_with_distances.csv",
        distances_file="data/constructors_distances.csv"
    )
    state_size = len(env.state)
    action_size = 10
    agent = RLAgent(state_size, action_size)

    test_agent(env, agent)
