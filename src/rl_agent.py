import numpy as np
import pickle
import os

class RLAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.5, epsilon_min=0.1, epsilon_decay=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay factor for epsilon

        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(-5, 5)  # Adjust inventory by -5 to +5 units
        else:
            return np.argmax(self.q_table[state]) - 5  # Offset to handle negative adjustments

    def learn(self, state, action, reward, next_state):
        action_idx = action + 5  # Shift action space to positive index range
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action_idx]
        self.q_table[state, action_idx] += self.alpha * td_error

        # Decay epsilon after learning
        self._decay_epsilon()

    def _decay_epsilon(self):
        # Reduce epsilon by the decay factor but ensure it doesn't go below epsilon_min
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_model(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(self.q_table, file)
        print(f"Model saved to {file_path}")

    @classmethod
    def load_model(cls, file_path):
        if not os.path.exists(file_path):
            print(f"Model file not found at {file_path}")
            return None
        with open(file_path, 'rb') as file:
            q_table = pickle.load(file)
        print(f"Model loaded from {file_path}")
        state_size, action_size = q_table.shape
        agent = cls(state_size, action_size)
        agent.q_table = q_table
        return agent
