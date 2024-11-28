import numpy as np

class RLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.01, discount_factor=0.95, exploration_rate=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.exploration_min = 0.01

        # Q-table for discrete actions
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        """Choose an action based on the current state."""
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(0, self.action_size)  # Random action
        else:
            return np.argmax(self.q_table[state])  # Best action

    def learn(self, state, action, reward, next_state):
        """Update Q-values using the Q-Learning formula."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error

        # Decay exploration rate
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
