from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class ProjectAgent:
    def __init__(self, state_dim, action_dim, gamma=0.93, lr=1e-3, batch_size=32, buffer_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.q_network = DeepQNetwork(state_dim, action_dim)
        self.target_network = DeepQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer = deque(maxlen=buffer_size)

        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def act(self, observation, use_random=False):
        if use_random or random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(observation).unsqueeze(0)
                q_values = self.q_network(state)
                return torch.argmax(q_values).item()

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        #for i, state in enumerate(states):
        #    print(f"State {i}: {state}, Type: {type(state)}")
        #    if isinstance(state, (list, tuple)):
        #        for j, part in enumerate(state):
        #            print(f"  Part {j}: {part}, Type: {type(part)}, Shape: {np.shape(part)}")
        #
        #    state = np.array(state).flatten()

       #print("Faulty states are ", type(states))
        #print("Faulty states are ", np.stack(states))
        states = torch.FloatTensor(np.stack(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.stack(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def add_to_replay_buffer(self, transition):
        state, action, reward, next_state, done = transition
        #state = np.array(state, dtype=object).flatten()
        #next_state = np.array(next_state, dtype=object).flatten()
        self.replay_buffer.append((state, action, reward, next_state, done))


if __name__ == "__main__":
    # Training loop
    env.reset()
    agent = ProjectAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    num_episodes = 300
    update_target_every = 20
    reward_history = []

    for episode in range(num_episodes):
        state, _  = env.reset()
        total_reward = 0

        for t in range(200):  # max steps per episode
            action = agent.act(state, use_random=True)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            agent.add_to_replay_buffer((state, action, reward, next_state, done))
            agent.train_step()
            state = next_state
            total_reward += reward

            if done:
                break

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        if episode % update_target_every == 0:
            agent.update_target_network()

        total_reward = total_reward / 200
        reward_history.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    plt.plot(reward_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
    # Save the model
    agent.save("dqn_model.pth")