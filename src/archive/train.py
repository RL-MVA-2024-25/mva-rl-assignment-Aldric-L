from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import random
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=768
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fclast = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = torch.nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = torch.nn.functional.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = torch.nn.functional.leaky_relu(self.fc4(x), negative_slope=0.01)
        
        #x = torch.tanh(self.fc2(x))
        #x = torch.tanh(self.fc3(x))
        #x = torch.nn.functional.leaky_relu(self.fc3(x), negative_slope=0.01)
        #x = torch.tanh(self.fc4(x))
        return self.fclast(x)
        #return torch.sigmoid(self.fclast(x))

class ProjectAgent: #lr=5e-4, batch_size=128
    def __init__(self, state_dim, action_dim, gamma=0.98, lr=1e-3, batch_size=768, buffer_size=1000000):
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
        #self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.95)  # Decays LR by 0.9 every 20 episodes
        self.replay_buffer = deque(maxlen=buffer_size)

        self.epsilon = 0.95
        self.epsilon_decay = 0.998 #0.995
        self.lr_decay = 0.999 #0.995
        self.epsilon_min = 0.15

        #torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.epsilon_decay, last_epoch=-1)

    def act(self, observation, use_random=False):
        if use_random and random.random() < self.epsilon:
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
            return 0

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.stack(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.stack(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q_values = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, target_q_values) + nn.L1Loss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self, tau=0.05):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        #self.target_network.load_state_dict(self.q_network.state_dict())

    def add_to_replay_buffer(self, transition):
        state, action, reward, next_state, done = transition
        #state = np.array(state, dtype=object).flatten()
        #next_state = np.array(next_state, dtype=object).flatten()
        self.replay_buffer.append((state, action, reward, next_state, done))


if __name__ == "__main__":
    # def seed_everything(seed: int = 42):
    #     random.seed(seed)
    #     os.environ["PYTHONHASHSEED"] = str(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    #     torch.cuda.manual_seed_all(seed)

    def sigmoid(z):
        return 1/(1 + np.exp(-z))

    #seed_everything(seed=42)
    # Training loop
    env.reset()
    agent = ProjectAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)

    #print("Before training: ", evaluate_HIV(agent=agent, nb_episode=5))
    #print("Before training: ", evaluate_HIV_population(agent=agent, nb_episode=20))

    num_episodes = 200
    update_target_every = 1
    reward_history = []
    loss_history = []
    validation_history = []

    state, _  = env.reset()
    first_reward = 0
    first_loss = 0
    for t in range(1028):  # max steps per episode
        action = agent.act(state, use_random=False)
        next_state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        reward = sigmoid(reward/ 100000)
        if reward < 0:
            print("Reward negative ", reward)
        agent.add_to_replay_buffer((state, action, reward, next_state, done))
        first_loss += agent.train_step()
        state = next_state
        first_reward += reward

        if done:
            break
    first_reward = first_reward / (t+1)
    first_validation_score = evaluate_HIV(agent=agent, nb_episode=1)


    for episode in range(num_episodes):
        state, _  = env.reset()
        total_reward = 0
        total_loss = 0
        for t in range(1028):  # max steps per episode
            action = agent.act(state, use_random=True)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            reward = sigmoid(reward/ 100000)
            if reward < 0:
                print("Reward negative ", reward)
            #reward = reward / 200000 + 0.1
            agent.add_to_replay_buffer((state, action, reward, next_state, done))
            total_loss += agent.train_step()
            state = next_state
            total_reward += reward

            if done:
                break

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        total_reward = total_reward / (t+1)
        total_loss = total_loss / (t+1)
        reward_history.append(total_reward/first_reward)
        loss_history.append(total_loss/first_loss)
        validation_score = evaluate_HIV(agent=agent, nb_episode=5)

        if episode % update_target_every == 0  :
            #and (len(validation_history) == 0 or validation_score/first_validation_score >= validation_history[-1])
            #and (len(loss_history) == 0 or total_loss/first_loss <= loss_history[-1])
            agent.update_target_network(tau=0.005)
            print("We update the target network !")
        validation_history.append(validation_score/first_validation_score)

        print(f"Episode {episode}, Total Reward: {total_reward/first_reward}, Total Loss: {total_loss/first_loss}, Epsilon: {agent.epsilon}, Validation Score: {validation_score/first_validation_score}")
        #agent.scheduler.step()

    plt.figure(figsize=(10, 6))  # Adjust the size of the plot
    plt.plot(reward_history, label='Average Reward', color='blue', linewidth=2)
    plt.plot([loss  for loss in loss_history], label='Average Loss', color='red', linewidth=2, linestyle='--')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.title('Training Progress: Reward and Loss over Episodes', fontsize=16)
    plt.grid(alpha=0.3)  # Add a light grid for better readability
    plt.legend(fontsize=12)  # Add a legend for clarity
    plt.tight_layout()  # Ensure the layout fits well
    plt.show()
    # Save the model
    agent.save("dqn_model.pth")

    print("After training: ", evaluate_HIV(agent=agent, nb_episode=5))
    print("After training: ", evaluate_HIV_population(agent=agent, nb_episode=20))
