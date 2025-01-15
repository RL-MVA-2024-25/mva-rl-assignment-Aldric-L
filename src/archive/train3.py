# Imports

import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt


from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population


env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=350
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

###############################################################################
# We implement a DQN



class ProjectAgent:
    def act(self, observation, use_random=False):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path):
        self.path = path + "/model_v3.pt"
        torch.save(self.model.state_dict(), self.path)
        return 

    def load(self):
        device = torch.device('cpu')
        self.path = os.getcwd() + "/model_v3.pt"
        self.model = self.myDQN({}, device)
        self.model.load_state_dict(torch.load(self.path, map_location=device))
        self.model.eval()
        return 
    
    # Now we add our methods
    # Function to take the greedy action
    def act_greedy(self, myDQN, state):
        device = "cuda" if next(myDQN.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = myDQN(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()
        
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step() 

    def myDQN(self, config, device):
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n  # Number of possible actions
        nb_neurons = 256  # Hidden layer size

        # Define the dueling DQN model
        class DuelingDQN(nn.Module):
            def __init__(self, state_dim, n_action, nb_neurons):
                super(DuelingDQN, self).__init__()
                # Shared layers
                dropout_prob = 0.2
                self.feature = nn.Sequential(
                    nn.Linear(state_dim, nb_neurons),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(nb_neurons, nb_neurons),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(nb_neurons, nb_neurons),
                    nn.LeakyReLU(negative_slope=0.01),
                )
                # Value stream
                self.value = nn.Sequential(
                    nn.Dropout(p=dropout_prob),
                    #nn.Linear(nb_neurons, nb_neurons),
                    #nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(nb_neurons, nb_neurons),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(nb_neurons, 1)  # Output a single value V(s)
                )
                # Advantage stream
                self.advantage = nn.Sequential(
                    nn.Dropout(p=dropout_prob),
                    #nn.Linear(nb_neurons, nb_neurons),
                    #nn.LeakyReLU(negative_slope=0.01),
                    #nn.Linear(nb_neurons, nb_neurons),
                    #nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(nb_neurons, nb_neurons),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(nb_neurons, n_action)  # Output advantages for all actions A(s, a)
                )

                self.residual = nn.Linear(state_dim, nb_neurons)

            def forward(self, x):
                residual = self.residual(x)
                features = self.feature(x) + residual
                value = self.value(features)
                advantage = self.advantage(features)
                # Combine streams to compute Q-values
                q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
                return q_values

        # Instantiate the model and move it to the device
        model = DuelingDQN(state_dim, n_action, nb_neurons).to(device)
        return model


    def train(self):

        config = {'nb_actions': env.action_space.n,
                'learning_rate': 0.00005,
                'gamma': 0.98,
                'buffer_size': 1000000,
                'epsilon_min': 0.05,
                'epsilon_max': 1.,
                'epsilon_decay_period': 21000, 
                'epsilon_delay_decay': 300,
                'batch_size': 790,
                'gradient_steps': 1,
                'update_target_strategy': 'ema', # or 'ema' (tried but replace seems to work better for this model/problem)
                'update_target_freq': 400,
                'update_target_tau': 0.001,
                'criterion': torch.nn.SmoothL1Loss()}

    
        device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        self.model = self.myDQN(config, device)
        self.target_model = deepcopy(self.model).to(device)

        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']

        # epsilon greedy strategy
        epsilon_max = config['epsilon_max']
        epsilon_min = config['epsilon_min']
        epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        epsilon_step = (epsilon_max-epsilon_min)/epsilon_stop

        # memory buffer
        self.memory = ReplayBuffer(config['buffer_size'], device)

        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer2 = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=88, gamma=0.1)

        nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1

        # target network
        update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005


        previous_val = 0

        max_episode = 350 

        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = epsilon_max
        step = 0

        ## TRAIN NETWORK

        while episode < max_episode:
            # update epsilon
            if step > epsilon_delay:
                epsilon = max(epsilon_min, epsilon-epsilon_step)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act_greedy(self.model, state)
            
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            
            for _ in range(nb_gradient_steps + (episode > 100)): 
                self.gradient_step()

            if update_target_strategy == 'replace': 
                if step % update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            
            step += 1
            if done or trunc:
                episode += 1
                
                validation_score = evaluate_HIV(agent=self, nb_episode=1)
                                
                print(f"Episode {episode:3d} | "
                      f"Epsilon {epsilon:6.2f} | "
                      f"Batch Size {len(self.memory):5d} | "
                      f"Episode Return {episode_cum_reward:.2e} | "
                      f"Evaluation Score {validation_score:.2e}")
                state, _ = env.reset()

                if validation_score > previous_val:
                    previous_val = validation_score
                    self.best_model = deepcopy(self.model).to(device)
                    path = os.getcwd()
                    self.save(path)
                episode_return.append(episode_cum_reward)
                
                episode_cum_reward = 0
            else:
                state = next_state

        self.model.load_state_dict(self.best_model.state_dict())
        path = os.getcwd()
        self.save(path)
        return episode_return

###############################################################################
# We create a replay buffer (replay_buffer2 given in class):
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity 
        self.data = []
        self.index = 0
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    

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
    agent = ProjectAgent()
    agent.train()