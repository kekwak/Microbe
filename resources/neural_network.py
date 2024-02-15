import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import random as rd

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, lr, gamma, device):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

        self.lr = lr
        self.gamma = gamma
        self.device = device

        self.state_size = state_size
        self.action_size = action_size

        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        self.loss_function = nn.SmoothL1Loss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def choose_action(self, state, epsilon=0):
        if rd.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_values = self(state)
            return np.argmax(action_values.cpu().numpy())
        else:
            return rd.choice(list(range(0, self.action_size)))
    
    def train_step(self, state, action, reward, next_state):
        self.state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        self.action = torch.tensor([action], device=self.device)
        self.reward = torch.tensor([reward], device=self.device)

        self.current_q = self(self.state).squeeze(0)[self.action]
        self.max_next_q = self(self.next_state).squeeze(0).max()
        self.expected_q = (self.reward + (self.gamma * self.max_next_q)).detach()

        self.loss = self.loss_function(self.current_q, self.expected_q)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

class Graph:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.scores_record = list()
        self.ax.grid(True)

        self.fig.canvas.manager.set_window_title('Бактерии - График')

        plt.xlabel(f'Эпизоды')
        plt.ylabel('Средняя награда')
        plt.title('Средняя награда по эпизодам')

        self.line, = self.ax.plot(list(), list())
    
    def update(self, reward):
        self.scores_record.append(reward)

        self.line.set_data(np.arange(len(self.scores_record)), self.scores_record)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
    
    def get_scores_list(self):
        return self.scores_record

class Helper:
    def __init__(self, possible_actions, num_categories):
        self.possible_actions = possible_actions
        self.num_categories = num_categories

        self.old_score = 0
        self.new_score = 0
    
    def set_old_score(self, value):
        self.old_score = value
    
    def set_new_score(self, value):
        self.new_score = value
    
    def get_old_score(self):
        return self.old_score
    
    def get_new_score(self):
        return self.new_score

    def one_hot_encode(self, intersection, intersection_dist):
        one_hot = [0, 0, 0]
        one_hot[intersection-1] = intersection_dist

        return one_hot

    def get_state(self, unit):
        state = []
        for line in unit.lines:
            intersection_recalculated = min(10, 10 / (line.intersection_dist)) if line.intersection_dist != 0 else 0
            state.extend(self.one_hot_encode(line.intersection, intersection_recalculated))
        return state

    def execute_action(self, unit, action):
        unit.move(self.possible_actions[action])

    def get_reward(self, unit, action):
        reward = self.new_score - self.old_score

        inters = [min(10, 10 / (line.intersection_dist)) for line in unit.lines if line.intersection == 3]
        if not inters:
            inters = [0]
        reward += 1 * (10 * len(inters) - sum(inters)) / len(inters) / 10

        if action == 0:
            reward += 0.1

        return reward