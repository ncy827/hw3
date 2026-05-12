import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# --- 網路架構 ---

# Dueling DQN 網路架構 (用於 HW3-2)
class DuelingQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingQNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        # 狀態價值流 (Value)
        self.value_stream = nn.Linear(128, 1)
        # 動作優勢流 (Advantage)
        self.advantage_stream = nn.Linear(128, output_dim)

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return value + (advantages - advantages.mean())

# 標準 DQN 網路架構
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

# --- 代理人類別 ---

class DQNAgent:
    def __init__(self, input_dim=64, output_dim=4, lr=1e-3, gamma=0.9, mode="naive"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.mode = mode # naive, double, dueling
        
        # 根據模式選擇網路
        if mode == "dueling":
            self.model = DuelingQNet(input_dim, output_dim)
            self.target_model = DuelingQNet(input_dim, output_dim)
        else:
            self.model = QNet(input_dim, output_dim)
            self.target_model = QNet(input_dim, output_dim)
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0 # 初始探索率

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values).item()

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # 當前 Q 值
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 計算目標 Q 值
        with torch.no_grad():
            if self.mode == "double":
                # Double DQN: 用 Online 模型選動作，用 Target 模型算分數
                best_actions = self.model(next_states).argmax(1).unsqueeze(1)
                max_next_q = self.target_model(next_states).gather(1, best_actions).squeeze(1)
            else:
                # Naive DQN & Dueling: 直接用 Target 模型取最大 Q 值
                max_next_q = self.target_model(next_states).max(1)[0]
            
            expected_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = nn.MSELoss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 探索率衰減
        if self.epsilon > 0.1:
            self.epsilon *= 0.995
            
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())