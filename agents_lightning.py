import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import random
import numpy as np
from collections import deque

# --- 模型架構 (與之前一致) ---
class DuelingQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, output_dim)
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return value + (advantages - advantages.mean())

class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

# --- 核心 Agent ---
class LightningDQNAgent(pl.LightningModule):
    def __init__(self, input_dim=64, output_dim=4, lr=1e-3, gamma=0.9, mode="naive"):
        super().__init__()
        self.save_hyperparameters() # 這是 Lightning 的核心功能，用於保存參數
        self.mode = mode
        self.gamma = gamma
        self.output_dim = output_dim
        
        if mode == "dueling":
            self.model = DuelingQNet(input_dim, output_dim)
            self.target_model = DuelingQNet(input_dim, output_dim)
        else:
            self.model = QNet(input_dim, output_dim)
            self.target_model = QNet(input_dim, output_dim)
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.memory = deque(maxlen=5000)
        self.epsilon = 1.0
        
        # 手動初始化優化器，避開對 Trainer 的依賴
        self.opt = optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state_t)).item()

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            if self.mode == "double":
                best_actions = self.model(next_states).argmax(1).unsqueeze(1)
                max_next_q = self.target_model(next_states).gather(1, best_actions).squeeze(1)
            else:
                max_next_q = self.target_model(next_states).max(1)[0]
            expected_q = rewards + (1 - dones) * self.gamma * max_next_q

        # 穩定性技術 1：Huber Loss (SmoothL1Loss)
        loss = nn.SmoothL1Loss()(current_q, expected_q)
        
        self.opt.zero_grad()
        loss.backward()
        
        # 穩定性技術 2：梯度裁剪 (使用標準 PyTorch 函式)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.opt.step()
        
        if self.epsilon > 0.1:
            self.epsilon *= 0.997 

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())