from collections import deque
import random
import torch
import torch.nn as nn
from poke_env.player.player import Player


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, tup):
        # state, action, next_state, reward
        self.memory.append(tup)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return self.memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
class DQN(nn.Module):
    def __init__(self, state_size, len_action_space):
        super(DQN, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len_action_space),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class DQNAgent(Player):
    def __init__(self, state_size, action_space, batch_size=128, gamma=0.99):
        super().__init__()
        self.model = DQN(state_size, action_space)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.memory = ReplayMemory(10000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.embed_battle = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
    # double check this function
    def _train_one_step(self):
        batch = self.memory.sample(self.batch_size)
        for state, action, next_state, rwd, terminal in batch:
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                state = torch.tensor([state]).float().to(self.device)
                q_values = self.model(state)
                target_q_values = q_values
                # double check this
                if terminal:
                    target_q_values[0][action] = rwd
                else:
                    target_q_values[0][action] = rwd + self.gamma * torch.max(torch.tensor([next_state]).float().to(self.device)).numpy()
                loss = torch.sum((q_values - target_q_values) ** 2)
                loss.backward()
                self.optimizer.step()
    
    def train_one_episode(self, env):
        env.reset_battles()
        done = False
        state = env.reset()
        
        ct = 0
        while not done:
            print(ct)
            ct += 1
            action = self._best_action(state)
            next_state, rwd, done, _ =  env.step(action)
            self.memory.push((state, action, next_state, rwd, done))
            self._train_one_step()
            state = next_state

        # state = env.reset()
        # env.complete_current_battle()
        
    def set_embed_battle(self, embed_battle):
        self.embed_battle = embed_battle
        
    def _best_action(self, state):
        state = torch.tensor([state]).float().to(self.device)
        q_values = self.model(state)
        action = torch.argmax(q_values).numpy()
        return action
        
    def choose_move(self, battle):
        state = self.embed_battle(battle)
        return self._best_action(state)
        
