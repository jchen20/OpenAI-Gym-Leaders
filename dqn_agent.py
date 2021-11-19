from collections import deque
import random
import torch
import torch.nn as nn
import numpy as np
from poke_env.player.player import Player, BattleOrder
from poke_env.player.battle_order import ForfeitBattleOrder

from utils import player_action_to_move


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

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
            nn.Linear(state_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, len_action_space),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class DQNAgent(Player):
    def __init__(self, state_size, action_space, batch_size=32, gamma=0.99, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = DQN(state_size, action_space)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-6,
                                          weight_decay=1e-4)
        self.memory = ReplayMemory(1000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.embed_battle = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    # double check this function // George: changed this to actually use the batch.
    def _train_one_step(self):
        batch = self.memory.sample(self.batch_size)
        state, action, next_state, rwd, terminal = zip(*batch)
        with torch.set_grad_enabled(True):
            self.optimizer.zero_grad()
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            rwd = torch.tensor(rwd, dtype=torch.float32).to(self.device)
            terminal = torch.tensor(terminal, dtype=bool).to(self.device)
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            q_values = self.model(state)
            target_q_values = torch.clone(q_values).detach() # copies
            target_q_values[range(len(q_values)), action] = rwd
            valid_next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)[~terminal]
            target_q_values[~terminal, action[~terminal]] += self.gamma * torch.max(self.model(valid_next_state), dim=1)[0].detach()
            loss = torch.sum((q_values - target_q_values) ** 2)
            loss.backward()
            self.optimizer.step()

    def train_one_episode(self, env):
        env.reset_battles()
        done = False
        state = env.reset()

        ct = 0
        while not done:
            # print(ct)
            ct += 1
            action = self._best_action(state)
            next_state, rwd, done, _ = env.step(action)
            performed_action = env.last_action
            self.memory.push((state, performed_action, next_state, rwd, done))
            state = next_state
            if ct % (self.batch_size // 4) == 0:
                self._train_one_step()
        # state = env.reset()
        # env.complete_current_battle()
    
    def run_one_episode(self, env):
        env.reset_battles()
        done = False
        state = env.reset()

        ct = 0
        while not done:
            # print(ct)
            ct += 1
            action = self._best_action(state)
            next_state, rwd, done, _ = env.step(action)
            performed_action = env.last_action
            self.memory.push((state, performed_action, next_state, rwd, done))
            state = next_state
        

    def set_embed_battle(self, embed_battle):
        self.embed_battle = embed_battle

    def _best_action(self, state):
        state = torch.tensor([state]).float().to(self.device)
        q_values = self.model(state)
        action = torch.argmax(q_values[0]).item()
        return action

    def choose_move(self, battle):
        state = self.embed_battle(battle)
        action = self._best_action(state)
        return player_action_to_move(self, action, battle)

