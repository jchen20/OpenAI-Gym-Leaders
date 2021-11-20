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
            # nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, len_action_space),
        )

    def forward(self, x, mask):
        x = self.model(x)
        x *= mask
        return x


class DQNAgent(Player):
    def __init__(self, state_size, action_space, eps=0.05, batch_size=32, gamma=0.99, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = DQN(state_size, action_space)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5,
                                          weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 0.995)
        self.memory = ReplayMemory(1000)
        self.eps = eps
        self.batch_size = batch_size
        self.gamma = gamma
        self.embed_battle = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    # double check this function // George: changed this to actually use the batch.
    def _train_one_step(self):
        batch = self.memory.sample(self.batch_size)
        state, mask, action, next_state, next_mask, rwd, terminal = zip(*batch)
        with torch.set_grad_enabled(True):
            self.optimizer.zero_grad()
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
            next_mask = torch.tensor(next_mask, dtype=torch.float32).to(self.device)
            rwd = torch.tensor(rwd, dtype=torch.float32).to(self.device)
            terminal = torch.tensor(terminal, dtype=bool).to(self.device)

            q_values = self.model(state, mask)
            q_values = q_values[range(len(q_values)), action]
            target_q_values = torch.clone(rwd).detach()

            valid_next_state = next_state[~terminal]
            valid_next_mask = next_mask[~terminal]
            valid_next_q_values = torch.max(self.model(valid_next_state, valid_next_mask), dim=1)[0]

            target_q_values[~terminal] += self.gamma * valid_next_q_values.detach()

            loss = torch.mean((q_values - target_q_values)**2)
            loss.backward()
            self.optimizer.step()

    def train_one_episode(self, env, no_train=False):
        env.reset_battles()
        done = False
        trained = False
        state, mask = env.reset()

        self.model.train()
        ct = 0
        while not done:
            # print(ct)
            ct += 1
            if random.random() < self.eps:
                action = random.choice(np.where(mask)[0])
            else:
                action = self._best_action(state, mask)
            (next_state, next_mask), rwd, done, _ = env.step(action)
            performed_action = env.last_action
            self.memory.push((state, mask, performed_action, next_state, next_mask, rwd, done))
            state = next_state
            mask = next_mask
            if not no_train and random.random() < (2 / self.batch_size):
                trained = True
                self._train_one_step()
        if not no_train and trained:
            self.scheduler.step()
        

    def set_embed_battle(self, embed_battle):
        self.embed_battle = embed_battle

    def _best_action(self, state, mask):
        state = torch.tensor([state]).float().to(self.device)
        mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
        q_values = self.model(state, mask)
        action = torch.argmax(q_values[0]).item()
        return action

    def choose_move(self, battle):
        self.model.eval()
        state, mask = self.embed_battle(battle)
        action = self._best_action(state, mask)
        return player_action_to_move(self, action, battle)

