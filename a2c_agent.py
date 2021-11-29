from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from poke_env.player.player import Player, BattleOrder
from poke_env.player.battle_order import ForfeitBattleOrder

from utils import player_action_to_move


class A2C(nn.Module):
    def __init__(self, state_size, len_action_space):
        super().__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(state_size, 1024),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, len_action_space),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 1024),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def forward(self, x, mask):
        policy = self.actor(x)
        policy *= mask
        dist = Categorical(logits=F.log_softmax(policy, dim=1))
        values = self.critic(x)
        return dist, torch.squeeze(values)


class A2CAgent(Player):
    def __init__(self, state_size, action_space, eps=0.05, batch_size=32, gamma=0.99, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = A2C(state_size, action_space)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-5,
                                          weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 0.995)
        self.eps = eps
        self.batch_size = batch_size
        self.gamma = gamma
        self.embed_battle = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
    
    # double check this function // George: changed this to actually use the batch.
    def _train_one_step(self, batch):
        state, mask, action, next_state, next_mask, rwd, terminal = zip(*batch)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        next_mask = torch.tensor(next_mask, dtype=torch.float32).to(self.device)
        rwd = torch.tensor(rwd, dtype=torch.float32).to(self.device)
        terminal = torch.tensor(terminal, dtype=bool).to(self.device)

        _, next_values = self.model(next_state, next_mask) # can probably find a way to not make this redundant computation
        with torch.set_grad_enabled(True):
            self.optimizer.zero_grad()
            dist, values = self.model(state, mask)
    
            # following this: https://github.com/lcswillems/torch-ac/blob/master/torch_ac/algos/a2c.py
            # and this: https://github.com/lcswillems/torch-ac/blob/master/torch_ac/algos/base.py
            advantages = torch.zeros(len(batch), device=self.device)
            for i in reversed(range(len(batch))):
                next_state_mask_i = ~terminal[i]
                next_value_i = next_values[i]
                next_advantage = advantages[i + 1] if i < len(batch) - 1 else 0
        
                delta = rwd[i] + self.gamma * next_value_i * next_state_mask_i - values[i]
                advantages[i] = delta + self.gamma * next_advantage * next_state_mask_i
            
            actor_loss = -(dist.log_prob(action) * advantages.detach()).mean()
            critic_loss = ((advantages + values).detach() - values).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss
    
            loss.backward()
            self.optimizer.step()
    
    def train_one_episode(self, env, no_train=False):
        env.reset_battles()
        done = False
        trained = False
        state, mask = env.reset()
        one_batch = deque([])
        
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
            one_batch.append((state, mask, performed_action, next_state, next_mask, rwd, done))
            state = next_state
            mask = next_mask
            if not no_train and (len(one_batch) >= self.batch_size or done):
                trained = True
                self._train_one_step(list(one_batch))
                one_batch.clear()
        if not no_train and trained:
            self.scheduler.step()
    
    def set_embed_battle(self, embed_battle):
        self.embed_battle = embed_battle
    
    def _best_action(self, state, mask):
        state = torch.tensor([state]).float().to(self.device)
        mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
        dist, _ = self.model(state, mask)
        action = dist.sample().item()
        return action
    
    def choose_move(self, battle):
        self.model.eval()
        state, mask = self.embed_battle(battle)
        action = self._best_action(state, mask)
        return player_action_to_move(self, action, battle)

