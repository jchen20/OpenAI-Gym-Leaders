from collections import deque
import random
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
import numpy as np
from poke_env.player.player import Player, BattleOrder
from poke_env.player.battle_order import ForfeitBattleOrder

from utils import player_action_to_move, one_hot


class A2C(nn.Module):
    def __init__(self, state_size, len_action_space):
        super().__init__()

        # self.base = nn.Sequential(
        #     nn.Linear(state_size, 1024),
        #     nn.LeakyReLU(),
        #     nn.Linear(1024, 512),
        #     nn.LeakyReLU(),
        # )

        self.actor = nn.Sequential(
            nn.Linear(state_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            # self.base,
            # nn.Linear(512, 128),
            # nn.LeakyReLU(),
            nn.Linear(512, len_action_space)
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            # self.base,
            # nn.Linear(512, 128),
            # nn.LeakyReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x, mask):
        policy = self.actor(x)
        policy *= mask
        dist = Categorical(logits=policy)
        values = self.critic(x)
        return dist, torch.squeeze(values)
    
    def actor_forward(self, x, mask):
        policy = self.actor(x)
        policy *= mask
        dist = Categorical(logits=policy)
        return dist
    
    def critic_forward(self, x):
        values = self.critic(x)
        return torch.squeeze(values)


class A2CMove(nn.Module):
    def __init__(self, state_size, len_action_space):
        super().__init__()
        self.actor_move = nn.Sequential(
            nn.Linear(14, 56),
            nn.LeakyReLU(),
            nn.Linear(56, 56),
        )
        
        self.actor_1 = nn.Sequential(
            nn.Linear(state_size - 56, 1024 - 56*4),
            nn.LeakyReLU(),
        )
        self.actor_2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, len_action_space)
        )
        
        self.critic_move = nn.Sequential(
            nn.Linear(14, 56),
            nn.LeakyReLU(),
            nn.Linear(56, 56),
        )
        
        self.critic_1 = nn.Sequential(
            nn.Linear(state_size - 56, 1024 - 56*4),
            nn.LeakyReLU(),
        )
        self.critic_2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x, mask):
        policy_move = self.actor_move(x[:, :56].reshape((-1, 4, 14))).flatten(start_dim=1)
        policy_pre = self.actor_1(x[:, 56:])
        policy = self.actor_2(torch.cat((policy_move, policy_pre), dim=-1))
        policy *= mask
        dist = Categorical(logits=policy)

        values_move = self.critic_move(x[:, :56].reshape((-1, 4, 14))).flatten(start_dim=1)
        values_pre = self.critic_1(x[:, 56:])
        values = self.critic((torch.cat((values_move, values_pre), dim=-1)))
        return dist, torch.squeeze(values)
    
    def actor_forward(self, x, mask):
        policy_move = self.actor_move(x[:, :56].reshape((-1, 4, 14))).flatten(start_dim=1)
        policy_pre = self.actor_1(x[:, 56:])
        policy = self.actor_2(torch.cat((policy_move, policy_pre), dim=-1))
        policy *= mask
        dist = Categorical(logits=policy)
        return dist
    
    def critic_forward(self, x):
        values_move = self.critic_move(x[:, :56].reshape((-1, 4, 14))).flatten(start_dim=1)
        values_pre = self.critic_1(x[:, 56:])
        values = self.critic_2((torch.cat((values_move, values_pre), dim=-1)))
        return torch.squeeze(values)


class A2CAgentFullTrajectoryUpdate(Player):
    def __init__(self, state_size, action_space, batch_size=8, gamma=0.99, gae_lambda=0.3, model=None,
                 move_encoder=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if model:
            self.model = model
        else:
            if move_encoder:
                self.model = A2CMove(state_size + action_space, action_space)
            else:
                self.model = A2C(state_size + action_space, action_space)
        self.state_size = state_size
        self.action_space = action_space
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 0.999)
        self.batch_size = batch_size # batch size is max horizon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps = 0.1
        self.entropy_beta = 0.03 / np.log(action_space)
        self.alpha = 10
        self.embed_battle = None
        self.episode_reward = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.lambda_scale = torch.tensor([self.gamma**i for i in range(self.batch_size)][::-1], device=self.device)
        self.gamma_scale = torch.tensor([self.gae_lambda**i for i in range(self.batch_size)][::-1], device=self.device)
        self.lambda_scale = self.lambda_scale.float()
        self.gamma_scale = self.gamma_scale.float()
        self.last_action = None

        self.median_max_probs = []
        self.actor_losses = []
        self.critic_losses = []
        self.steps = 0
        self.cum_train_steps = []

        self.force_non_greedy = False

        self.model.to(self.device)
    
    def _train_one_step(self, batch):
        state, mask, action, next_state, _, rwd, terminal = zip(*batch)
        # discounted_rwd = discount(rwd)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        rwd = torch.tensor(rwd, dtype=torch.float32).to(self.device)
        terminal = torch.tensor(terminal, dtype=bool).to(self.device)

        prev_actions_state = torch.cat((torch.zeros(1, self.action_space, device=self.device), F.one_hot(action[:-1], self.action_space).float()), dim=0)
        prev_actions_next_state = F.one_hot(action, self.action_space).float()
        state = torch.cat((state, prev_actions_state), dim=1)
        next_state = torch.cat((next_state, prev_actions_next_state), dim=1)
        
        with torch.set_grad_enabled(True):
            self.optimizer.zero_grad()
            dist = self.model.actor_forward(state, mask)
            td_lambda_err = self.td_lambda_err(state, action, next_state, rwd, terminal)
            log_probs = dist.log_prob(action)
            actor_losses = -log_probs * td_lambda_err.detach()
            mask_dist = Categorical((mask + 1e-10) / torch.sum(mask, dim=1, keepdim=True))
            ent_scale = self.entropy_beta * kl_divergence(dist, mask_dist) * (1 + 5 * F.relu(-td_lambda_err.detach()))
            actor_loss = (actor_losses + ent_scale).mean()
            critic_loss = td_lambda_err.pow(2).mean()
            loss = actor_loss + self.alpha * critic_loss

            med_max_prob = torch.median(torch.max(dist.probs, dim=1)[0]).item()
            self.median_max_probs.append(med_max_prob)
            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())
            self.steps += 1
            if random.random() < 0.01:
                print(f'{actor_losses.mean().item():.2E}', f'{critic_loss.item():.2E}', f'{med_max_prob:.2E}')
                print(f'{td_lambda_err.mean().item():.2E}, {ent_scale.mean().item():.2E}')
            
            loss.backward()
            self.optimizer.step()
    
    def train_one_episode(self, env, no_train=False):
        env.reset_battles()
        done = False
        state, mask = env.reset()
        one_batch = deque([], maxlen=self.batch_size)
        train_prob = 0.1
        self.last_action = None
        
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
            self.episode_reward = rwd
            performed_action = env.last_action
            self.last_action = performed_action
            one_batch.append((state, mask, performed_action, next_state, next_mask, rwd, done))
            state = next_state
            mask = next_mask
            if ct > 1 and not no_train and random.random() < train_prob:
                self._train_one_step(list(one_batch))
        if not no_train:
            self._train_one_step(list(one_batch))
            self.scheduler.step()


    def td_lambda_err(self, state, action, next_state, reward, terminal):
        # n = len(state)
        # lambda_scale = self.lambda_scale[:n]
        # gamma_scale = self.gamma_scale[:n]
        # # prev_values = self.model.critic_forward(state)[range(n), action]
        # prev_values = self.model.critic_forward(state)
        # next_values = reward
        # # next_values[~terminal] += self.gamma * torch.max(self.model.critic_forward(next_state[~terminal]), dim=1)[0]
        # next_values[~terminal] += self.gamma * self.model.critic_forward(next_state[~terminal])
        # td_error = next_values - prev_values
        # td_error = td_error * lambda_scale * gamma_scale
        # return td_error

        n = len(state)
        lambda_scale = torch.repeat_interleave(self.lambda_scale[:n][None, :], n, dim=0)
        gamma_scale = torch.repeat_interleave(self.gamma_scale[:n][None, :], n, dim=0)
        # prev_values = self.model.critic_forward(state)[range(n), action]
        prev_values = self.model.critic_forward(state)
        next_values = reward
        # next_values[~terminal] += self.gamma * torch.max(self.model.critic_forward(next_state[~terminal]), dim=1)[0]
        next_values[~terminal] += self.gamma * self.model.critic_forward(next_state[~terminal])
        td_error = next_values.detach() - prev_values
        td_error_mat = torch.zeros((n, n), device=self.device)
        for i in range(n):
            td_error_mat[i, i:(n-1)] = td_error[0:(n-i-1)].detach()
            td_error_mat[i, n-1] = td_error[n-i-1]
        td_error_mat = td_error_mat * lambda_scale * gamma_scale
        td_error_vec = torch.sum(td_error_mat, dim=-1)
        return td_error_vec

    
    def set_embed_battle(self, embed_battle):
        self.embed_battle = embed_battle
    
    def _best_action(self, state, mask, greedy=False):
        with torch.no_grad():
            if self.last_action is None:
                state = np.append(state, np.zeros(self.action_space))
            else:
                state = np.append(state, one_hot(self.last_action + 1, self.action_space))
            state = torch.tensor([state]).float().to(self.device)
            mask = torch.tensor(mask, dtype=torch.float32).to(self.device)
            dist = self.model.actor_forward(state, mask)
            if greedy:
                action = torch.argmax(dist.probs, dim=1)[0].item()
            else:
                action = dist.sample().item()
            self.last_action = action
            return action
    
    def choose_move(self, battle):
        self.model.eval()
        state, mask = self.embed_battle(battle)
        action = self._best_action(state, mask, greedy=(not self.force_non_greedy))
        return player_action_to_move(self, action, battle)
