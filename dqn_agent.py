from collections import deque
import random
import torch
import torch.nn as nn
from poke_env.player.player import Player, BattleOrder
from poke_env.player.battle_order import ForfeitBattleOrder


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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001,
                                          weight_decay=1e-4)
        self.memory = ReplayMemory(10000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.embed_battle = None
        self.episode_reward = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

    # double check this function
    def _train_one_step(self):
        # batch = self.memory.sample(self.batch_size)
        # for state, action, next_state, rwd, terminal in batch:
        #     self.optimizer.zero_grad()
        #     with torch.set_grad_enabled(True):
        #         state = torch.tensor([state]).float().to(self.device)
        #         q_values = self.model(state)
        #         target_q_values = q_values
        #         # double check this
        #         if terminal:
        #             target_q_values[0][action] = rwd
        #         else:
        #             target_q_values[0][action] = rwd + self.gamma * torch.max(torch.tensor([next_state]).float().to(self.device)).numpy()
        #         loss = torch.sum((q_values - target_q_values) ** 2)
        #         loss.backward()
        #         self.optimizer.step()
        batch = self.memory.sample(self.batch_size)
        for state, action, next_state, rwd, terminal in batch:
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                state = torch.tensor([state]).float().to(self.device)
                q_values = self.model(state)
                target_q_values = q_values.detach() # copies
                # double check this // George: this should be correct now
                if terminal:
                    target_q_values[0, action] = rwd
                else:
                    next_state = torch.tensor([next_state]).float().to(self.device)
                    target_q_values[0, action] = rwd + self.gamma * torch.max(self.model(next_state)).detach()
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
<<<<<<< Updated upstream
            action = self._best_action(state)
            next_state, rwd, done, _ = env.step(action)
            self.memory.push((state, action, next_state, rwd, done))
            self._train_one_step()
=======
            if random.random() < self.eps:
                action = random.choice(np.where(mask)[0])
            else:
                action = self._best_action(state, mask)
            (next_state, next_mask), rwd, done, _ = env.step(action)
            self.episode_reward += rwd
            performed_action = env.last_action
            self.memory.push((state, mask, performed_action, next_state, next_mask, rwd, done))
>>>>>>> Stashed changes
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

    def get_move_array(self, battle):
        available_orders = [BattleOrder(move) for move in
                            battle.available_moves]
        available_orders.extend(
            [BattleOrder(switch) for switch in battle.available_switches]
        )
        if battle.can_mega_evolve:
            available_orders.extend(
                [BattleOrder(move, mega=True) for move in
                 battle.available_moves]
            )
        if battle.can_dynamax:
            available_orders.extend(
                [BattleOrder(move, dynamax=True) for move in
                 battle.available_moves]
            )
        if battle.can_z_move and battle.active_pokemon:
            available_z_moves = set(battle.active_pokemon.available_z_moves)
            available_orders.extend(
                [
                    BattleOrder(move, z_move=True)
                    for move in battle.available_moves
                    if move in available_z_moves
                ]
            )
        return available_orders

    def _action_to_move(self, action, battle):
        """Converts actions to move orders.
        The conversion is done as follows:
        action = -1:
            The battle will be forfeited.
        0 <= action < 4:
            The actionth available move in battle.available_moves is executed.
        4 <= action < 8:
            The action - 4th available move in battle.available_moves is executed, with
            z-move.
        8 <= action < 12:
            The action - 8th available move in battle.available_moves is executed, with
            mega-evolution.
        8 <= action < 12:
            The action - 8th available move in battle.available_moves is executed, with
            mega-evolution.
        12 <= action < 16:
            The action - 12th available move in battle.available_moves is executed,
            while dynamaxing.
        16 <= action < 22
            The action - 16th available switch in battle.available_switches is executed.
        If the proposed action is illegal, a random legal move is performed.
        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        if action == -1:
            return ForfeitBattleOrder()
        elif (
                action < 4
                and action < len(battle.available_moves)
                and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action])
        elif (
                not battle.force_switch
                and battle.can_z_move
                and battle.active_pokemon
                and 0
                <= action - 4
                < len(battle.active_pokemon.available_z_moves)  # pyre-ignore
        ):
            return self.create_order(
                battle.active_pokemon.available_z_moves[action - 4], z_move=True
            )
        elif (
                battle.can_mega_evolve
                and 0 <= action - 8 < len(battle.available_moves)
                and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action - 8],
                                     mega=True)
        elif (
                battle.can_dynamax
                and 0 <= action - 12 < len(battle.available_moves)
                and not battle.force_switch
        ):
            return self.create_order(battle.available_moves[action - 12],
                                     dynamax=True)
        elif 0 <= action - 16 < len(battle.available_switches):
            return self.create_order(battle.available_switches[action - 16])
        else:
            return self.choose_random_move(battle)

    def choose_move(self, battle):
        state = self.embed_battle(battle)
        action = self._best_action(state)
        # move_arr = self.get_move_array(battle)
        # avail = battle.available_moves
        return self._action_to_move(action - 1, battle)

