import copy
import asyncio
import numpy as np
import time

import matplotlib.pyplot as plt

from poke_env.data import to_id_str
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.environment.battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from rl_env import RLEnvPlayer
from dqn_agent import DQNAgent
from networking import custom_play_against, battle_against_wrapper, evaluate_model


def main():
    start = time.time()
    bf = "gen8ou"

    team_2_pokemon = """
Goodra (M) @ Assault Vest
Ability: Sap Sipper
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
IVs: 0 Atk
- Dragon Pulse
- Flamethrower
- Sludge Wave
- Thunderbolt

Sylveon (M) @ Leftovers
Ability: Pixilate
EVs: 248 HP / 244 Def / 16 SpD
Calm Nature
IVs: 0 Atk
- Hyper Voice
- Mystical Fire
- Protect
- Wish
"""

    team_1_pokemon = """
Goodra (M) @ Assault Vest
Ability: Sap Sipper
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
IVs: 0 Atk
- Dragon Pulse
- Flamethrower
- Sludge Wave
- Thunderbolt
"""

    # Initialize agent
    team_used = team_2_pokemon
    emb_dim = 250

    env_player = RLEnvPlayer(battle_format=bf, team=team_used)
    dqn = DQNAgent(emb_dim, len(env_player.action_space), battle_format=bf, team=team_used)
    dqn.set_embed_battle(env_player.embed_battle)

    # Initialize random player
    random_player = RandomPlayer(battle_format=bf, team=team_used)
    max_dmg_player = MaxBasePowerPlayer(battle_format=bf, team=team_used)
    heur_player = SimpleHeuristicsPlayer(battle_format=bf, team=team_used)

    num_burn_in = 20
    for i in range(num_burn_in):
        print(f'Burn in episode {i}')
        custom_play_against(
            env_player=env_player,
            env_algorithm=dqn.run_one_episode,
            opponent=heur_player,
        )
        custom_play_against(
            env_player=env_player,
            env_algorithm=dqn.run_one_episode,
            opponent=max_dmg_player,
        )

    num_episodes = 10
    training_per_episode = 100
    n_eval_battles = 20
    episodes = np.arange(1, num_episodes + 1)
    agent_wins_cum = 0
    agent_random_wins = np.zeros(num_episodes, dtype=int)
    agent_max_dmg_wins = np.zeros(num_episodes, dtype=int)
    agent_heur_wins = np.zeros(num_episodes, dtype=int)

    for i in range(num_episodes):
        print('\n\n-------------------------')
        print(f'Training episode {i}')

        # Train env_player
        for j in range(training_per_episode):
            custom_play_against(
                env_player=env_player,
                env_algorithm=dqn.train_one_episode,
                opponent=max_dmg_player,
            )
            custom_play_against(
                env_player=env_player,
                env_algorithm=dqn.train_one_episode,
                opponent=heur_player,
            )

        # Evaluate

        print('\nEvaluating against Random Player:')
        evaluate_model(
            player=dqn,
            opponent=random_player,
            n_battles=n_eval_battles
        )
        if i == 0:
            agent_random_wins[i] = dqn.n_won_battles
        else:
            agent_random_wins[i] = dqn.n_won_battles - agent_wins_cum
        agent_wins_cum = dqn.n_won_battles
        print(f'Wins: {agent_random_wins[i]} out of {n_eval_battles}')

        print('\nEvaluating against Max Damage Player:')
        evaluate_model(
            player=dqn,
            opponent=max_dmg_player,
            n_battles=n_eval_battles
        )
        agent_max_dmg_wins[i] = dqn.n_won_battles - agent_wins_cum
        agent_wins_cum = dqn.n_won_battles
        print(f'Wins: {agent_max_dmg_wins[i]} out of {n_eval_battles}')

        print('\nEvaluating against Heuristic Player:')
        evaluate_model(
            player=dqn,
            opponent=heur_player,
            n_battles=n_eval_battles
        )
        agent_heur_wins[i] = dqn.n_won_battles - agent_wins_cum
        agent_wins_cum = dqn.n_won_battles
        print(f'Wins: {agent_heur_wins[i]} out of {n_eval_battles}')

    print(agent_random_wins)
    print(agent_max_dmg_wins)
    print(agent_heur_wins)

    plt.figure()
    plt.plot(episodes, agent_random_wins, '-b', label="Agent Wins against Random")
    plt.plot(episodes, agent_max_dmg_wins, '-g', label="Agent Wins against Max Dmg")
    plt.plot(episodes, agent_heur_wins, '-r', label="Agent Wins against Heuristic")
    plt.xlabel("Episode")
    plt.ylabel("Number of Wins")
    plt.title("Agent Wins Per Episode")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()




