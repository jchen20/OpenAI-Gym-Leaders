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
from a2c_agent import A2CAgentFullTrajectoryUpdate
from networking import custom_play_against, battle_against_wrapper, \
    evaluate_model, custom_train_agents
from utils import set_random_seed
import teams


def main():
    method = 'a2c'

    set_random_seed(0)

    start = time.time()
    bf = "gen8ou"
    # bf = 'gen8randombattle'

    adversarial_train = False

    # Initialize agent
    team_used = teams.wall_six_team
    emb_dim = 371

    env_player = RLEnvPlayer(battle_format=bf, team=team_used)
    if method == 'dqn':
        agent = DQNAgent(emb_dim, len(env_player.action_space) - 8,
                         battle_format=bf, team=team_used)
    else:
        agent = A2CAgentFullTrajectoryUpdate(emb_dim,
                                             len(env_player.action_space) - 8,
                                             battle_format=bf, team=team_used)
    agent.set_embed_battle(env_player.embed_battle)

    if adversarial_train:
        env_player2 = RLEnvPlayer(battle_format=bf, team=team_used)
        if method == 'dqn':
            agent2 = DQNAgent(emb_dim, len(env_player.action_space) - 8,
                              battle_format=bf, team=team_used)
        else:
            agent2 = A2CAgentFullTrajectoryUpdate(emb_dim, len(
                env_player.action_space) - 8, battle_format=bf, team=team_used)
        agent2.set_embed_battle(env_player2.embed_battle)

    # Initialize random player
    random_player = RandomPlayer(battle_format=bf, team=team_used)
    max_dmg_player = MaxBasePowerPlayer(battle_format=bf, team=team_used)
    heur_player = SimpleHeuristicsPlayer(battle_format=bf, team=team_used)

    num_burn_in = 20 if method == 'dqn' else 0
    run_one_episode = lambda x: agent.train_one_episode(x, no_train=True)
    for i in range(num_burn_in):
        print(f'Burn in episode {i}')
        custom_play_against(
            env_player=env_player,
            env_algorithm=run_one_episode,
            opponent=heur_player,
        )
        custom_play_against(
            env_player=env_player,
            env_algorithm=run_one_episode,
            opponent=max_dmg_player,
        )
    if adversarial_train:
        for i in range(num_burn_in):
            print(f'Burn in episode {i}')
            custom_play_against(
                env_player=env_player2,
                env_algorithm=run_one_episode,
                opponent=heur_player,
            )
            custom_play_against(
                env_player=env_player2,
                env_algorithm=run_one_episode,
                opponent=max_dmg_player,
            )

    num_episodes = 10
    training_per_episode = 100

    train_max_weight = 1
    train_heuristic_weight = 3
    train_self_weight = 2

    n_eval_battles = 50
    episodes = np.arange(1, num_episodes + 1)
    trainings = np.arange(1, num_episodes*training_per_episode + 1)
    agent_wins_cum = 0
    agent_random_wins = np.zeros(num_episodes, dtype=int)
    agent_max_dmg_wins = np.zeros(num_episodes, dtype=int)
    agent_heur_wins = np.zeros(num_episodes, dtype=int)

    agent_random_rewards = np.zeros(num_episodes*training_per_episode, dtype=float)
    agent_max_dmg_rewards = np.zeros(num_episodes*training_per_episode, dtype=float)
    agent_heur_rewards = np.zeros(num_episodes*training_per_episode, dtype=float)

    agent2_wins_cum = 0
    agent2_random_wins = np.zeros(num_episodes, dtype=int)
    agent2_max_dmg_wins = np.zeros(num_episodes, dtype=int)
    agent2_heur_wins = np.zeros(num_episodes, dtype=int)

    for i in range(num_episodes):
        print('\n\n-------------------------')
        print(f'Training episode {i}')
        if adversarial_train:
            agent2.model = copy.deepcopy(agent.model)
        # Train env_player
        for j in range(training_per_episode):
            for k in range(train_max_weight):
                custom_play_against(
                    env_player=env_player,
                    env_algorithm=agent.train_one_episode,
                    opponent=max_dmg_player,
                )
                if k + 1 == train_max_weight:
                    agent_max_dmg_rewards[i*training_per_episode + j] = agent.episode_reward
                agent.episode_reward = 0
            for l in range(train_heuristic_weight):
                custom_play_against(
                    env_player=env_player,
                    env_algorithm=agent.train_one_episode,
                    opponent=heur_player,
                )
                if l + 1 == train_heuristic_weight:
                    agent_heur_rewards[i*training_per_episode + j] = agent.episode_reward
                agent.episode_reward = 0
            if adversarial_train and i != 0:
                for m in range(train_self_weight):
                    custom_play_against(
                        env_player=env_player,
                        env_algorithm=agent.train_one_episode,
                        opponent=agent2
                    )

        # Evaluate
        print('\nAgent 1:')
        print('\nEvaluating against Random Player:')
        evaluate_model(
            player=agent,
            opponent=random_player,
            n_battles=n_eval_battles
        )
        if i == 0:
            agent_random_wins[i] = agent.n_won_battles
        else:
            agent_random_wins[i] = agent.n_won_battles - agent_wins_cum
        agent_wins_cum = agent.n_won_battles
        print(f'Wins: {agent_random_wins[i]} out of {n_eval_battles}')

        print('\nEvaluating against Max Damage Player:')
        evaluate_model(
            player=agent,
            opponent=max_dmg_player,
            n_battles=n_eval_battles
        )
        agent_max_dmg_wins[i] = agent.n_won_battles - agent_wins_cum
        agent_wins_cum = agent.n_won_battles
        print(f'Wins: {agent_max_dmg_wins[i]} out of {n_eval_battles}')

        print('\nEvaluating against Heuristic Player:')
        evaluate_model(
            player=agent,
            opponent=heur_player,
            n_battles=n_eval_battles
        )
        agent_heur_wins[i] = agent.n_won_battles - agent_wins_cum
        agent_wins_cum = agent.n_won_battles
        print(f'Wins: {agent_heur_wins[i]} out of {n_eval_battles}')
        
        # if adversarial_train:
        #     # Evaluate
        #     print('\nAgent 2:')
        #     print('\nEvaluating against Random Player:')
        #     evaluate_model(
        #         player=agent2,
        #         opponent=random_player,
        #         n_battles=n_eval_battles
        #     )
        #     if i == 0:
        #         agent2_random_wins[i] = agent2.n_won_battles
        #     else:
        #         agent2_random_wins[i] = agent2.n_won_battles - agent2_wins_cum
        #     agent2_wins_cum = agent2.n_won_battles
        #     print(f'Wins: {agent2_random_wins[i]} out of {n_eval_battles}')
        #
        #     print('\nEvaluating against Max Damage Player:')
        #     evaluate_model(
        #         player=agent2,
        #         opponent=max_dmg_player,
        #         n_battles=n_eval_battles
        #     )
        #     agent2_max_dmg_wins[i] = agent2.n_won_battles - agent2_wins_cum
        #     agent2_wins_cum = agent2.n_won_battles
        #     print(f'Wins: {agent2_max_dmg_wins[i]} out of {n_eval_battles}')
        #
        #     print('\nEvaluating against Heuristic Player:')
        #     evaluate_model(
        #         player=agent2,
        #         opponent=heur_player,
        #         n_battles=n_eval_battles
        #     )
        #     agent2_heur_wins[i] = agent2.n_won_battles - agent2_wins_cum
        #     agent2_wins_cum = agent2.n_won_battles
        #     print(f'Wins: {agent2_heur_wins[i]} out of {n_eval_battles}')

    print(agent_random_wins)
    print(agent_max_dmg_wins)
    print(agent_heur_wins)

    plt.figure()
    plt.plot(episodes, agent_random_wins, '-b',
             label="Agent Wins against Random")
    plt.plot(episodes, agent_max_dmg_wins, '-g',
             label="Agent Wins against Max Dmg")
    plt.plot(episodes, agent_heur_wins, '-r',
             label="Agent Wins against Heuristic")
    plt.xlabel("Episode")
    plt.ylabel("Number of Wins")
    plt.title("Agent Wins Per Episode")
    plt.legend()
    plt.show()

    plt.figure()
    plt.semilogy(trainings, agent_max_dmg_rewards, '-g',
             label="Agent Reward against Max Dmg")
    plt.semilogy(trainings, agent_heur_rewards, '-r',
             label="Agent Reward against Heuristic")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Agent Reward Per Episode")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()




