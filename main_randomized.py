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
from a2cq_agent import A2CQAgentFullTrajectoryUpdate
from td3_agent import TD3AgentFullTrajectoryUpdate
from td3v_agent import TD3VAgentFullTrajectoryUpdate
from networking import custom_play_against, battle_against_wrapper, \
    evaluate_model, custom_train_agents
from utils import set_random_seed, RandomTeamFromPool, random_team
import teams


def main():
    method = 'td3'

    set_random_seed(0)

    start = time.time()
    bf = "gen8ou"
    # bf = 'gen8randombattle'

    adversarial_train = False
    num_pokemon_in_team = 6
    # Initialize agent
    team_used = RandomTeamFromPool(teams.random_pokemon_list,num_pokemon_in_team, reset_team_cycle=10)
    #team_used = teams.four_team
    #team_used = random_team(teams.random_pokemon_list,num_pokemon_in_team)
    emb_dim = 329

    move_encoder = False
    env_player = RLEnvPlayer(battle_format=bf, team=team_used)
    if method == 'dqn':
        agent = DQNAgent(emb_dim, len(env_player.action_space) - 12,
                         battle_format=bf, team=team_used)
    elif method == 'a2c':
        agent = A2CAgentFullTrajectoryUpdate(emb_dim,
                                             len(env_player.action_space) - 12,
                                             move_encoder=move_encoder,
                                             battle_format=bf, team=team_used)
    elif method == 'a2cq':
        agent = A2CQAgentFullTrajectoryUpdate(emb_dim,
                                              len(env_player.action_space) - 12,
                                              battle_format=bf, team=team_used)
    elif method == 'td3':
        agent = TD3AgentFullTrajectoryUpdate(emb_dim,
                                             len(env_player.action_space) - 12,
                                             battle_format=bf, team=team_used)
    elif method == 'td3v':
        agent = TD3VAgentFullTrajectoryUpdate(emb_dim,
                                              len(env_player.action_space) - 12,
                                              battle_format=bf, team=team_used)
    agent.set_embed_battle(env_player.embed_battle)

    if adversarial_train:
        env_player2 = RLEnvPlayer(battle_format=bf, team=team_used)
        if method == 'dqn':
            agent2 = DQNAgent(emb_dim, len(env_player.action_space) - 12,
                              battle_format=bf, team=team_used)
        elif method == 'a2c':
            agent2 = A2CAgentFullTrajectoryUpdate(emb_dim,
                                                  len(env_player.action_space) - 12,
                                                  move_encoder=move_encoder,
                                                  battle_format=bf, team=team_used)
        elif method == 'a2cq':
            agent2 = A2CQAgentFullTrajectoryUpdate(emb_dim,
                                                   len(env_player.action_space) - 12,
                                                   battle_format=bf, team=team_used)
        elif method == 'td3':
            agent2 = TD3AgentFullTrajectoryUpdate(emb_dim,
                                                  len(env_player.action_space) - 12,
                                                  battle_format=bf, team=team_used)
        elif method == 'td3v':
            agent2 = TD3VAgentFullTrajectoryUpdate(emb_dim,
                                                   len(env_player.action_space) - 12,
                                                   battle_format=bf, team=team_used)
        agent2.set_embed_battle(env_player2.embed_battle)

    # Initialize random player
    opponent_team = RandomTeamFromPool(teams.random_pokemon_list, num_pokemon_in_team, reset_team_cycle=10)
    random_player = RandomPlayer(battle_format=bf, team=opponent_team)
    max_dmg_player = MaxBasePowerPlayer(battle_format=bf, team=opponent_team)
    heur_player = SimpleHeuristicsPlayer(battle_format=bf, team=opponent_team)

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
    # if adversarial_train:
    #     for i in range(num_burn_in):
    #         print(f'Burn in episode {i}')
    #         custom_play_against(
    #             env_player=env_player2,
    #             env_algorithm=run_one_episode,
    #             opponent=heur_player,
    #         )
    #         custom_play_against(
    #             env_player=env_player2,
    #             env_algorithm=run_one_episode,
    #             opponent=max_dmg_player,
    #         )

    num_episodes = 10
    training_per_episode = 100

    train_max_weight = 1
    train_heuristic_weight = 3
    train_self_weight = 2

    n_eval_battles = 50
    episodes = np.arange(1, num_episodes + 1)
    trainings = np.arange(1, num_episodes*training_per_episode + 1)

    agent_random_wins = np.zeros(num_episodes, dtype=int)
    agent_max_dmg_wins = np.zeros(num_episodes, dtype=int)
    agent_heur_wins = np.zeros(num_episodes, dtype=int)

    agent_max_dmg_rewards = np.zeros(num_episodes, dtype=float)
    agent_heur_rewards = np.zeros(num_episodes, dtype=float)

    agent2_random_wins = np.zeros(num_episodes, dtype=int)
    agent2_max_dmg_wins = np.zeros(num_episodes, dtype=int)
    agent2_heur_wins = np.zeros(num_episodes, dtype=int)

    for i in range(num_episodes):
        print('\n\n-------------------------')
        print(f'Training episode {i}')
        if adversarial_train:
            agent2.model = copy.deepcopy(agent.model)
            agent2.force_non_greedy = True
        # Train env_player
        for j in range(training_per_episode):
            for k in range(train_max_weight):
                custom_play_against(
                    env_player=env_player,
                    env_algorithm=agent.train_one_episode,
                    opponent=max_dmg_player,
                )
                agent_max_dmg_rewards[i] = agent.episode_reward
                agent.episode_reward = 0
            for l in range(train_heuristic_weight):
                custom_play_against(
                    env_player=env_player,
                    env_algorithm=agent.train_one_episode,
                    opponent=heur_player,
                )
                agent_heur_rewards[i] = agent.episode_reward
                agent.episode_reward = 0
            if adversarial_train:
                for m in range(train_self_weight):
                    custom_play_against(
                        env_player=env_player,
                        env_algorithm=agent.train_one_episode,
                        opponent=agent2
                    )
        
        agent_max_dmg_rewards[i] /= (training_per_episode * train_max_weight)
        agent_heur_rewards[i] /= (training_per_episode * train_heuristic_weight)
        agent.cum_train_steps.append(agent.steps)

        # Evaluate
        print('\nAgent 1:')
        print('\nEvaluating against Random Player:')
        evaluate_model(
            player=agent,
            opponent=random_player,
            n_battles=n_eval_battles
        )
        agent_random_wins[i] = agent.n_won_battles
        print(f'Wins: {agent_random_wins[i]} out of {n_eval_battles}')
        agent.reset_battles()

        print('\nEvaluating against Max Damage Player:')
        evaluate_model(
            player=agent,
            opponent=max_dmg_player,
            n_battles=n_eval_battles
        )
        agent_max_dmg_wins[i] = agent.n_won_battles
        print(f'Wins: {agent_max_dmg_wins[i]} out of {n_eval_battles}')
        agent.reset_battles()

        print('\nEvaluating against Heuristic Player:')
        evaluate_model(
            player=agent,
            opponent=heur_player,
            n_battles=n_eval_battles
        )
        agent_heur_wins[i] = agent.n_won_battles
        print(f'Wins: {agent_heur_wins[i]} out of {n_eval_battles}')
        agent.reset_battles()
        
        # if adversarial_train:
        #     # Evaluate
        #     print('\nAgent 2:')
        #     print('\nEvaluating against Random Player:')
        #     evaluate_model(
        #         player=agent2,
        #         opponent=random_player,
        #         n_battles=n_eval_battles
        #     )
        #     agent2_random_wins[i] = agent2.n_won_battles
        #     print(f'Wins: {agent2_random_wins[i]} out of {n_eval_battles}')
        #     agent2.reset_battles()
        
        #     print('\nEvaluating against Max Damage Player:')
        #     evaluate_model(
        #         player=agent2,
        #         opponent=max_dmg_player,
        #         n_battles=n_eval_battles
        #     )
        #     agent2_max_dmg_wins[i] = agent2.n_won_battles
        #     print(f'Wins: {agent2_max_dmg_wins[i]} out of {n_eval_battles}')
        #     agent2.reset_battles()
        
        #     print('\nEvaluating against Heuristic Player:')
        #     evaluate_model(
        #         player=agent2,
        #         opponent=heur_player,
        #         n_battles=n_eval_battles
        #     )
        #     agent2_heur_wins[i] = agent2.n_won_battles
        #     print(f'Wins: {agent2_heur_wins[i]} out of {n_eval_battles}')
        #     agent2.reset_battles()

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
    plt.semilogy(np.arange(agent.steps), agent.actor_losses, '-g', linewidth=0.1, label='Actor Loss')
    plt.semilogy(np.arange(agent.steps), agent.critic_losses, '-b', linewidth=0.1, label='Critic Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Agent Losses Per Step')
    plt.legend()

    fig, ax1 = plt.subplots()
    ax1.plot(agent.cum_train_steps, agent_heur_wins / n_eval_battles, '-r', label='Agent Winrate against Heuristic')
    ax1.set_xlabel('steps')
    ax1.set_ylabel('win rate')
    ax2 = ax1.twinx()
    ax2.plot(np.arange(agent.steps), agent.median_max_probs, '-k', linewidth=0.1, label='Median Max Probability Action Choice')
    ax2.set_ylabel('max action probability')
    fig.tight_layout()

    plt.figure()
    plt.plot(episodes, agent_max_dmg_rewards, '-g',
             label="Agent Reward against Max Dmg")
    plt.plot(episodes, agent_heur_rewards, '-r',
             label="Agent Reward against Heuristic")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Agent Average Reward Per Episode")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()




