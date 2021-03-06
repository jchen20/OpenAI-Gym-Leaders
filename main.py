import copy
import asyncio
import numpy as np
import time

import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter, uniform_filter1d
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
from utils import set_random_seed
import teams


def main():
    method = 'a2c'

    set_random_seed(0)

    start = time.time()
    bf = "gen8ou"
    # bf = 'gen8randombattle'

    adversarial_train = True

    # Initialize agent
    team_used = teams.six_team
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

    num_episodes = 20
    training_per_episode = 50

    train_max_weight = 2
    train_heuristic_weight = 2
    train_self_weight = 0
    
    max_dmg_threshold_1 = 0.6
    max_dmg_threshold_2 = 0.8
    heuristic_threshold = 0.3

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
        print(f'max damage: {train_max_weight}')
        print(f'heuristic: {train_heuristic_weight}')
        print(f'self: {train_self_weight}')
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
                agent_max_dmg_rewards[i] += agent.episode_reward
                agent.episode_reward = 0
            for l in range(train_heuristic_weight):
                custom_play_against(
                    env_player=env_player,
                    env_algorithm=agent.train_one_episode,
                    opponent=heur_player,
                )
                agent_heur_rewards[i] += agent.episode_reward
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

        if agent_max_dmg_wins[i] / n_eval_battles > max_dmg_threshold_1:
            train_max_weight = 1
            train_heuristic_weight = 3
            if agent_max_dmg_wins[i] / n_eval_battles > max_dmg_threshold_2 and \
                agent_heur_wins[i] / n_eval_battles > heuristic_threshold:
                train_self_weight = 2
            else:
                train_self_weight = 1
        else:
            train_max_weight = 2
            train_heuristic_weight = 2
            train_self_weight = 0
        
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
    plt.plot(episodes, agent_random_wins, '-',
             label="Agent Wins against Random")
    plt.plot(episodes, agent_max_dmg_wins, '-',
             label="Agent Wins against Max Dmg")
    plt.plot(episodes, agent_heur_wins, '-',
             label="Agent Wins against Heuristic")
    plt.xlabel("Episode")
    plt.ylabel("Number of Wins")
    plt.title("Agent Wins Per Episode")
    plt.legend()
    plt.savefig('winrate.png', bbox_inches='tight')

    plt.figure()
    a_loss_smoothed = median_filter(agent.actor_losses, size=151, mode='nearest')
    c_loss_smoothed = median_filter(agent.critic_losses, size=151, mode='nearest')
    plt.plot(np.arange(agent.steps), a_loss_smoothed, '-g', linewidth=0.5, label='Actor Loss')
    plt.plot(np.arange(agent.steps), c_loss_smoothed, '-b', linewidth=0.5, label='Critic Loss')
    plt.yscale('symlog', linthresh=0.01)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Agent Losses Per Step')
    plt.legend()
    plt.savefig('loss.png', bbox_inches='tight')

    fig, ax1 = plt.subplots()
    ln1 = ax1.plot(agent.cum_train_steps, agent_heur_wins / n_eval_battles, '-r', \
        label='Agent Winrate against Heuristic')[0]
    ax1.set_xlabel('steps')
    ax1.set_ylabel('win rate')
    ax2 = ax1.twinx()
    moving_avg_meds = uniform_filter1d(agent.median_max_probs, 20)
    ln2 = ax2.plot(np.arange(agent.steps), moving_avg_meds, '-k', linewidth=0.3, \
        label='Moving Average of Median Max Probability Action Choice')[0]
    ax2.set_ylabel('max action probability')
    fig.tight_layout()
    ax1.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()])
    fig.suptitle('Winrate and Probability of Max Action')
    plt.subplots_adjust(top=0.92)
    plt.savefig('maxprob.png', bbox_inches='tight')

    plt.figure()
    plt.plot(episodes, agent_max_dmg_rewards, '-g',
             label="Agent Avg Reward against Max Dmg")
    plt.plot(episodes, agent_heur_rewards, '-r',
             label="Agent Avg Reward against Heuristic")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Agent Average Reward Per Episode")
    plt.legend()
    plt.savefig('reward.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()




