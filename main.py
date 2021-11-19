import copy
import asyncio
import numpy as np
import time

import matplotlib.pyplot as plt

from poke_env.data import to_id_str
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer

from dqn_agent import DQNAgent
from networking import custom_play_against, battle_against_wrapper, evaluate_model

def one_hot(locations, size):
    vector = np.zeros(size)
    locations = np.array(locations) - 1
    vector[locations] = 1
    return vector

class RLEnvPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4, dtype=int)
        moves_dmg_multiplier = np.zeros(4)
        moves_accuracy = np.zeros(4)
        moves_categories = np.zeros(4, dtype=int)
        moves_status = np.zeros(4, dtype=int)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100 # Simple rescaling to facilitate learning
            moves_accuracy[i] = move.accuracy
            moves_categories[i] = move.category.value
            if move.status:
                moves_status[i] = move.status.value
            else:
                moves_status[i] = 0
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                )
    
        # We count how many pokemons have not fainted in each team
        remaining_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )
        
        team_types = [t.value for t in battle.active_pokemon.types if t is not None]
        team_one_hot_types = one_hot(team_types, 18)
    
        opponent_types = [t.value for t in battle.opponent_active_pokemon.types if t is not None]
        opponent_one_hot_types = one_hot(opponent_types, 18)
        
        moves_categories -= 1
        category_matrix = np.zeros((4, 3))
        category_matrix[np.arange(4), moves_categories] = 1
        
        status_matrix = np.zeros((4, 7))
        for i, status_type in enumerate(moves_status):
            if status_type != 0:
                status_matrix[i, status_type - 1] = 1
        
        team_stats = battle.active_pokemon.stats
        stats_array = np.zeros(5)
        stats_array[0] = team_stats['atk']
        stats_array[1] = team_stats['def']
        stats_array[2] = team_stats['spa']
        stats_array[3] = team_stats['spd']
        stats_array[4] = team_stats['spe']
        stats_array /= 100
        
        # Final vector with 10 components
        return np.concatenate(
            [moves_base_power,
             moves_dmg_multiplier,
             moves_accuracy,
             category_matrix.flatten(),
             status_matrix.flatten(),
             team_one_hot_types,
             [battle.active_pokemon.level],
             stats_array,
             opponent_one_hot_types,
             [battle.opponent_active_pokemon.level],
             [remaining_mon_team, remaining_mon_opponent]]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2,
            hp_value=1,
            victory_value=30,
        )


def main():
    start = time.time()
    bf = "gen8randombattle"

    # Initialize agent
    env_player = RLEnvPlayer(battle_format=bf)
    dqn = DQNAgent(97, len(env_player.action_space))
    dqn.set_embed_battle(env_player.embed_battle)

    # Initialize random player
    random_player = RandomPlayer(battle_format=bf)
    max_dmg_player = MaxBasePowerPlayer(battle_format=bf)
    heur_player = SimpleHeuristicsPlayer(battle_format=bf)

    num_burn_in = 10
    for i in range(num_burn_in):
        print(f'Burn in episode {i}')
        custom_play_against(
            env_player=env_player,
            env_algorithm=dqn.run_one_episode,
            opponent=heur_player,
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




