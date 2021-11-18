import copy
import asyncio
import numpy as np
import time

import matplotlib.pyplot as plt

from poke_env.data import to_id_str
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer

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

# async def battle_against_wrapper(player, opponent, n_battles):
#     await player.battle_against(opponent, n_battles)

# async def train_wrapper(env, player, opponent):
#     await asyncio.gather(
#         player.send_challenges(opponent=to_id_str(opponent.username), n_challenges=1, to_wait=opponent.logged_in),
#         opponent.accept_challenges(opponent=to_id_str(player.username), n_challenges=1),
#         player.train_one_episode(env)
#     )

# async def train_and_evaluate_wrapper(env, player, opponent, n_eval_battles):
#     await env.play_against(
#         env_algorithm=player.train_one_episode,
#         opponent=opponent
#     )
#     await player.battle_against(opponent, n_eval_battles)

def main():
    start = time.time()
    bf = "gen8randombattle"

    # Initialize agent
    env_player = RLEnvPlayer(battle_format=bf)
    dqn = DQNAgent(97, len(env_player.action_space))
    dqn.set_embed_battle(env_player.embed_battle)

    # Initialize random player
    random_player = RandomPlayer(battle_format=bf)

    num_episodes = 10
    episodes = np.arange(1, num_episodes + 1)
    agent_games_cum = np.zeros(num_episodes)
    agent_wins_cum = np.zeros(num_episodes)
    agent_games = np.zeros(num_episodes)
    agent_wins = np.zeros(num_episodes)

    for i in range(num_episodes):
        print(f'Training episode {i}')

        # Train env_player
        custom_play_against(
            env_player=env_player,
            env_algorithm=dqn.train_one_episode,
            opponent=random_player,
        )

        # Evaluate
        evaluate_model(
            player=dqn,
            opponent=random_player,
            n_battles=100
        )

        print(dqn.n_finished_battles)
        print(dqn.n_won_battles)

        agent_games_cum[i] = dqn.n_finished_battles
        agent_wins_cum[i] = dqn.n_won_battles

        if i == 0:
            agent_games[i] = agent_games_cum[i]
            agent_wins[i] = agent_wins_cum[i]
        else:
            agent_games[i] = agent_games_cum[i] - agent_games_cum[i-1]
            agent_wins[i] = agent_wins_cum[i] - agent_wins_cum[i-1]

    print(agent_games)
    print(agent_wins)

    plt.figure()
    plt.plot(episodes, agent_wins, '-b', label="Agent Wins")
    plt.xlabel("Episode")
    plt.ylabel("Number of Wins")
    plt.title("Agent Wins Per Episode")
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    main()




