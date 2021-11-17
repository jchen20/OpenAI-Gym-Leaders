import copy
import numpy as np
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from dqn_agent import DQNAgent


class RLEnvPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100 # Simple rescaling to facilitate learning
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

        # Final vector with 10 components
        return np.concatenate(
            [moves_base_power, moves_dmg_multiplier, [remaining_mon_team, remaining_mon_opponent]]
        )

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2,
            hp_value=1,
            victory_value=30,
        )

NB_TRAINING_STEPS = 10000
NB_EVALUATION_EPISODES = 100
def main():
    env = RLEnvPlayer(battle_format="gen8randombattle")
    player = DQNAgent(10, len(env.action_space))
    player.set_embed_battle(env.embed_battle)
    opponent = RandomPlayer()
    
    num_episodes = 2
    for i in range(num_episodes):
        # Uncomment below for self play
        # opponent = copy.deepcopy(player)
        print(f'Training episode {i}')
        env.play_against(
            env_algorithm=player.train_one_episode,
            opponent=opponent
        )
    player.battle_against(opponent, n_battles=5)


if __name__ == '__main__':
    main()

