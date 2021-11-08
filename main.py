from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from agent_template import Agent
import numpy as np

class SimpleRLPlayer(Gen8EnvSinglePlayer):
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



def main():
    env = SimpleRLPlayer(battle_format="gen8randombattle")

    opponet = RandomPlayer(battle_format="gen8randombattle")
    model = Agent()
    num_episodes = 100
    env.reset()

    for i in range(num_episodes):

        # opponet2 = Agent.copy()
        env.play_against(
            env_algorithm=model.training,
            opponent=opponet,
            env_algorithm_kwargs={}
        )








if __name__ == '__main__':
    main()

