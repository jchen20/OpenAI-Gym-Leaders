import copy
import asyncio
import numpy as np
import time

from poke_env.data import to_id_str
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

async def battle_against_wrapper(player, opponent, n_battles):
    await player.battle_against(opponent, n_battles)
    # await player.send_challenges(
    #     to_id_str(opponent.username), n_battles, to_wait=opponent.logged_in
    # )
    # await opponent.accept_challenges(to_id_str(player.username), n_battles)

def evaluate(env, player):
    env.reset_battles()
    done = False
    state = env.reset()

    while not done:
        action = player._best_action(state)
        next_state, rwd, done, _ = env.step(action)
        state = next_state


def main():
    start = time.time()
    bf = "gen8randombattle"

    # Initialize agent
    env_player = RLEnvPlayer(battle_format="gen8randombattle")
    dqn = DQNAgent(10, len(env_player.action_space))
    dqn.set_embed_battle(env_player.embed_battle)

    # Initialize random player
    random_player = RandomPlayer(battle_format=bf)

    num_episodes = 2
    agent_wins = np.zeros(num_episodes)

    for i in range(num_episodes):
        print(f'Training episode {i}')

        # Train env_player
        env_player.play_against(
            env_algorithm=dqn.train_one_episode,
            opponent=random_player,
        )

        env_player.play_against(
            env_algorithm=evaluate,
            opponent=random_player,
            env_algorithm_kwargs={"player": dqn}
        )

        # print('trying to instantiate loop')
        # loop = asyncio.get_event_loop()
        # print('loop instantiated')
        # loop.run_until_complete(battle_against_wrapper(dqn, random_player, 5))
        # print('playing battles')

        # Evaluate env_player
        # dqn.battle_against(random_player, n_battles=5)
        print(dqn.n_finished_battles)
        print(dqn.n_won_battles)
        agent_wins[i] = dqn.n_won_battles

    print(agent_wins)

if __name__ == '__main__':
    # asyncio.get_event_loop().run_until_complete(main())
    main()




