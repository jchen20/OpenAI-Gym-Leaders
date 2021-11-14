import asyncio
from tabulate import tabulate
from poke_env.player.random_player import RandomPlayer
from poke_env.player.utils import cross_evaluate
from poke_env.player_configuration import PlayerConfiguration

from agents.max_damage import MaxDamagePlayer

async def main():
    max_damage_player = MaxDamagePlayer(
        player_configuration=PlayerConfiguration('Max_Damage_Bot', None),
        battle_format="gen8randombattle",
    )
    # Now, let's evaluate our player
    await max_damage_player.send_challenges('Lost Fellow', n_challenges=1)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
