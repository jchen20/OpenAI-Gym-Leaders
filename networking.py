# Imports copied over from env_player.py
from abc import ABC, abstractmethod, abstractproperty
from gym.core import Env  # pyre-ignore

from queue import Queue
from threading import Thread

from typing import Any, Callable, List, Optional, Tuple, Union

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
from poke_env.player.player import Player
from poke_env.player.env_player import EnvPlayer
from poke_env.player_configuration import PlayerConfiguration
from poke_env.server_configuration import ServerConfiguration
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.utils import to_id_str

import asyncio
import numpy as np  # pyre-ignore
import time



def custom_play_against(
        env_player: EnvPlayer, env_algorithm: Callable, opponent: Player, env_algorithm_kwargs=None
    ):
        """Executes a function controlling the player while facing opponent.

        The env_algorithm function is executed with the player environment as first
        argument. It exposes the open ai gym API.

        Additional arguments can be passed to the env_algorithm function with
        env_algorithm_kwargs.

        Battles against opponent will be launched as long as env_algorithm is running.
        When env_algorithm returns, the current active battle will be finished randomly
        if it is not already.

        :param env_algorithm: A function that controls the player. It must accept the
            player as first argument. Additional arguments can be passed with the
            env_algorithm_kwargs argument.
        :type env_algorithm: callable
        :param opponent: A player against with the env player will player.
        :type opponent: Player
        :param env_algorithm_kwargs: Optional arguments to pass to the env_algorithm.
            Defaults to None.
        """

        async def launch_battles(player: EnvPlayer, opponent: Player):
            battles_coroutine = asyncio.gather(
                player.send_challenges(
                    opponent=to_id_str(opponent.username),
                    n_challenges=1,
                    to_wait=opponent.logged_in,
                ),
                opponent.accept_challenges(
                    opponent=to_id_str(player.username), n_challenges=1
                ),
            )
            await battles_coroutine

        def env_algorithm_wrapper(player, kwargs):
            env_algorithm(player, **kwargs)

        loop = asyncio.get_event_loop()

        if env_algorithm_kwargs is None:
            env_algorithm_kwargs = {}

        thread = Thread(
            target=lambda: env_algorithm_wrapper(env_player, env_algorithm_kwargs)
        )
        thread.start()

        loop.run_until_complete(launch_battles(env_player, opponent))

        thread.join()


async def battle_against_wrapper(player, opponent, n_battles):
    await player.battle_against(opponent, n_battles)

def evaluate_model(player, opponent, n_battles):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(battle_against_wrapper(player, opponent, n_battles=n_battles))

