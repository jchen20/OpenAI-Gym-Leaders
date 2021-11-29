from poke_env.player.baselines import SimpleHeuristicsPlayer
from rl_env import RLEnvPlayer
from dqn_agent import DQNAgent
import asyncio
import teams

async def main():
    bf = "gen8ou"

    player_username = 'qreqwerqr'
    # heur_player = SimpleHeuristicsPlayer(battle_format=bf, team=team_2_pokemon)
    env_player = RLEnvPlayer(battle_format=bf, team=teams.two_team_1_2)
    dqn = DQNAgent(250, len(env_player.action_space), battle_format=bf, team=teams.two_team_1_2)
    # await heur_player.send_challenges(player_username, 1)
    await dqn.send_challenges(player_username, 1)

asyncio.get_event_loop().run_until_complete(main())
