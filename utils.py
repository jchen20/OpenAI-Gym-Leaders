from poke_env.player.battle_order import ForfeitBattleOrder
import numpy as np
import torch
import random

from networking import battle_against_wrapper
from poke_env.teambuilder.teambuilder import Teambuilder


class RandomTeamFromTeamPool(Teambuilder):
    def __init__(self, pokemon_team_list, reset_team_cycle=1):
        self.pokemon_team_list = pokemon_team_list
        self.reset_team_cycle = reset_team_cycle
        self.ct = 0
        self.current_team = None

    def yield_team(self):
        if self.ct % self.reset_team_cycle == 0:
            self.current_team = random.sample(self.pokemon_team_list,k=1)[0]
        self.ct += 1
        return self.join_team(self.parse_showdown_team(self.current_team))


class RandomTeamFromPool(Teambuilder):
    def __init__(self, pokemon_list, num_pokemon, reset_team_cycle=1):
        self.pokemon_list = pokemon_list
        self.num_pokemon = num_pokemon
        self.reset_team_cycle = reset_team_cycle
        self.ct = 0
        self.current_team = None

    def yield_team(self):
        if self.ct % self.reset_team_cycle == 0:
            choices = random.sample(self.pokemon_list,k=self.num_pokemon)
            self.current_team = ''.join(choices)
        self.ct += 1
        return self.join_team(self.parse_showdown_team(self.current_team))

def random_team(pokemon_list,num_pokemon):
    choices = random.sample(pokemon_list,num_pokemon)
    team_str = ''.join(choices)
    return team_str

def one_hot(locations, size, weight=None):
    vector = np.zeros(size)
    locations = np.array(locations) - 1
    if weight is None:
        vector[locations] = 1
    else:
        vector[locations] = weight
    return vector

def set_random_seed(sd):
    random.seed(sd)
    np.random.seed(sd)
    torch.manual_seed(sd)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def player_action_to_move(player, action, battle):
    """
    NOTE: This is from poke-env, but fixed/simplified so that invalid z-move,
    mega, etc. is gone.
    
    Converts actions to move orders.
    The conversion is done as follows:
    action = -1:
        The battle will be forfeited.
    0 <= action < 4:
        The actionth available move in battle.available_moves is executed.
    [DEPRECATED] 4 <= action < 8:
        The action - 4th available move in battle.available_moves is executed,
        while dynamaxing.
    4 <= action < 10
        The action - 4th available switch in battle.available_switches is executed.
    If the proposed action is illegal, a random legal move is performed.
    :param action: The action to convert.
    :type action: int
    :param battle: The battle in which to act.
    :type battle: Battle
    :return: the order to send to the server.
    :rtype: str
    """
    if action == -1:
        return ForfeitBattleOrder()
    elif (
            action < 4
            and action < len(battle.available_moves)
            and not battle.force_switch
    ):
        return player.create_order(battle.available_moves[action])
    # elif (
    #         battle.can_dynamax
    #         and 0 <= action - 4 < len(battle.available_moves)
    #         and not battle.force_switch
    # ):
    #     return player.create_order(battle.available_moves[action - 4],
    #                                 dynamax=True)
    elif 0 <= action - 4 < len(battle.available_switches):
        return player.create_order(battle.available_switches[action - 4])
    else:
        return player.choose_random_move(battle)

def convert_real_action_to_pokeenv_action(action):
    if action < 4:
        return action
    else:
        return action - 8

def get_valid_actions_mask(battle):
    mask = np.zeros(10)
    if not battle.force_switch:
        ell = len(battle.available_moves)
        mask[0:ell] = 1
        # if battle.can_dynamax:
        #     mask[4:4+ell] = 1
    mask[4:4+len(battle.available_switches)] = 1

    return mask