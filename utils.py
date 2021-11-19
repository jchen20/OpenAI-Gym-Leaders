from poke_env.player.battle_order import ForfeitBattleOrder
import numpy as np

def player_action_to_move(player, action, battle):
    """Converts actions to move orders.
    The conversion is done as follows:
    action = -1:
        The battle will be forfeited.
    0 <= action < 4:
        The actionth available move in battle.available_moves is executed.
    4 <= action < 8:
        The action - 4th available move in battle.available_moves is executed, with
        z-move.
    8 <= action < 12:
        The action - 8th available move in battle.available_moves is executed, with
        mega-evolution.
    8 <= action < 12:
        The action - 8th available move in battle.available_moves is executed, with
        mega-evolution.
    12 <= action < 16:
        The action - 12th available move in battle.available_moves is executed,
        while dynamaxing.
    16 <= action < 22
        The action - 16th available switch in battle.available_switches is executed.
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
    elif (
            not battle.force_switch
            and battle.can_z_move
            and battle.active_pokemon
            and 0
            <= action - 4
            < len(battle.active_pokemon.available_z_moves)  # pyre-ignore
    ):
        return player.create_order(
            battle.active_pokemon.available_z_moves[action - 4], z_move=True
        )
    elif (
            battle.can_mega_evolve
            and 0 <= action - 8 < len(battle.available_moves)
            and not battle.force_switch
    ):
        return player.create_order(battle.available_moves[action - 8],
                                    mega=True)
    elif (
            battle.can_dynamax
            and 0 <= action - 12 < len(battle.available_moves)
            and not battle.force_switch
    ):
        return player.create_order(battle.available_moves[action - 12],
                                    dynamax=True)
    elif 0 <= action - 16 < len(battle.available_switches):
        return player.create_order(battle.available_switches[action - 16])
    else:
        return player.choose_random_move(battle)

def one_hot(locations, size):
    vector = np.zeros(size)
    locations = np.array(locations) - 1
    vector[locations] = 1
    return vector


