from poke_env.player.player import Player
import random
import numpy as np

class MaxDamagePlayer(Player):

    def choose_move(self, battle):
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        else:
            return self.force_switch_move(battle)
    
    def force_switch_move(self, battle):
        switches = battle.available_switches
        if battle.opponent_active_pokemon is not None:
            opp_type_1, opp_type_2 = battle.opponent_active_pokemon.types
            naive_switch_advs = [pokemon.type_1.damage_multiplier(opp_type_1, opp_type_2) for pokemon in switches]
            for i in range(len(switches)):
                if switches[i].type_2 is not None:
                    naive_switch_advs[i] *= switches[i].type_2.damage_multiplier(opp_type_1, opp_type_2)
            return self.create_order(switches[np.argmax(naive_switch_advs)])
        else:
            return self.create_order(random.choice(switches))
