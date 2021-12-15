import copy
import asyncio
import numpy as np
import time

import matplotlib.pyplot as plt

from poke_env.data import to_id_str
from poke_env.player.env_player import Gen8EnvSinglePlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.environment.battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder

from utils import get_valid_actions_mask, one_hot, player_action_to_move


class RLEnvPlayer(Gen8EnvSinglePlayer):

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4, dtype=int)
        moves_dmg_multiplier = np.zeros(4)
        moves_accuracy = np.zeros(4)
        moves_categories = np.zeros(4, dtype=int)
        moves_status = np.zeros(4, dtype=int)
        moves_heal = np.zeros(4, dtype=int)
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
                if move.type in battle.active_pokemon.types:
                    moves_dmg_multiplier[i] *= 1.5 # STAB
            moves_heal[i] = move.heal
        
        # dynamax_moves_base_power = -np.ones(4, dtype=int)
        # dynamax_moves_dmg_multiplier = np.zeros(4)
        # dynamax_moves_accuracy = np.zeros(4)
        # dynamax_moves_categories = np.zeros(4, dtype=int)
        # dynamax_moves_status = np.zeros(4, dtype=int)
        # dynamax_moves_heal = np.zeros(4, dtype=int)
        # for i, move in enumerate(battle.available_moves):
        #     move = move.dynamaxed
        #     dynamax_moves_base_power[i] = move.base_power / 100 # Simple rescaling to facilitate learning
        #     dynamax_moves_accuracy[i] = move.accuracy
        #     dynamax_moves_categories[i] = move.category.value
        #     if move.status:
        #         dynamax_moves_status[i] = move.status.value
        #     else:
        #         dynamax_moves_status[i] = 0
        #     if move.type:
        #         dynamax_moves_dmg_multiplier[i] = move.type.damage_multiplier(
        #             battle.opponent_active_pokemon.type_1,
        #             battle.opponent_active_pokemon.type_2,
        #         )
        #         if move.type in battle.active_pokemon.types:
        #             dynamax_moves_dmg_multiplier[i] *= 1.5 # STAB
        #     dynamax_moves_heal[i] = move.heal

        moves_categories -= 1
        category_matrix = np.zeros((4, 3))
        category_matrix[np.arange(4), moves_categories] = 1
        
        status_matrix = np.zeros((4, 7))
        for i, status_type in enumerate(moves_status):
            if status_type != 0:
                status_matrix[i, status_type - 1] = 1

        moves_matrix = np.concatenate([
            moves_base_power,
            moves_dmg_multiplier,
            moves_accuracy,
            moves_heal,
            category_matrix.flatten(),
            status_matrix.flatten(),
        ])
        moves_matrix = moves_matrix.reshape((14, 4)).T

        # dynamax_moves_categories -= 1
        # dynamax_category_matrix = np.zeros((4, 3))
        # dynamax_category_matrix[np.arange(4), dynamax_moves_categories] = 1

        # dynamax_status_matrix = np.zeros((4, 7))
        # for i, status_type in enumerate(dynamax_moves_status):
        #     if status_type != 0:
        #         dynamax_status_matrix[i, status_type - 1] = 1

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )
        
        curr_types = [t.value for t in battle.active_pokemon.types if t is not None]
        curr_one_hot_types = one_hot(curr_types, 18)
    
        opponent_types = [t.value for t in battle.opponent_active_pokemon.types if t is not None]
        opponent_one_hot_types = one_hot(opponent_types, 18)
        
        curr_stats = battle.active_pokemon.stats
        curr_stats_array = np.zeros(5)
        stat_order = ['atk', 'def', 'spa', 'spd', 'spe']
        for i in range(5):
            curr_stats_array[i] = curr_stats[stat_order[i]]
        curr_stats_array /= 100

        curr_base_stats = battle.active_pokemon.base_stats
        curr_base_stats_array = np.zeros(6)
        base_stat_order = stat_order + ['hp']
        for i in range(6):
            curr_base_stats_array[i] = curr_base_stats[base_stat_order[i]]
        curr_base_stats_array /= 100

        curr_boosts = battle.active_pokemon.boosts
        curr_boosts_array = np.zeros(7)
        boost_order = ['accuracy', 'atk', 'def', 'evasion', 'spa', 'spd', 'spe']
        for i in range(7):
            curr_boosts_array[i] = curr_boosts[boost_order[i]]
        
        curr_level = battle.active_pokemon.level / 100
        curr_hp = battle.active_pokemon.current_hp_fraction

        opp_base_stats = battle.opponent_active_pokemon.base_stats
        opponent_base_stats_array = np.zeros(6)
        for i in range(6):
            opponent_base_stats_array[i] = opp_base_stats[base_stat_order[i]]
        opponent_base_stats_array /= 100

        opp_boosts = battle.opponent_active_pokemon.boosts
        opponent_boosts_array = np.zeros(7)
        for i in range(7):
            opponent_boosts_array[i] = opp_boosts[boost_order[i]]

        opponent_level = battle.opponent_active_pokemon.level / 100
        opponent_hp = battle.opponent_active_pokemon.current_hp_fraction

        switch_matrix = np.zeros((5, 25))
        for i, pokemon in enumerate(battle.available_switches):
            switch_matrix[i, :5] = np.array([pokemon.stats[stat_order[j]] for j in range(5)]) / 100
            switch_matrix[i, 5:7] = [pokemon.current_hp_fraction, pokemon.level / 100]
            pokemon_types = [t.value for t in pokemon.types if t is not None]
            switch_matrix[i, 7:25] += one_hot(pokemon_types, 18)
        
        status_arr = np.zeros(7) if battle.active_pokemon.status is None \
            else one_hot(battle.active_pokemon.status.value, 7)
        opp_status_arr = np.zeros(7) if battle.opponent_active_pokemon.status is None \
            else one_hot(battle.opponent_active_pokemon.status.value, 7)

        side_conds = battle.side_conditions
        cond_idxs = []
        cond_cts = []
        for k, v in side_conds.items():
            cond_idxs.append(k.value)
            cond_cts.append(v)
        if cond_idxs:
            side_cond_arr = one_hot(cond_idxs, 20, weight=cond_cts)
        else:
            side_cond_arr = np.zeros(20)
        opp_side_conds = battle.opponent_side_conditions
        opp_cond_idxs = []
        opp_cond_cts = []
        for k, v in opp_side_conds.items():
            opp_cond_idxs.append(k.value)
            opp_cond_cts.append(v)
        if opp_cond_idxs:
            opp_side_cond_arr = one_hot(opp_cond_idxs, 20, weight=opp_cond_cts)
        else:
            opp_side_cond_arr = np.zeros(20)

        fields_idx = list(map(lambda x: x.value, battle.fields.keys()))
        if fields_idx:
            fields_arr = one_hot(fields_idx, 13)
        else:
            fields_arr = np.zeros(13)

        weather_idx = list(map(lambda x: x.value, battle.weather.keys()))
        if weather_idx:
            weather_arr = one_hot(weather_idx, 8)
        else:
            weather_arr = np.zeros(8)
        
        # Final vector with many components
        state_vector = np.concatenate([
            # moves_base_power,
            # moves_dmg_multiplier,
            # moves_accuracy,
            # moves_heal,
            # category_matrix.flatten(),
            # status_matrix.flatten(),
            moves_matrix.flatten(),
            # dynamax_moves_base_power,
            # dynamax_moves_dmg_multiplier,
            # dynamax_moves_accuracy,
            # dynamax_moves_heal,
            # dynamax_category_matrix.flatten(),
            # dynamax_status_matrix.flatten(),
            curr_one_hot_types,
            [curr_level, curr_hp],
            curr_stats_array,
            curr_base_stats_array,
            curr_boosts_array,
            opponent_one_hot_types,
            [opponent_level, opponent_hp],
            opponent_base_stats_array,
            opponent_boosts_array,
            [remaining_mon_team, remaining_mon_opponent],
            status_arr,
            opp_status_arr,
            switch_matrix.flatten(),
            side_cond_arr,
            opp_side_cond_arr,
            fields_arr,
            weather_arr
        ])
        
        return state_vector, get_valid_actions_mask(battle)

    def compute_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=1,
            hp_value=1,
            victory_value=7.5,
            status_value=0.1,
        )
    
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle not in self._observations or battle not in self._actions:
            self._init_battle(battle)
        self._observations[battle].put(self.embed_battle(battle))
        action = self._actions[battle].get()
        self.last_action = action

        return player_action_to_move(self, action, battle)
