from poke_env.player.player import Player
import random

class Agent(Player):

    def training(self, env):
        done = False
        rews = []
        acts = []
        while not done:
            act = random.choice(env.action_space)
            _, rew, done, _ =  env.step(act)
            print(rew)
            rews.append(rew)
            acts.append(act)
        
        #Do weight update here

    def choose_move(self, battle):
        moves = battle.available_moves
        return random.choice(moves)
        
