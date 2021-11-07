from poke_env.player.player import Player

def Agent(Player):

    def __init__(self):
        super.__init__(self)

    def training(self, env):
        done = False
        while not done:
            act = self.choose_move(env.current_battle)
            env.step(act)





    def choose_move(self, battle):
        moves = battle.available_moves
        return moves[0]
        
