def train(env, model):
    pass


def main():

    env = gym.make()
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    # TODO: 
    # 1) Train your model for 650 episodes, passing in the environment and the agent. 
    # 2) Append the total reward of the episode into a list keeping track of all of the rewards. 
    # 3) After training, print the average of the last 50 rewards you've collected.

    # TODO: Visualize your rewards.
    totalRewards = []
    for i in range(650):
        rew =train(env, model)
        totalRewards.append(rew)
    
    finalRew = np.mean(np.asarray(totalRewards)[shape(totalRewards)[0]-50:])
    print(finalRew)

    visualize_data(totalRewards)



if __name__ == '__main__':
    main()

