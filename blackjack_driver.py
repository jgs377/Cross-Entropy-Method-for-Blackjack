from RL_agent import Blackjack_CEM
import sys
import gym

#   @@@@@@@@@  README  @@@@@@@@@@@@
# - CEM model will converge before 300 iterations, so more is useless
# - CEM model will never reach a 1:1 win:loss ratio, because it is unable to understand aces
# - for evaluating, command line supports -train/test followed by X (where X = # of episodes)
# - I recommend more testing_episodes than 10, to see more representative score (~ -0.17)

# make the gym
env = gym.make('Blackjack-v0')

# CEM variables
top_vectors = 8
sample_size = 400

# Other Variables
training_episodes = 500
testing_episodes = 10

# load the blackjack CEM class
model = Blackjack_CEM(p=top_vectors, n=sample_size)

# train/test
if "-train" in sys.argv:
    if len(sys.argv) == 3:
        # Lets you set training episodes in command line
        training_episodes = int(sys.argv[2])
    print("Training for ",training_episodes, " episodes...")
    model.train(env, training_episodes)
    model.save()
elif "-test" in sys.argv:
    if len(sys.argv) == 3:
        # Lets you set testing episodes in command line
        testing_episodes = int(sys.argv[2])
    model.load()
    print("Testing for ", testing_episodes, " episodes...")
    model.test(env, testing_episodes)

env.close()
