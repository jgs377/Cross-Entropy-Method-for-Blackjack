import numpy as np
import matplotlib.pyplot as plt
import pickle
import operator
from random import shuffle


class Blackjack_CEM(object):
    def __init__(self, p=8, n=300):
        # vector of means
        self.mu = np.random.uniform(size=3)
        # vector of standard deviations
        self.sigma = np.random.uniform(low = 0.001, size=3)
        # amount of top vectors
        self.p = p
        # sample size
        self.n = n
        # self.running_reward = 0

    # cross product of vector W, with state info
    def soft_max(self, state, W):
        action = np.dot(W,state)
        action = int(action > 1)
        return [action]

    # W is a vector of parameters
    # returns reward for 1 episode
    def noisy_evaluate(self, env, W):
        reward_sum = 0
        state = env.reset()
        while True:
            action = self.soft_max(state,W)
            state, reward, done,_ = env.step(int(action[0]))
            reward_sum += reward
            if done:
                break
        return reward_sum

    def sample_policy(self, training=True):
        if training:
            ws = np.zeros(3)
            for i in range(3):
                ws[i] = np.random.normal(loc=self.mu[i], scale=self.sigma[i]+1e17)
            return ws
        else:
            ws = np.zeros(3)
            testsigma = np.full(self.sigma.shape[0],0.01)
            for i in range(3):
                ws[i] = np.random.normal(loc=self.mu[i], scale=testsigma[i]+1e17)
            return ws

    def train(self, env, iters):
        running_reward = 0
        state = env.reset()
        i = 0
        plt.ion()
        while i < iters:
            dims = 3
            w_array = np.zeros((self.n, dims))
            reward_sums = np.zeros(self.n)

            for k in range(self.n):
                w_array[k] = self.sample_policy()
                reward_sums[k] = self.noisy_evaluate(env, w_array[k])
            env.reset()

            rankings = np.argsort(reward_sums)

            top_vectors = w_array[rankings, :]
            top_vectors = top_vectors[-self.p:, :]

            for q in range(top_vectors.shape[1]):
                self.mu[q] = top_vectors[:,q].mean()
                self.sigma[q] = top_vectors[:,q].std()

            running_reward = 0.99 * running_reward + 0.01 * reward_sums.mean()

            plt.scatter(i, reward_sums.mean(), color='y')
            plt.scatter(i, running_reward, color='r')
            plt.pause(0.001)
            i += 1

    def test(self, env, iterations):
        wins, losses, draws, score_sum = 0, 0, 0, 0

        for i in range(iterations):
            policy = self.sample_policy(training=False)
            reward = self.noisy_evaluate(env, policy)
            env.reset()
            score_sum += reward
            if reward == 1:
                wins += 1
            elif reward == 0:
                draws += 1
            elif reward == -1:
                losses += 1
            else:
                print("ERROR!")

        print("###################")
        print("Average Score: ", score_sum / iterations)
        print("Wins: ", wins)
        print("Draws: ", draws)
        print("Losses: ", losses)
        print("###################")

    def save(self):
        with open('blackjackCEM.model', 'wb') as f:
            pickle.dump(self.mu,f)
            print("Saved model information to 'blackjackCEM.model'")

    def load(self):
        with open('blackjackCEM.model', 'rb') as f:
            self.mu = pickle.load(f)
            print("Loaded model from 'blackjackCEM.model'")