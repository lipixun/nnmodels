# encoding=utf8

""" The q-table for frozen lake
    Author: lipixun
    Created Time : 2017-10-26 22:04:48

    File Name: frozenlake.py
    Description:

        Reference: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

"""

import gym
import numpy as np

env = gym.make("FrozenLake-v0")

# Declare the q table
qTable = np.zeros([env.observation_space.n, env.action_space.n])

print "QTable Shape (Observation, Action):", qTable.shape

# Hyper parameters
LR = 0.8
DecayFactor = 0.95
nEpoch = 10000

# Record total rewards
totalRewards = []

# Run training
print "Start training"
for epoch in range(nEpoch):
    totalReward = 0
    state = env.reset()
    for step in range(100):
        # Choose an action by greedily (with noise) picking from q table
        # NOTE: The noise becomes smaller during the training progress
        action = np.argmax(qTable[state,:] + np.random.randn(1, env.action_space.n) * (1.0 / (epoch + 1))) # pylint: disable=no-member
        # Execute by the environment
        newState, reward, terminated, _ = env.step(action)
        # Update the q-table with new observation
        # NOTE: Here we're updating the value by a small value
        qTable[state, action] += LR * (reward + DecayFactor * qTable[newState, :].max() - qTable[state, action])
        # Update
        totalReward += reward
        state = newState
        if terminated:
            break
    totalRewards.append(totalReward)

# Print
print "Total reward tendency:"
for i in range(nEpoch / 500):
    print "\tEpoch [%d-%d]:" % (i*500, (i+1)*500), sum(totalRewards[i*500: (i+1)*500]) / 500.0
print "Q Table:"
print qTable
