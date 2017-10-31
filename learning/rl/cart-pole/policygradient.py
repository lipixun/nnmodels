# encoding=utf8
# pylint: disable=redefined-outer-name

""" The policy gradient algorithm to play cart pole
    Author: lipixun
    File Name: policygradient.py
    Description:

"""

import gym
import numpy as np
import tensorflow as tf

def calculateRewards(rewards, discount):
    """Calculate the rewards
    Args:
        rewards(np.array): The 1-d reward array
        discount(float): The discount factor in [0.0, 1.0]
    Returns:
        np.array: The calculated rewards
    NOTE:
        finalReward(t) = reward(t) + discount * finalRewards(t + 1)
    """
    finalRewards = np.zeros_like(rewards)
    counter = 0.0
    for i in reversed(xrange(rewards.shape[0])):
        counter = rewards[i] + counter * discount
        finalRewards[i] = counter
    return finalRewards

class Agent(object):
    """The agent
    """
    def __init__(self, stateSize, actionSize, hiddenSize=32):
        """Create a new Agent
        """
        self.state = tf.placeholder(tf.float32, shape=[None, stateSize])
        # Hidden layer
        W1 = tf.Variable(tf.random_normal([stateSize, hiddenSize]))
        b1 = tf.Variable(tf.zeros([hiddenSize]))
        layer1 = tf.nn.relu(tf.nn.xw_plus_b(self.state, W1, b1))
        # Output layer
        W2 = tf.Variable(tf.random_normal([hiddenSize, actionSize]))
        b2 = tf.Variable(tf.zeros([actionSize]))
        self.output = tf.nn.softmax(tf.nn.xw_plus_b(layer1, W2, b2))
        self.prediction = tf.argmax(self.output, axis=1)
        # Train method
        self.targetReward = tf.placeholder(tf.float32, shape=[None])
        self.actualAction = tf.placeholder(tf.int32, shape=[None])
        self.loss = -tf.reduce_mean(tf.log(tf.reduce_sum(self.output * tf.one_hot(self.actualAction, actionSize), axis=1)) * self.targetReward) # pylint: disable=invalid-unary-operand-type
        self.updateop = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

    def predict(self, state, session):
        """Predict
        """
        return session.run(self.prediction, feed_dict={self.state: state})

    def getOutput(self, state, session):
        """Get output
        """
        return session.run(self.output, feed_dict={self.state: state})

    def update(self, state, reward, action, session):
        """Update
        """
        session.run(self.updateop, feed_dict={self.state: state, self.targetReward: reward, self.actualAction: action})

#
# Train
#
totalEpisodes = 100000
maxEpoch = 999
discountFactor = 0.99

env = gym.make("CartPole-v0")
agent = Agent(4, 2)
totalReward = []

with tf.Session() as session:
    # Init all variables
    session.run(tf.global_variables_initializer())
    # Run training
    for episode in xrange(totalEpisodes):
        state = env.reset()
        episodeReward = 0.0
        history = []
        for epoch in range(maxEpoch):
            # Choose an action by its probability
            actionProbaDist = agent.getOutput([state], session)[0]
            action = np.random.choice([0,1], p=actionProbaDist) # pylint: disable=no-member
            # Run the environment
            newState, reward, terminated, _ = env.step(action)
            # Add to history
            history.append([state, action, reward])
            # Replace state and continue
            state = newState
            episodeReward += reward
            # Run training at the end of the game
            if terminated:
                history = np.array(history)
                # Calculate the discounted rewards
                rewards = calculateRewards(history[:,2], discountFactor)
                # Update
                agent.update(np.stack(history[:, 0]), rewards, history[:, 1], session)
                # Done
                break
        # Update total reward
        totalReward.append(episodeReward)
        # Print monitor vars
        if episode % 100 == 0:
            print np.mean(totalReward[-100:])
