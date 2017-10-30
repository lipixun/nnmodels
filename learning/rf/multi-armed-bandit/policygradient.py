# encoding=utf8

""" The policy gradient algorithm
    Author: lipixun
    File Name: policygradient.py
    Description:

        See: https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149

"""

import numpy as np
import tensorflow as tf

#
# A method to simulate
#
Bandits     = [0.2, 0.0, -0.2, -5.0]
NBandits    = len(Bandits)

def pullBandit(no):
    """Pull the bandit
    Args:
        no(int): The number of the bandit
    Returns:
        int: The reward
    """
    return 1 if np.random.randn(1) > Bandits[no] else -1    # pylint: disable=no-member

#
# Define the agent network
#
weights = tf.Variable(tf.ones((NBandits,)))
prediction = tf.argmax(weights)

# The train method
targetReward = tf.placeholder(tf.float32, shape=(1,))
actualAction = tf.placeholder(tf.int32, shape=(1,))
loss = -(tf.log(tf.reduce_sum(tf.one_hot(actualAction, NBandits) * tf.reshape(weights, (1, NBandits)))) * targetReward)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
update = optimizer.minimize(loss)

#
# Train the agent
#
totalEpisodes = 1000
totalReward = np.zeros(NBandits)
e = 1e-1

with tf.Session() as session:
    # Init all variables
    session.run(tf.global_variables_initializer())
    # Run training
    for epoch in xrange(totalEpisodes):
        # Choose an action either by network or by random
        if np.random.rand(1) < e:   # pylint: disable=no-member
            action = np.random.randint(NBandits)    # pylint: disable=no-member
        else:
            action = session.run(prediction)
        # Get the reward
        reward = pullBandit(action)
        # Update the network
        session.run(update, feed_dict={targetReward: [reward], actualAction: [action]})
        # Update the monitor vars
        totalReward[action] += reward
        if epoch % 50 == 0:
            print "Rewards:", totalReward
    # Done. Get the weights
    w = session.run(weights)
    print "Weights:", w
