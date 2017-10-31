# encoding=utf8
# pylint: disable=redefined-outer-name

""" The contextual bandit
    Author: lipixun
    File Name: contextual.py
    Description:

"""

import numpy as np
import tensorflow as tf

class ContextualBandit(object):
    """The contextual bandit
    """
    def __init__(self):
        """Create a new ContextualBandit
        """
        self.no = 0 # The number of current bandit
        self.bandits = np.array([[0.2, 0, -0.0, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]])

    @property
    def banditNum(self):
        """The bandit number
        """
        return self.bandits.shape[0]

    @property
    def actionNum(self):
        """The action number
        """
        return self.bandits.shape[1]

    def getState(self):
        """Get current state (The state will be changed everytime this method is called)
        """
        self.no = np.random.randint(0, self.banditNum)  # pylint: disable=no-member
        return self.no

    def pull(self, action):
        """Pull the bandit
        """
        return 1 if np.random.randn(1) > self.bandits[self.no, action] else -1  # pylint: disable=no-member

# The agent
class Agent(object):
    """The agent
    """
    def __init__(self, banditNum, actionNum):
        """Create a new agent
        """
        self.state = tf.placeholder(tf.int32, shape=(1,))
        # A full connected layer
        W = tf.Variable(tf.random_normal((banditNum, actionNum)))
        b = tf.Variable(tf.zeros(actionNum, ))
        self.output = tf.nn.sigmoid(tf.nn.xw_plus_b(tf.one_hot(self.state, banditNum), W, b))
        # The prediction
        self.prediction = tf.argmax(self.output, axis=1)
        # The train methods
        self.targetReward = tf.placeholder(tf.float32, shape=(1,))
        self.actualAction = tf.placeholder(tf.int32, shape=(1,))
        self.loss = -(tf.log(tf.reduce_sum(self.output * tf.one_hot(self.actualAction, actionNum))) * self.targetReward)
        self.updateop = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

    def predict(self, state, session):
        """Predict
        """
        return session.run(self.prediction, feed_dict={self.state: [state]})[0]

    def update(self, state, reward, action, session):
        """Update
        """
        session.run(self.updateop, feed_dict={self.state: [state], self.targetReward: [reward], self.actualAction: [action]})

#
# Run training
#

bandit = ContextualBandit()
agent = Agent(bandit.banditNum, bandit.actionNum)

totalEpisodes = 100000
totalRewards = np.zeros([bandit.banditNum, bandit.actionNum])
e = 1e-1

with tf.Session() as session:
    # Init all variables
    session.run(tf.global_variables_initializer())
    # Run training
    for epoch in xrange(totalEpisodes):
        state = bandit.getState()
        # Choose an action
        if np.random.rand(1) < e:   # pylint: disable=no-member
            action = np.random.randint(bandit.actionNum)    # pylint: disable=no-member
        else:
            action = agent.predict(state, session)
        # Get reward
        reward = bandit.pull(action)
        # Update
        agent.update(state, reward, action, session)
        # Update the monitor vars
        totalRewards[state, action] += reward
        if epoch % 500 == 0:
            print "Mean rewards:", totalRewards.mean(axis=1)
    # Print prediction
    for i in range(bandit.banditNum):
        print "Prediction of [%d] is [%d]" % (i, agent.predict(i, session))
