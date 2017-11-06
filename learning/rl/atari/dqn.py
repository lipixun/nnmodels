# encoding=utf8
# pylint: disable=not-context-manager,redefined-outer-name

""" Use dqn to play atari games
    Author: lipixun
    File Name: dqn.py
    Description:

"""

import random

import gym
import numpy as np
import tensorflow as tf

class QNetwork(object):
    """The q network
    """
    def __init__(self, actionNums, imageHeight=210, imageWidth=160, imageDepth=3):
        """Create a new QNetwork
        """
        self.state = tf.placeholder(tf.float32, [None, imageHeight, imageWidth, imageDepth])
        # Apply cnn layers
        with tf.variable_scope("cnn-0"):
            cnn0 = self.conv2d(self.state, 32, [8, 8], [1, 4, 4, 1])
        with tf.variable_scope("cnn-1"):
            cnn1 = self.conv2d(cnn0, 64, [4, 4], [1, 2, 2, 1])
        with tf.variable_scope("cnn-2"):
            cnn2 = self.conv2d(cnn1, 64, [3, 3], [1, 1, 1, 1])
        cnnOut = tf.reshape(cnn2, [-1, np.prod([d.value for d in cnn2.shape[1:]])])
        # Duel-QDN
        with tf.variable_scope("value"):
            with tf.variable_scope("fc"):
                fcOut = self.fc(cnnOut, 512, tf.nn.relu)
            with tf.variable_scope("out"):
                value = self.fc(fcOut, 1)
        with tf.variable_scope("advantage"):
            with tf.variable_scope("fc"):
                fcOut = self.fc(cnnOut, 512, tf.nn.relu)
            with tf.variable_scope("out"):
                advantage = self.fc(fcOut, actionNums)
        # Output
        self.output = value + advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True)
        self.prediction = tf.argmax(self.output, axis=1)
        # Train method
        self.targetRewards = tf.placeholder(tf.float32, [None])
        self.actualActions = tf.placeholder(tf.int32, [None])
        # Huber loss
        actualQValues = tf.reduce_sum(tf.one_hot(self.actualActions, actionNums) * self.output, axis=1)
        #self.loss = tf.reduce_mean(tf.square(self.targetRewards - actualQValues))
        error = self.targetRewards - actualQValues
        self.loss = tf.reduce_mean(tf.where(tf.abs(error) > 1.0, tf.abs(error), tf.square(error)))  # Huber loss
        self.updateop = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

    def conv2d(self, inp, filters, ksize, strides):
        """Conv 2d
        """
        W = tf.get_variable("W", list(ksize) + [inp.shape[-1].value, filters], tf.float32, tf.random_normal_initializer())
        b = tf.get_variable("b", [filters], tf.float32, tf.zeros_initializer())
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inp, W, strides, "VALID"), b))

    def fc(self, inp, size, act=None):
        """Add a full connected layer
        """
        W = tf.get_variable("W", [inp.shape[-1].value, size], tf.float32, tf.random_normal_initializer())
        b = tf.get_variable("b", [size], tf.float32, tf.zeros_initializer())
        out = tf.nn.xw_plus_b(inp, W, b)
        if act:
            out = act(out)
        return out

    def predict(self, states, session):
        """Predict
        """
        return session.run([self.prediction, self.output], feed_dict={self.state: states})

    def update(self, states, rewards, actions, session):
        """Update the model
        """
        _, loss = session.run([self.updateop, self.loss], feed_dict={self.state: states, self.targetRewards: rewards, self.actualActions: actions})
        return loss

class ExperienceBuffer(object):
    """The experience buffer
    """
    def __init__(self, size=50000):
        """Create a new ExperienceBuffer
        """
        self.size = size
        self.index = -1
        self.buffer = []

    def add(self, experience):
        """Add an experience
        """
        self.index = (self.index + 1) % self.size
        # Add to buffer
        if len(self.buffer) < self.size:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience

    def reset(self):
        """Reset the buffer
        """
        self.index = -1
        self.buffer = []

    def sample(self, size):
        """Sample from buffer
        """
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

#
# NOTE:
#   The `Policy Graph` here is `mainQN` in the original codes
#   The `Value Graph` here is `targetQN` in the original codes
#

def buildValueGraphUpdateOp(policyGraphVars, valueGraphVars, r):
    """Build the update op of value graph
    """
    ops = []
    r = tf.constant(r)
    for i, var in enumerate(policyGraphVars):
        value = var * r + (1 - r) * valueGraphVars[i]
        ops.append(tf.assign(valueGraphVars[i], value))
    # Group all operations together and return
    return tf.group(*ops)

if __name__ == "__main__":

    totalEpisodes   = 10000000
    preTrainSteps   = 50000
    maxEpochLength  = 100
    updateFreq      = 5
    batchSize       = 256
    discountFactor  = 0.99

    eStart, eEnd, eReduceStepNum = 1.0, 0.1, 1000000

    env = gym.make("Breakout-v0")
    expBuffer = ExperienceBuffer(size=100000)

    # Create networks
    with tf.variable_scope("policy") as scope:
        policyGraph = QNetwork(env.action_space.n)
        policyGraphVars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.GLOBAL_VARIABLES)
    with tf.variable_scope("value") as scope:
        valueGraph = QNetwork(env.action_space.n)
        valueGraphVars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.GLOBAL_VARIABLES)
        # Get the update op of value graph
        valueGraphUpdateOp = buildValueGraphUpdateOp(policyGraphVars, valueGraphVars, 1e-3)

    #
    # Run training
    #
    e = eStart
    eStepReduceValue = float(eStart - eEnd) / float(eReduceStepNum)
    gStep = 0

    stepRecords = []
    rewardRecords = []

    with tf.Session() as session:
        # Init all variables
        session.run(tf.global_variables_initializer())
        for episode in xrange(totalEpisodes):
            # Reset the environment and buffer
            state = env.reset()
            totalReward = 0.0
            # Run
            for epoch in xrange(maxEpochLength):
                # Choose an action
                if gStep < preTrainSteps or np.random.rand(1) < e: # pylint: disable=no-member
                    action = env.action_space.sample()
                else:
                    action, _ = policyGraph.predict([state], session)
                    action = action[0]
                # Execute the environment
                newState, reward, terminated, _ = env.step(action)
                expBuffer.add(np.array([state, newState, action, reward, terminated]).reshape(1, -1))    # Force terminated at the end of max epoch length
                gStep += 1
                if e > eEnd:
                    e -= eStepReduceValue
                # Replace & update
                totalReward += reward
                state = newState
                if terminated:
                    break
            # Update network
            if gStep > preTrainSteps and gStep % updateFreq == 0:
                exps = expBuffer.sample(batchSize)
                # Calculate the target rewards
                policyPreds, _ = policyGraph.predict(np.stack(exps[:, 1]), session)
                _, valueOuts = valueGraph.predict(np.stack(exps[:, 1]), session)
                terminateFactor = np.invert(exps[:, 4]).astype(np.float32)    # pylint: disable=no-member
                finalOuts = valueOuts[range(batchSize), policyPreds]   # final outs = The output reward of value network of each action that is predicted by policy network
                targetRewards = exps[:, 3] + (finalOuts * discountFactor * terminateFactor)
                # Update policy & value network
                loss = policyGraph.update(np.stack(exps[:, 0]), targetRewards, exps[:, 2], session)
                session.run(valueGraphUpdateOp)
                print "Train loss:", loss
            stepRecords.append(epoch + 1)
            rewardRecords.append(totalReward)
            if episode % 10 == 0:
                print "Episode [%d] Global Step [%d] E[%.4f] Mean Step [%.4f] Mean Reward [%.4f]" % (episode, gStep, e, np.mean(stepRecords[-10:]), np.mean(rewardRecords[-10:]))
