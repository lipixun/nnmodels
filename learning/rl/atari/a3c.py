# encoding=utf8
# pylint: disable=not-context-manager,redefined-outer-name

""" The `Asynchronus Advantage Actor-Critic` version
    Author: lipixun
    File Name: a3c.py
    Description:

"""

import random

import gym
import numpy as np
import tensorflow as tf

batchSize       = 256
preTrainSteps   = 50000
discountFactor  = 0.99

eStart, eEnd, eReduceStepNum = 1.0, 0.1, 1000000
eStepReduceValue = float(eStart - eEnd) / float(eReduceStepNum)

class A3CNetwork(object):
    """The a3c network
    """
    def __init__(self, name, trainer, actionNums, imageHeight=210, imageWidth=160, imageDepth=3, globalVars=None):
        """Create a new QNetwork
        """
        with tf.variable_scope(name) as scope:
            self.state = tf.placeholder(tf.float32, [None, imageHeight, imageWidth, imageDepth])
            # Apply cnn layers
            with tf.variable_scope("cnn-0"):
                cnn0 = self.conv2d(self.state, 32, [8, 8], [1, 4, 4, 1])
            with tf.variable_scope("cnn-1"):
                cnn1 = self.conv2d(cnn0, 64, [4, 4], [1, 2, 2, 1])
            with tf.variable_scope("cnn-2"):
                cnn2 = self.conv2d(cnn1, 64, [3, 3], [1, 1, 1, 1])
            cnnOut = tf.reshape(cnn2, [-1, np.prod([d.value for d in cnn2.shape[1:]])])
            # The value output
            with tf.variable_scope("value"):
                with tf.variable_scope("fc"):
                    fcOut = self.fc(cnnOut, 512, tf.nn.relu)
                with tf.variable_scope("out"):
                    self.value = self.fc(fcOut, 1)
            # The policy output
            with tf.variable_scope("policy"):
                with tf.variable_scope("fc"):
                    fcOut = self.fc(cnnOut, 512, tf.nn.relu)
                with tf.variable_scope("out"):
                    self.policy = self.fc(fcOut, actionNums, tf.nn.softmax)
            # All trainable vars
            self.trainableVars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.GLOBAL_VARIABLES)
            #
            # The train method
            #
            if globalVars:
                with tf.variable_scope("train"):
                    self.actions = tf.placeholder(tf.int32, [None])
                    self.targetValue = tf.placeholder(tf.float32, [None])
                    advantages = self.targetValue - tf.reshape(self.value, [-1]))
                    # The loss
                    valueLoss = 0.5 * tf.square(advantages)
                    #Huber loss: tf.reduce_mean(tf.where(tf.abs(error) > 1.0, tf.abs(error), tf.square(error)))
                    policyLoss = tf.log(tf.reduce_sum(tf.one_hot(self.actions, actionNums) * self.policy, axis=1)) * tf.stop_gradient(self.advantages)
                    policyEntropy = 1e-2 * tf.reduce_sum(self.policy * tf.log(self.policy), axis=1)
                    self.loss = tf.reduce_mean(policyLoss + valueLoss + policyEntropy)
                    # The train method (To global network)
                    gradients = tf.gradients(self.loss, self.trainableVars)
                    grads, _ = tf.clip_by_global_norm(gradients, 40.0)
                    self.updateop = trainer.apply_gradients(zip(grads, globalVars))

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
        Returns:
            tuple: A tuple of (value, policy)
        """
        return session.run([self.value, self.policy], feed_dict={self.state: states})

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

class AgentWorker(object):
    """The agent worker
    """
    def __init__(self, gNetwork, localNetwork, expBufferSize):
        """Create a new agent worker
        """
        self.gNetwork = gNetwork
        self.localNetwork = localNetwork
        # Create an op used to sync local network to global network
        ops = []
        for gVar, lVar in zip(gNetwork.trainableVars, localNetwork.trainableVars):
            ops.append(tf.assign(lVar, gVar))
        self.syncOp = tf.group(*ops)
        # Create the experience buffer
        self.expBuffer = ExperienceBuffer(expBufferSize)
        # Create a new environment
        self.env = gym.make("Breakout-v0")

    def __call__(self, session):
        """Run this worker
        """
        e = eStart
        gStep = 0
        gRewards, gRewardIndex = [0.0] * 100, -1
        gSteps, gStepIndex = [0.0] * 100, -1
        while True:
            # Sync network
            session.run(self.syncOp)
            # Reset env
            expBuffer.reset()
            state = self.env.reset()
            epoch = 0
            totalReward = 0.0
            while True:
                epoch += 1
                # Choose an action
                value, actions = self.localNetwork.predict([state], session)
                if gStep < preTrainSteps or np.random.rand(1) < e: # pylint: disable=no-member
                    action = env.action_space.sample()
                else:
                    actions = actions[0]
                    action = np.random.choice(actions, p=actions)   # pylint: disable=no-member
                    action = np.argmax(action == actions)
                value = value[0, 0]
                # Execute in environment
                newState, reward, terminated, _ = env.step(action)
                expBuffer.add(np.array([state, newState, action, reward, terminated, value]).reshape(1, -1))    # Force terminated at the end of max epoch length
                gStep += 1
                if e > eEnd:
                    e -= eStepReduceValue
                # Update
                state = newState
                totalReward += reward
                if terminated:
                    # Update gRewards and gSteps
                    gStepIndex = (gStepIndex + 1) % 100
                    gRewardIndex = (gRewardIndex + 1) % 100
                    gSteps[gStepIndex] = epoch
                    gRewards[gRewardIndex] = totalReward
                    # Done
                    break
            #
            # Train
            #
            expBatch = np.array(expBuffer.buffer)
            states = expBatch[:, 0]
            actions = expBatch[:, 2]
            nextStates = expBatch[:, 1]
            rewards = expBatch[:, 3]
            values = expBatch[:, 5]
            # The target values
            targetValues = self.discount(rewards, discountFactor)[:-1]
            # The target advantages
            targetAdvantages = rewards[:-1] + discountFactor * values[1:] - values[:-1]
            targetAdvantages = self.discount(targetAdvantages, discountFactor)

    def discount(self, values, factor):
        """Discount the value
        """
        discountedValues = []

if __name__ == "__main__":

    totalEpisodes   = 10000000
    preTrainSteps   = 50000
    maxEpochLength  = 100
    updateFreq      = 5
    batchSize       = 256
    discountFactor  = 0.99



    env = gym.make("Breakout-v0")
    expBuffer = ExperienceBuffer(size=100000)


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
            epoch = 0
            while True:
                epoch += 1
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
            stepRecords.append(epoch + 1)
            rewardRecords.append(totalReward)
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
            if episode % 10 == 0:
                print "Episode [%d] Global Step [%d] E[%.4f] Mean Step [%.4f] Mean Reward [%.4f]" % (episode, gStep, e, np.mean(stepRecords[-10:]), np.mean(rewardRecords[-10:]))
