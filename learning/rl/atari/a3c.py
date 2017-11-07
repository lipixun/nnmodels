# encoding=utf8
# pylint: disable=not-context-manager,redefined-outer-name

""" The `Asynchronus Advantage Actor-Critic` version
    Author: lipixun
    File Name: a3c.py
    Description:

"""

import sys
import time
import math
import random
import threading
import multiprocessing

import gym
import numpy as np
import tensorflow as tf

batchSize       = 32
preTrainSteps   = 50000
discountFactor  = 0.99

ValueLossFactor = 0.5
PolicyEntropyFactor = 1e-2
MaxEpoch        = 10000

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
            cnnOut = tf.reshape(cnn1, [-1, np.prod([d.value for d in cnn1.shape[1:]])])
            with tf.variable_scope("fc"):
                fcOut = self.fc(cnnOut, 512, tf.nn.relu)
            # The value output
            with tf.variable_scope("value"):
                self.value = self.fc(fcOut, 1)
            # The policy output
            with tf.variable_scope("policy"):
                self.policy = self.fc(fcOut, actionNums, tf.nn.softmax)
            # All trainable vars
            self.trainableVars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.GLOBAL_VARIABLES)
            #
            # The train method
            #
            if globalVars:
                with tf.variable_scope("train"):
                    self.actions = tf.placeholder(tf.int32, [None])
                    self.targetValue = tf.placeholder(tf.float32, [None])   # The discounted v(s) + r
                    advantage = self.targetValue - tf.reshape(self.value, [-1])
                    # The loss consists of three parts: Policy loss, Value loss and Policy entropy
                    policyLogProba = tf.log(self.policy + 1e-6)
                    # NOTE: The error == advantage only when we're using `1-step return loss`. If we're using `n-step return loss`, the error should be discounted(v(s)) - v(s)
                    valueLoss = tf.where(tf.abs(advantage) > 1.0, tf.abs(advantage), tf.square(advantage))  # Huber loss
                    policyLoss = tf.negative(tf.reduce_sum(tf.one_hot(self.actions, actionNums) *policyLogProba, axis=1)) * tf.stop_gradient(advantage)
                    policyEntropy = tf.negative(tf.reduce_sum(self.policy *policyLogProba, axis=1))
                    self.loss = tf.reduce_mean(policyLoss + ValueLossFactor * valueLoss + PolicyEntropyFactor * policyEntropy)
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
    def __init__(self, name, actionNums, gNetwork, localNetwork):
        """Create a new agent worker
        """
        self.name = name
        self.actionNums = actionNums
        self.gNetwork = gNetwork
        self.localNetwork = localNetwork
        # Create an op used to sync local network to global network
        ops = []
        for gVar, lVar in zip(gNetwork.trainableVars, localNetwork.trainableVars):
            ops.append(tf.assign(lVar, gVar))
        self.syncOp = tf.group(*ops)
        # Create a new environment
        self.env = gym.make("Breakout-v0")

    def __call__(self, session):
        """Run this worker
        """
        print >>sys.stderr, "Worker [%s] Start" % self.name
        e = eStart
        gStep = 0
        gEpisode = 0
        gRewards, gRewardIndex = [0.0] * 100, -1
        gSteps, gStepIndex = [0.0] * 100, -1
        gLosses, gLossIndex = [0.0] * 100, -1
        while True:
            gEpisode += 1
            # Sync network
            session.run(self.syncOp)
            # Reset env
            exps = []
            state = self.env.reset()
            epoch = 0
            totalReward = 0.0
            while epoch < MaxEpoch:
                epoch += 1
                gStep += 1
                # Choose an action
                while True:
                    if gStep < preTrainSteps or np.random.rand(1) < e: # pylint: disable=no-member
                        action = self.env.action_space.sample()
                    else:
                        _, actions = self.localNetwork.predict([state], session)
                        action = np.random.choice(self.actionNums, p=actions[0])   # pylint: disable=no-member
                    # Execute in environment
                    newState, reward, terminated, _ = self.env.step(action)
                    if terminated or not np.array_equal(newState, state):
                        break
                # Clip reward
                totalReward += reward
                reward = max(-1, min(1, reward))
                exps.append([state, newState, action, reward, terminated])
                if e > eEnd:
                    e -= eStepReduceValue
                # Update
                state = newState
                if terminated:
                    break
            # Update gRewards and gSteps
            gStepIndex = (gStepIndex + 1) % 100
            gRewardIndex = (gRewardIndex + 1) % 100
            gSteps[gStepIndex] = epoch
            gRewards[gRewardIndex] = totalReward
            #
            # Train
            #
            if exps:
                expBatch = np.stack(exps)
                states = np.stack(expBatch[:, 0])
                actions = np.stack(expBatch[:, 2])
                nextStates = np.stack(expBatch[:, 1])
                rewards = np.stack(expBatch[:, 3])
                terminates = np.invert(expBatch[:, 4].astype(np.bool)).astype(np.float32)   # pylint: disable=no-member
                values, _ = self.localNetwork.predict(nextStates, session)
                values = values.reshape(-1)
                # The target values
                targetValues = rewards + values * discountFactor * terminates
                # Update
                losses = []
                for i in range(int(math.ceil(expBatch.shape[0] / float(batchSize)))):
                    _, loss = session.run([self.localNetwork.updateop, self.localNetwork.loss], feed_dict={
                        self.localNetwork.state: states[i*batchSize: (i+1)*batchSize, :],
                        self.localNetwork.actions: actions[i*batchSize: (i+1)*batchSize],
                        self.localNetwork.targetValue: targetValues[i*batchSize: (i+1)*batchSize],
                        })
                    losses.append(loss)
                loss = np.mean(losses)
                gLossIndex = (gLossIndex + 1) % 100
                gLosses[gLossIndex] = loss
            # Show metric
            if gEpisode % 100 == 0:
                print >>sys.stderr, "Worker [%s] Episode [%d] E [%.2f] Mean Loss [%f] Mean Step[%.2f] Mean Reward[%.4f]" % (self.name, gEpisode, e, np.mean(gLosses), np.mean(gSteps), np.mean(gRewards))

if __name__ == "__main__":

    from argparse import ArgumentParser

    def getArguments():
        """Get arguments
        """
        parser = ArgumentParser(description="Atari A3C")
        parser.add_argument("-p", dest="p", type=int, help="The parallel worker number. Will use all cpu cores if not specified")
        return parser.parse_args()

    def main():
        """The main entry
        """
        args = getArguments()
        # Start training
        with tf.Session() as session:
            # Create global network and workers
            #env = gym.make("Breakout-v0")
            trainer = tf.train.AdamOptimizer(learning_rate=1e-3, use_locking=True)
            gNetwork = A3CNetwork("global", trainer, 4)
            nWorkers = args.p if args.p else multiprocessing.cpu_count()
            workers = [ AgentWorker("worker-%d" % i, 4, gNetwork, A3CNetwork("worker-%i" % i,trainer, 4, globalVars=gNetwork.trainableVars)) for i in range(nWorkers) ]
            # Init all variables
            session.run(tf.global_variables_initializer())
            # Start all threads and wait
            threads = []
            for worker in workers:
                t = threading.Thread(target=worker, kwargs=dict(session=session))
                t.daemon = True
                t.start()
                threads.append(t)
            # Run
            while True:
                time.sleep(1)

    main()
