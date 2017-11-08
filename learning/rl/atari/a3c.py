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
import tfutils
import numpy as np
import tensorflow as tf

batchSize       = 32
preTrainSteps   = 50000
discountFactor  = 0.99

ValueLossFactor = 0.5
PolicyEntropyFactor = 1e-2
MaxEpoch        = 10000

eStart, eEnd, eReduceStepNum = 1.0, 0.1, 100000
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

class EnvGroup(object):
    """The env group
    """
    def __init__(self, name, num):
        """Create a new EnvGroup
        """
        self.num = num
        self.envs = [gym.make(name) for _ in range(num)]
        self.envTerminates = [False] * num

    @property
    def actionNums(self):
        """Get the action nums
        """
        return self.envs[0].action_space.n

    def render(self):
        """Render
        """
        self.envs[0].render()

    def reset(self):
        """Reset all envs
        """
        self.envTerminates = [False] * self.num
        return np.stack([env.reset() for env in self.envs])

    def step(self, actions):
        """Run envs
        """
        index = 0
        states, rewards, terminates, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            if self.envTerminates[i]:
                continue
            s, r, t, info = env.step(actions[index])
            index += 1
            states.append(s)
            rewards.append(r)
            terminates.append(t)
            infos.append(info)
            if t:
                self.envTerminates[i] = True
        return np.stack(states), np.stack(rewards), np.stack(terminates), infos

class AgentWorker(object):
    """The agent worker
    """
    def __init__(self, name, envs, trainer, gNetwork):
        """Create a new agent worker
        """
        self.name = name
        self.envs = envs
        self.gNetwork = gNetwork
        self.localNetwork = A3CNetwork(name, trainer, envs.actionNums, globalVars=gNetwork.trainableVars)
        # Create an op used to sync local network to global network
        ops = []
        for gVar, lVar in zip(gNetwork.trainableVars, self.localNetwork.trainableVars):
            ops.append(tf.assign(lVar, gVar))
        self.syncOp = tf.group(*ops)

    def __call__(self, session):
        """Run this worker
        """
        print >>sys.stderr, "Worker [%s] Start with [%d] envs" % (self.name, self.envs.num)
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
            states = self.envs.reset()
            epoch = 0
            totalReward = 0.0
            while epoch < MaxEpoch:
                epoch += 1
                gStep += 1
                # Choose an action
                if gStep < preTrainSteps or np.random.rand(1) < e: # pylint: disable=no-member
                    actions = [np.random.randint(0, self.envs.actionNums) for _ in range(states.shape[0])] # pylint: disable=no-member
                else:
                    _, actionProbas = self.localNetwork.predict(states, session)
                    actions = self.selectActions(actionProbas)
                # Execute in environment
                newStates, rewards, terminates, _ = self.envs.step(actions)
                totalReward += rewards.sum()
                # Clip reward
                rewards[rewards > 1] = 1
                rewards[rewards < -1] = -1
                # Add to exps
                for i, newState in enumerate(newStates):
                    exps.append([states[i], newState, actions[i], rewards[i], terminates[i]])
                if e > eEnd:
                    e -= eStepReduceValue
                # Update
                newStates = [s for (i, s) in enumerate(newStates) if not terminates[i]]
                if not newStates:
                    break
                states = np.stack(newStates)
            # Update gRewards and gSteps
            gStepIndex = (gStepIndex + 1) % 100
            gRewardIndex = (gRewardIndex + 1) % 100
            gSteps[gStepIndex] = epoch
            gRewards[gRewardIndex] = totalReward / float(self.envs.num)
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
                values = []
                for i in range(int(math.ceil(expBatch.shape[0] / float(batchSize)))):
                    v, _ = self.localNetwork.predict(nextStates[i*batchSize: (i+1)*batchSize, ...], session)
                    values.append(v)
                values = np.concatenate(values).reshape(-1)
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

    def selectActions(self, probas):
        """Select actions from action probas
        """
        actions = []
        for proba in probas:
            actions.append(np.random.choice(proba.shape[0], p=proba)) # pylint: disable=no-member
        return actions

if __name__ == "__main__":

    from argparse import ArgumentParser

    def getArguments():
        """Get arguments
        """
        parser = ArgumentParser(description="Atari A3C")
        parser.add_argument("-p", "--parallel", dest="p", type=int, help="The parallel worker number. Will use all cpu cores if not specified")
        parser.add_argument("-n", "--name", dest="name", default="Breakout-v0", help="The game env name")
        parser.add_argument("-e", "--env-num-per-worker", dest="envNumPerWorker", type=int, default=64, help="The environment number per worker")
        return parser.parse_args()

    def main():
        """The main entry
        """
        args = getArguments()
        # Start training
        with tf.Session(config=tfutils.session.newConfigProto(0.25)) as session:
            # Create global network and workers
            env = gym.make(args.name)
            trainer = tf.train.AdamOptimizer(learning_rate=1e-3, use_locking=True)
            gNetwork = A3CNetwork("global", trainer, env.action_space.n)
            workers = [ AgentWorker("worker-%d" % i, EnvGroup(args.name, args.envNumPerWorker), trainer, gNetwork) for i in range(args.p if args.p else multiprocessing.cpu_count()) ]
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
