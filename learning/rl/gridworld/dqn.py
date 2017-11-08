# encoding=utf8
# pylint: disable=not-context-manager,redefined-outer-name

""" The basic dqn implementation
    Author: lipixun
    File Name: dqn.py
    Description:

        See: https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb

"""

import random

import numpy as np
import tfutils
import tensorflow as tf

from gridworld import GameEnv, ImageSize, ImageDepth

class QNetwork(object):
    """The q network
    """
    def __init__(self, actionNums):
        """Create a new QNetwork
        """
        self.state = tf.placeholder(tf.float32, [None, ImageSize, ImageSize, ImageDepth])
        # Apply cnn layers
        with tf.variable_scope("cnn-0"):
            cnn0 = self.conv2d(self.state, 32, [8, 8], [1, 4, 4, 1])
        with tf.variable_scope("cnn-1"):
            cnn1 = self.conv2d(cnn0, 64, [4, 4], [1, 2, 2, 1])
        with tf.variable_scope("cnn-2"):
            cnn2 = self.conv2d(cnn1, 64, [3, 3], [1, 1, 1, 1])
        cnnOut = tf.reshape(cnn2, [-1, np.prod([d.value for d in cnn2.shape[1:]])])
        cnnOut = tf.nn.dropout(cnnOut, 0.9)
        # Duel-DQN
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
        self.outputQ = value + advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True)
        self.prediction = tf.argmax(self.outputQ, axis=1)
        #
        # Train method
        #
        self.targetQ = tf.placeholder(tf.float32, [None])
        self.actions = tf.placeholder(tf.int32, [None])
        qValues = tf.reduce_sum(tf.one_hot(self.actions, actionNums) * self.outputQ, axis=1)
        error = self.targetQ - qValues
        self.loss = tf.reduce_mean(tf.where(tf.abs(error) > 1.0, tf.abs(error), tf.square(error))) # Huber loss
        self.updateop = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

    def conv2d(self, inp, filters, ksize, strides):
        """Conv 2d
        """
        W = tf.get_variable("W", list(ksize) + [inp.shape[-1], filters], tf.float32, tf.random_normal_initializer())
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
        return session.run([self.prediction, self.outputQ], feed_dict={self.state: states})

    def update(self, states, targetQ, actions, session):
        """Update the model
        """
        _, loss = session.run([self.updateop, self.loss], feed_dict={self.state: states, self.targetQ: targetQ, self.actions: actions})
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

    def __len__(self):
        """Length
        """
        return len(self.buffer)

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
        return np.stack(random.sample(self.buffer, size)).reshape([-1, 5])

#
# NOTE:
#   The `Policy Graph` here is `mainQN` in the original codes
#   The `Value Graph` here is `targetQN` in the original codes
#

def buildValueGraphUpdateOp(policyGraphVars, targetGraphVars):
    """Build the update op of value graph
    """
    ops = []
    for i, var in enumerate(policyGraphVars):
        ops.append(tf.assign(targetGraphVars[i], var))
    # Group all operations together and return
    return tf.group(*ops)

if __name__ == "__main__":

    from argparse import ArgumentParser

    def getArguments():
        """Get arguments
        """
        parser = ArgumentParser(description="GridWorld DQN")
        parser.add_argument("--pretrain-steps", dest="preTrainSteps", type=int, default=100, help="The pre-train steps")
        parser.add_argument("--update-target-network-steps", dest="updateTargetNetworkSteps", type=int, default=10000, help="The step interval used to update target network")
        parser.add_argument("--discount-factor", dest="discountFactor", type=float, default=0.99, help="The discount factor")
        parser.add_argument("--batch-size", dest="batchSize", default=256, help="The batch size")
        parser.add_argument("--max-epoch", dest="maxEpoch", type=int, default=100, help="The max epoch")
        parser.add_argument("--e-start", dest="eStart", type=float, default=1.0, help="The e start")
        parser.add_argument("--e-end", dest="eEnd", type=float, default=0.1, help="The e end")
        parser.add_argument("--e-reduce-steps", dest="eReduceSteps", type=int, default=1e7, help="The e reduce step number")
        parser.add_argument("--grid-size", dest="gridSize", type=int, default=5, help="The grid size")
        parser.add_argument("-e", "--env-nums", dest="envNums", type=int, default=64, help="The number of grid envs that is observed at the same time")
        return parser.parse_args()

    def main():
        """The main entry
        """
        args = getArguments()
        # Create environments
        envs = [GameEnv(False, args.gridSize) for _ in range(args.envNums)]
        expBuffer = ExperienceBuffer(size=100000)
        # Create networks
        with tf.variable_scope("policy") as scope:
            policyGraph = QNetwork(envs[0].actions)
            policyGraphVars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.GLOBAL_VARIABLES)
        with tf.variable_scope("value") as scope:
            targetGraph = QNetwork(envs[0].actions)
            targetGraphVars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.GLOBAL_VARIABLES)
            # Get the update op of value graph
            targetGraphUpdateOp = buildValueGraphUpdateOp(policyGraphVars, targetGraphVars)
        # Variables
        e = args.eStart
        gStep = 0
        eStepReduceValue = float(args.eStart - args.eEnd) / float(args.eReduceSteps)
        # Counters
        gLosses, gRewards = [], []
        # Train
        with tf.Session(config=tfutils.session.newConfigProto(0.25)) as session:
            # Init all variables
            session.run(tf.global_variables_initializer())
            episode = 0
            while True:
                episode += 1
                epoch = 0
                epochRewards = [0.0] * len(envs)
                # Reset the environment
                states = []
                for env in envs:
                    states.append(env.reset())
                states = np.stack(states)
                # Run
                while epoch < args.maxEpoch:
                    gStep += 1
                    epoch += 1
                    # Check update target network or not
                    if gStep % args.updateTargetNetworkSteps:
                        session.run(targetGraphUpdateOp)
                    # Choose actions (By e-greedy)
                    if gStep < args.preTrainSteps or np.random.rand(1) < e:     # pylint: disable=no-member
                        actions = [np.random.randint(0, envs[0].actions) for _ in envs]   # pylint: disable=no-member
                    else:
                        actions, _ = policyGraph.predict(states, session)
                        actions = actions.reshape(-1)
                    # Execute the environment
                    newStates = []
                    allTerminated = True
                    for i, action in enumerate(actions):
                        s, r, t = envs[i].step(action)
                        expBuffer.add(np.array([states[i], s, action, r, t]))    # Force terminated at the end of max epoch length
                        newStates.append(s)
                        if not t:
                            epochRewards[i] += r
                            allTerminated = False
                    if e > args.eEnd:
                        e -= eStepReduceValue
                    if allTerminated:
                        break
                    states = np.stack(newStates)
                gRewards.extend(epochRewards)
                # Update network
                if gStep > args.preTrainSteps and len(expBuffer) >= args.batchSize:
                    exps = expBuffer.sample(args.batchSize)
                    # Calculate the target rewards
                    policyPreds, _ = policyGraph.predict(np.stack(exps[:, 1]), session)
                    _, valueOuts = targetGraph.predict(np.stack(exps[:, 1]), session)
                    terminateFactor = np.invert(exps[:, 4].astype(np.bool)).astype(np.float32)    # pylint: disable=no-member
                    targetQ = exps[:, 3] + (valueOuts[range(args.batchSize), policyPreds] * args.discountFactor * terminateFactor)
                    # Update policy network
                    loss = policyGraph.update(np.stack(exps[:, 0]), targetQ, exps[:, 2], session)
                    # Add loss
                    gLosses.append(loss)
                # Show metrics
                if episode % 100 == 0:
                    print "Episode [%d] Global Step [%d] E[%.4f] | Latest 100 Episodes: Mean Loss [%f] Reward Mean [%.4f] Var [%.4f]" % (episode, gStep, e, np.mean(gLosses), np.mean(gRewards), np.var(gRewards))
                    gLosses, gRewards = [], []

    main()
