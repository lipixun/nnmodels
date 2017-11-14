# encoding=utf8
# pylint: disable=not-context-manager,redefined-outer-name

""" The basic dqn implementation
    Author: lipixun
    File Name: dqn.py
    Description:

        See: https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb

"""

import os
import os.path
import random
import shutil

import numpy as np
import tfutils
import tensorflow as tf

try:
    from moviepy.editor import ImageSequenceClip as Clip
except ImportError:
    Clip = None

from gridworld import GameEnv, ImageSize, ImageDepth

class CoreNet(object):
    """The core network
    """
    def __init__(self, actionNums, cnn=False, duel=False, huberLoss=False, lr=None):
        """Create a new QNetwork
        """
        self.images = tf.placeholder(tf.float32, [None, ImageSize, ImageSize, ImageDepth])
        self.temperature = tf.constant(1.0, dtype=tf.float32)
        # Apply cnn layers
        if cnn:
            # 3-conv layers
            with tf.variable_scope("cnn-0"):
                cnn0 = self.conv2d(self.images, 32, [8, 8], [1, 4, 4, 1])
            with tf.variable_scope("cnn-1"):
                cnn1 = self.conv2d(cnn0, 64, [4, 4], [1, 2, 2, 1])
            with tf.variable_scope("cnn-2"):
                cnn2 = self.conv2d(cnn1, 64, [3, 3], [1, 1, 1, 1])
            hout = tf.reshape(cnn2, [-1, np.prod([d.value for d in cnn2.shape[1:]])])
        else:
            # 3-fc layers
            with tf.variable_scope("fc-0"):
                fc0 = self.fc(tf.reshape(self.images, [-1, ImageSize * ImageSize * ImageDepth]), 2000, tf.nn.relu)
            with tf.variable_scope("fc-1"):
                fc1 = self.fc(fc0, 1000, tf.nn.relu)
            with tf.variable_scope("fc-2"):
                fc2 = self.fc(fc1, 1000, tf.nn.relu)
            hout = fc2
        if duel:
            # Duel-DQN
            with tf.variable_scope("value"):
                with tf.variable_scope("fc"):
                    fcOut = self.fc(hout, 512, tf.nn.relu)
                with tf.variable_scope("out"):
                    value = self.fc(fcOut, 1)
            with tf.variable_scope("advantage"):
                with tf.variable_scope("fc"):
                    fcOut = self.fc(hout, 512, tf.nn.relu)
                with tf.variable_scope("out"):
                    advantage = self.fc(fcOut, actionNums)
            # Output
            self.q = value + advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True)
        else:
            # Direct output
            with tf.variable_scope("output"):
                with tf.variable_scope("fc"):
                    fcOut = self.fc(hout, 512, tf.nn.relu)
                with tf.variable_scope("out"):
                    self.q = self.fc(fcOut, actionNums)
        self.qdist = tf.nn.softmax(self.q / self.temperature)
        self.pAction = tf.argmax(self.q, axis=1)
        #
        # Train method
        #
        self.targetQ = tf.placeholder(tf.float32, [None])
        self.actions = tf.placeholder(tf.int32, [None])
        error = self.targetQ - tf.reduce_sum(tf.one_hot(self.actions, actionNums) * self.q, axis=1)
        if huberLoss:
            self.loss = tf.reduce_mean(tf.where(tf.abs(error) > 1.0, tf.abs(error), tf.square(error))) # Huber loss
        else:
            self.loss = tf.square(error)
        self.updateop = tf.train.AdamOptimizer(learning_rate=lr or 1e-3).minimize(self.loss)

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

class QNetwork(object):
    """The q network
    """
    def init(self, session):
        """Init the session
        Args:
            session(tf.Session): The tensorflow session
        """
        raise NotImplementedError

    def predict(self, images, session, t=1.0):
        """Predict
        Args:
            images(np.array): The images in 2d-array
            session(tf.Session): The tensorflow session
            t(float): The temperature
        Returns:
            (action, q, qdist): The tuple of result
        """
        raise NotImplementedError

    def getq(self, images, session):
        """Get the q value
        Args:
            images(np.array): The images in 2d-array
            session(tf.Session): The tensorflow session
        Returns:
            np.array: The q values in 1d-array
        """
        raise NotImplementedError

    def update(self, episode, images, actions, targetQ, session, t=1.0):
        """Update
        Args:
            episode(int): The episode
            images(np.array): The images in 2d-array
            actions(np.array): The actions in 1d-array
            targetQ(np.array): The target q in 1d-array
            session(tf.Session): The tensorflow session
            t(float): The temperature
        """
        raise NotImplementedError

class SimpleQNetwork(QNetwork):
    """The simple q-network
    """
    def __init__(self, actionNums, cnn=False, duel=False, huberLoss=False):
        """Create a new SimpleQNetwork
        """
        self.net = CoreNet(actionNums, cnn=cnn, duel=duel, huberLoss=huberLoss)

    def init(self, session):
        """Init the session
        Args:
            session(tf.Session): The tensorflow session
        """
        pass

    def predict(self, images, session, t=1.0):
        """Predict
        Args:
            images(np.array): The images in 2d-array
            session(tf.Session): The tensorflow session
            t(float): The temperature
        Returns:
            (action, q, qdist): The tuple of result
        """
        return session.run([self.net.pAction, self.net.q, self.net.qdist], feed_dict={self.net.images: images, self.net.temperature: t})

    def getq(self, images, session):
        """Get the q value
        Args:
            images(np.array): The images in 2d-array
            session(tf.Session): The tensorflow session
        Returns:
            np.array: The q values in 1d-array
        """
        return session.run(self.net.q, feed_dict={self.net.images: images}).max(axis=1)

    def update(self, episode, images, actions, targetQ, session, t=1.0):
        """Update
        Args:
            episode(int): The episode
            images(np.array): The images in 2d-array
            actions(np.array): The actions in 1d-array
            targetQ(np.array): The target q in 1d-array
            session(tf.Session): The tensorflow session
            t(float): The temperature
        Returns:
            float: The loss
        """
        _, loss = session.run([self.net.updateop, self.net.loss], feed_dict={self.net.images: images, self.net.actions: actions, self.net.targetQ: targetQ, self.net.temperature: t})
        return loss

class DoubleQNetwork(QNetwork):
    """The double q-network
    """
    def __init__(self, actionNums, cnn=False, duel=False, huberLoss=False, updateIntervalEpisode=1e2):
        """Create a new DuelQNetwork
        """
        self.updateIntervalEpisode = updateIntervalEpisode
        # Create network
        with tf.variable_scope("policy") as scope:
            self.policyNet = CoreNet(actionNums, cnn=cnn, duel=duel, huberLoss=huberLoss)
            policyNetTrainVars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.GLOBAL_VARIABLES)
        with tf.variable_scope("target") as scope:
            self.targetNet = CoreNet(actionNums, cnn=cnn, duel=duel, huberLoss=huberLoss)
            targetNetTrainVars = tf.contrib.framework.get_variables(scope, collection=tf.GraphKeys.GLOBAL_VARIABLES)
        # The target network update op
        with tf.variable_scope("target-update"):
            ops = []
            for i, var in enumerate(policyNetTrainVars):
                ops.append(tf.assign(targetNetTrainVars[i], var))
            self.updateTargetOp = tf.group(*ops)

    def init(self, session):
        """Init the session
        Args:
            session(tf.Session): The tensorflow session
        """
        session.run(self.updateTargetOp)

    def predict(self, images, session, t=1.0):
        """Predict
        Args:
            images(np.array): The images in 2d-array
            session(tf.Session): The tensorflow session
            t(float): The temperature
        Returns:
            (action, q, qdist): The tuple of result
        """
        return session.run([self.policyNet.pAction, self.policyNet.q, self.policyNet.qdist], feed_dict={self.policyNet.images: images, self.policyNet.temperature: t})

    def getq(self, images, session):
        """Get the q value
        Args:
            images(np.array): The images in 2d-array
            session(tf.Session): The tensorflow session
        Returns:
            np.array: The q values in 1d-array
        """
        actions = session.run(self.policyNet.pAction, feed_dict={self.policyNet.images: images})
        q = session.run(self.targetNet.q, feed_dict={self.targetNet.images: images})
        return q[range(actions.shape[0]), actions]

    def update(self, episode, images, actions, targetQ, session, t=1.0):
        """Update
        Args:
            episode(int): The episode
            images(np.array): The images in 2d-array
            actions(np.array): The actions in 1d-array
            targetQ(np.array): The target q in 1d-array
            session(tf.Session): The tensorflow session
            t(float): The temperature
        Returns:
            float: The loss
        """
        _, loss = session.run([self.policyNet.updateop, self.policyNet.loss], feed_dict={self.policyNet.images: images, self.policyNet.actions: actions, self.policyNet.targetQ: targetQ, self.policyNet.temperature: t})
        if episode % self.updateIntervalEpisode == 0:
            session.run(self.updateTargetOp)
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
#

if __name__ == "__main__":

    from argparse import ArgumentParser

    def getArguments():
        """Get arguments
        """
        parser = ArgumentParser(description="GridWorld DQN")
        parser.add_argument("--cnn", dest="cnn", action="store_true", help="Use cnn instead of full connected layers")
        parser.add_argument("--huber", dest="huber", action="store_true", help="Use huber loss instead of square loss")
        parser.add_argument("--duel", dest="duel", action="store_true", help="Use duel-DQN")
        parser.add_argument("--double", dest="double", action="store_true", help="Use double-DQN")
        parser.add_argument("--pretrain-episodes", dest="preTrainEpisodes", type=int, default=10, help="The pre-train episodes before running training")
        parser.add_argument("--buffer-size", dest="bufferSize", type=int, default=1e5, help="The buffer size")  # Make this too large will consume too much memory
        parser.add_argument("--update-policy-episodes", dest="updatePolicyEpisodes", type=int, default=4, help="The episode interval used to update policy graph")
        parser.add_argument("--update-target-episodes", dest="updateTargetEpisodes", type=int, default=1e2, help="The episode interval used to update target network (Only available in Double-DQN)")
        parser.add_argument("--discount-factor", dest="discountFactor", type=float, default=0.99, help="The discount factor")
        parser.add_argument("--batch-size", dest="batchSize", type=int, default=256, help="The batch size")
        parser.add_argument("--max-epoch", dest="maxEpoch", type=int, default=1e2, help="The max epoch")
        parser.add_argument("--e-start", dest="eStart", type=float, default=1.0, help="The e start")
        parser.add_argument("--e-end", dest="eEnd", type=float, default=0.1, help="The e end")
        parser.add_argument("--e-reduce-steps", dest="eReduceSteps", type=int, default=1e6, help="The e reduce step number")
        parser.add_argument("-e", "--env-nums", dest="envNums", type=int, default=64, help="The number of grid envs that is observed at the same time")
        parser.add_argument("--grid-size", dest="gridSize", type=int, default=5, help="The grid size")
        parser.add_argument("--outside-reward", dest="outsideReward", type=float, default=0.0, help="The reword when move to outside")
        parser.add_argument("--step-cost", dest="stepCost", type=float, default=-1e-3, help="The cost for each step (This value will be added to reward)")
        parser.add_argument("--write-gif-path", dest="writeGIFPath", default="output-gifs", help="The generated gif images")
        parser.add_argument("--write-gif-episodes", dest="writeGIFEpisodes", type=int, default=100, help="The number of episode to write a gif")
        return parser.parse_args()

    def main():
        """The main entry
        """
        args = getArguments()
        # Check images
        if Clip and args.writeGIFEpisodes:
            if os.path.isdir(args.writeGIFPath):
                shutil.rmtree(args.writeGIFPath)
            os.makedirs(args.writeGIFPath)
        # Create environments
        envs = [GameEnv(False, args.gridSize, outsideReward=args.outsideReward, stepCost=args.stepCost) for _ in range(args.envNums)]
        actionNums = envs[0].actions
        expBuffer = ExperienceBuffer(size=args.bufferSize)
        # Create network
        if args.double:
            net = DoubleQNetwork(actionNums, cnn=args.cnn, duel=args.duel, huberLoss=args.huber, updateIntervalEpisode=args.updateTargetEpisodes)
        else:
            net = SimpleQNetwork(actionNums, cnn=args.cnn, duel=args.duel, huberLoss=args.huber)
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
            # Run network init
            net.init(session)
            # Start
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
                imageBuffer = [states[0]]
                # Run
                while epoch < args.maxEpoch:
                    gStep += 1
                    epoch += 1
                    # Choose actions (By e-greedy)
                    if episode < args.preTrainEpisodes or np.random.rand(1) < e:                 # pylint: disable=no-member
                        # By random
                        actions = [np.random.randint(0, envs[0].actions) for _ in envs]     # pylint: disable=no-member
                    else:
                        # By boltzmann
                        _, _, qdist = net.predict(states, session)
                        actions = []
                        for i in range(qdist.shape[0]):
                            actions.append(np.random.choice(qdist.shape[1], size=1, p=qdist[i]))    # pylint: disable=no-member
                        actions = np.array(actions)
                    # Execute the environment
                    newStates = []
                    allTerminated = True
                    for i, action in enumerate(actions):
                        s, r, t, _ = envs[i].step(action)
                        expBuffer.add(np.array([states[i], s, action, r, t]))    # Force terminated at the end of max epoch length
                        newStates.append(s)
                        if i == 0:
                            imageBuffer.append(s)
                        if not t:
                            epochRewards[i] += r
                            allTerminated = False
                    if e > args.eEnd:
                        e -= eStepReduceValue
                    if allTerminated:
                        break
                    states = np.stack(newStates)
                gRewards.extend(epochRewards)
                # Update policy network
                if episode > args.preTrainEpisodes and episode % args.updatePolicyEpisodes == 0 and len(expBuffer) >= args.batchSize * args.envNums:
                    expBatches = expBuffer.sample(args.batchSize * args.envNums)
                    for i in range(args.envNums):
                        exps = expBatches[i*args.batchSize: (i+1)*args.batchSize, ...]
                        # Calculate the target q
                        nextStates = np.stack(exps[:, 1])
                        terminateFactor = np.invert(exps[:, 4].astype(np.bool)).astype(np.float32)    # pylint: disable=no-member
                        targetQ = exps[:, 3] + net.getq(nextStates, session) * args.discountFactor * terminateFactor
                        # Update policy network
                        loss = net.update(episode, np.stack(exps[:, 0]), exps[:, 2], targetQ, session)
                        # Add loss
                        gLosses.append(loss)
                # Save gifs
                if Clip and args.writeGIFEpisodes and episode % args.writeGIFEpisodes == 0:
                    clip = Clip(imageBuffer, fps=1)
                    clip.write_gif(os.path.join(args.writeGIFPath, "episode-%d.gif" % episode), fps=1)
                # Show metrics
                if episode % 100 == 0:
                    print "Episode [%d] Global Step [%d] E[%.4f] | Latest 100 Episodes: Mean Loss [%f] Reward Mean [%.4f] Var [%.4f]" % (episode, gStep, e, np.mean(gLosses), np.mean(gRewards), np.var(gRewards))
                    gLosses, gRewards = [], []

    main()
