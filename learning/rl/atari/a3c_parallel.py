# encoding=utf8
# pylint: disable=not-context-manager,redefined-outer-name

""" The `Asynchronus Advantage Actor-Critic` implementation of atari game in (single machine) parallel mode
    Author: lipixun
    File Name: a3c.py
    Description:

"""

import sys
import time
import math
import signal
import shutil
import os
import os.path
import multiprocessing

import tfutils
import numpy as np
import tensorflow as tf

try:
    from moviepy.editor import ImageSequenceClip as Clip
except ImportError:
    Clip = None

from env import EnvGroup
from a3c_model import A3CNetwork

batchSize       = 32
preTrainSteps   = 50000
discountFactor  = 0.99

MaxEpoch        = 10000

eStart, eEnd, eReduceStepNum = 1.0, 0.1, 100000
eStepReduceValue = float(eStart - eEnd) / float(eReduceStepNum)

class AgentWorker(object):
    """The agent worker
    """
    def __init__(self, _id, name, envName, envNums, clusterSpec, taskIndex, writeLock, stopEvent, gpuMemFaction, gifPath, gifEpisode):
        """Create a new AgentWorker
        """
        self.id = _id
        self.name = name
        self.envName = envName
        self.envNums = envNums
        self.clusterSpec = clusterSpec
        self.taskIndex = taskIndex
        self.writeLock = writeLock
        self.stopEvent = stopEvent
        self.gpuMemFaction = gpuMemFaction
        self.gifPath = gifPath
        self.gifEpisode = gifEpisode

    def __call__(self):
        """Run this worker
        """
        with self.writeLock:
            print >>sys.stderr, "Worker [%s] starts with name [%s] environment [%s] nums [%d]" % (self.id, self.name, self.envName, self.envNums)
        try:
            # Start server
            server = tf.train.Server(self.clusterSpec, "worker", self.taskIndex)
            # Prepare environments
            envs = EnvGroup(self.envName, self.envNums)
            # Create network
            with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % self.taskIndex, cluster=self.clusterSpec)):
                network = A3CNetwork(self.name, envs.actionNums)
            # Init session
            with tf.train.MonitoredTrainingSession(master=server.target, is_chief=self.taskIndex == 0, config=tfutils.session.newConfigProto(self.gpuMemFaction)) as session:
                self.train(envs, network, session)
        finally:
            with self.writeLock:
                print >>sys.stderr, "Worker [%s] exits" % self.id

    def train(self, envs, network, session):
        """Run train
        """
        e = eStart
        gStep = 0
        gEpisode = 0
        gRewards, gRewardIndex = [0.0] * 100, -1
        gLosses, gLossIndex = [0.0] * 100, -1
        # Start
        with self.writeLock:
            print >>sys.stderr, "Worker [%s] starts to train" % self.id
        while not self.stopEvent.is_set():
            gEpisode += 1
            exps = []
            epoch = 0
            states = envs.reset()
            totalRewards = [0.0] * self.envNums
            while epoch < MaxEpoch:
                epoch += 1
                gStep += 1
                # Choose an action
                if gStep < preTrainSteps or np.random.rand(1) < e: # pylint: disable=no-member
                    actions = [np.random.randint(0, envs.actionNums) for _ in range(states.shape[0])] # pylint: disable=no-member
                else:
                    _, actionProbas = network.predict(states, session)
                    actions = []
                    for proba in actionProbas:
                        actions.append(np.random.choice(proba.shape[0], p=proba)) # pylint: disable=no-member
                # Execute in environment
                indexes, newStates, rewards, terminates, _ = envs.step(actions)
                for i, reward in enumerate(rewards):
                    totalRewards[indexes[i]] += reward
                # Clip reward
                rewards[rewards > 1] = 1
                rewards[rewards < -1] = -1
                # Add to exps
                exps.append([states, newStates, np.array(actions), np.array(rewards), np.array(terminates)])
                if e > eEnd:
                    e -= eStepReduceValue
                # Update
                newStates = [s for (i, s) in enumerate(newStates) if not terminates[i]]
                if not newStates:
                    break
                states = np.stack(newStates)
            # Update gRewards
            for reward in totalRewards:
                gRewardIndex = (gRewardIndex + 1) % 100
                gRewards[gRewardIndex] = reward
            #
            # Train
            #
            if exps:
                states = np.concatenate([exp[0] for exp in exps])
                actions = np.concatenate([exp[2] for exp in exps])
                nextStates = np.concatenate([exp[1] for exp in exps])
                rewards = np.concatenate([exp[3] for exp in exps])
                terminates = np.invert(np.concatenate([exp[4] for exp in exps])).astype(np.float32)   # pylint: disable=no-member
                values = []
                for i in range(int(math.ceil(len(exps) / float(batchSize)))):
                    v, _ = network.predict(nextStates[i*batchSize: (i+1)*batchSize, ...], session)
                    values.append(v)
                values = np.concatenate(values).reshape(-1)
                # The target values
                targetValues = rewards + values * discountFactor * terminates   # NOTE: 1-step return
                # Update
                losses = []
                for i in range(int(math.ceil(len(exps) / float(batchSize)))):
                    loss = network.update(states[i*batchSize: (i+1)*batchSize, :], actions[i*batchSize: (i+1)*batchSize], targetValues[i*batchSize: (i+1)*batchSize], session)
                    losses.append(loss)
                loss = np.mean(losses)
                gLossIndex = (gLossIndex + 1) % 100
                gLosses[gLossIndex] = loss
            # Write gif
            if Clip is not None and self.gifEpisode and gEpisode % self.gifEpisode == 0:
                imageBuffer = []
                for exp in exps:
                    imageBuffer.append(exp[0][0])
                clip = Clip(imageBuffer, fps=1)
                clip.write_gif(os.path.join(self.gifPath, "worker-%s-episode-%d.gif" % (self.id, gEpisode)), fps=8)
            # Show metric
            if gEpisode % 10 == 0:
                with self.writeLock:
                    print >>sys.stderr, "Worker [%s] Episode [%d] E [%.2f] | Latest episodes: Mean Loss [%f] Reward Mean [%.2f] Var [%.4f]" % (self.id, gEpisode, e, np.mean(gLosses),np.mean(gRewards), np.var(gRewards))

class ParameterServer(object):
    """The parameter server
    """
    def __init__(self, clusterSpec, taskIndex, stopEvent):
        """Create a new ParameterServer
        """
        self.clusterSpec = clusterSpec
        self.taskIndex = taskIndex
        self.stopEvent = stopEvent

    def __call__(self):
        """Run this server
        """
        server = tf.train.Server(self.clusterSpec, "ps", self.taskIndex)
        server.start()
        # Wait stop
        self.stopEvent.wait()

if __name__ == "__main__":

    from argparse import ArgumentParser

    def getArguments():
        """Get arguments
        """
        parser = ArgumentParser(description="Atari A3C")
        parser.add_argument("-p", "--parallel", dest="parallel", type=int, default=6, help="The parallel worker number. Will use all cpu cores if not specified")
        parser.add_argument("-n", "--name", dest="name", default="Breakout-v0", help="The game env name")
        parser.add_argument("-e", "--env-num-per-worker", dest="envNumPerWorker", type=int, default=64, help="The environment number per worker")
        parser.add_argument("-g", "--gpu-memory-faction", dest="gpuMemoryFaction", type=float, default=0.1, help="The gpu memory faction per worker")
        parser.add_argument("--port-start-from", dest="portStartFrom", type=int, default=51234, help="The port start from")
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
        # Create new process group
        os.setpgrp()
        try:
            # Create lock and event
            stopEvent = multiprocessing.Event()
            writeLock = multiprocessing.Lock()
            # Create cluster spec
            psAddrs = [ "127.0.0.1:%d" % args.portStartFrom ]
            workerAddrs = ["127.0.0.1:%d" % (args.portStartFrom + p + 1) for p in range(args.parallel)]
            clusterSpec = tf.train.ClusterSpec({"ps": psAddrs, "worker": workerAddrs})
            # Start servers
            processes = []
            process = multiprocessing.Process(target=ParameterServer(clusterSpec, 0, stopEvent))
            process.start()
            processes.append(process)
            for i in range(args.parallel):
                process = multiprocessing.Process(target=AgentWorker(i, "main", args.name, args.envNumPerWorker, clusterSpec, i, writeLock, stopEvent, args.gpuMemoryFaction, args.writeGIFPath, args.writeGIFEpisodes))
                process.start()
                processes.append(process)
            # Wait and exit
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                # Stop and wait
                stopEvent.set()
                for process in processes:
                    process.join()
        finally:
            # Kill all processes within the group
            os.killpg(0, signal.SIGKILL)

    main()
