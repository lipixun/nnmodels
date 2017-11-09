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
import multiprocessing

import tfutils
import numpy as np
import tensorflow as tf

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
    def __init__(self, _id, name, envName, envNums, clusterSpec, taskIndex, writeLock, stopEvent, gpuMemFaction):
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
        while not self.stopEvent.isSet():
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
                exps.append([states, newStates, actions, rewards, terminates])
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
                expBatch = np.stack(exps).reshape(-1, 5)
                states = np.stack(expBatch[:, 0])
                actions = np.stack(expBatch[:, 2])
                nextStates = np.stack(expBatch[:, 1])
                rewards = np.stack(expBatch[:, 3])
                terminates = np.invert(expBatch[:, 4].astype(np.bool)).astype(np.float32)   # pylint: disable=no-member
                values = []
                for i in range(int(math.ceil(expBatch.shape[0] / float(batchSize)))):
                    v, _ = network.predict(nextStates[i*batchSize: (i+1)*batchSize, ...], session)
                    values.append(v)
                values = np.concatenate(values).reshape(-1)
                # The target values
                targetValues = rewards + values * discountFactor * terminates   # NOTE: 1-step return
                # Update
                losses = []
                for i in range(int(math.ceil(expBatch.shape[0] / float(batchSize)))):
                    loss = network.update(states[i*batchSize: (i+1)*batchSize, :], actions[i*batchSize: (i+1)*batchSize], targetValues[i*batchSize: (i+1)*batchSize])
                    losses.append(loss)
                loss = np.mean(losses)
                gLossIndex = (gLossIndex + 1) % 100
                gLosses[gLossIndex] = loss
            # Show metric
            if gEpisode % 100 == 0:
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
        parser.add_argument("-p", "--parallel", dest="parallel", type=int, help="The parallel worker number. Will use all cpu cores if not specified")
        parser.add_argument("-n", "--name", dest="name", default="Breakout-v0", help="The game env name")
        parser.add_argument("-e", "--env-num-per-worker", dest="envNumPerWorker", type=int, default=64, help="The environment number per worker")
        parser.add_argument("-g", "--gpu-memory-faction", dest="gpuMemoryFaction", type=float, default=0.1, help="The gpu memory faction per worker")
        parser.add_argument("-p", "--port-start-from", dest="portStartFrom", type=int, default=51234, help="The port start from")
        return parser.parse_args()

    def main():
        """The main entry
        """
        args = getArguments()
        # Create lock and event
        stopEvent = multiprocessing.Event()
        writeLock = multiprocessing.Lock()
        # Create cluster spec
        psAddrs = [ "127.0.0.1:%d" % args.portStartFrom ]
        workerAddrs = ["127.0.0.1:%d" % (p + 1) for p in range(args.portStartFrom)]
        clusterSpec = tf.train.ClusterSpec({"ps": psAddrs, "worker": workerAddrs})
        # Start servers
        processes = []
        for i in psAddrs:
            process = multiprocessing.Process(target=ParameterServer(clusterSpec, i, stopEvent))
            process.start()
            processes.append(process)
        for i in workerAddrs:
            process = multiprocessing.Process(target=AgentWorker(i, "main", args.name, args.envNumPerWorker, clusterSpec, i, writeLock, stopEvent, args.gpuMemoryFaction))
            process.start()
            processes.append(process)
        # Wait and exit
        try:
            while True:
                time.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            # Stop and wait
            stopEvent.set()
            for process in processes:
                process.join()

    main()
