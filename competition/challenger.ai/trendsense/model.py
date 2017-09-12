#!/usr/bin/env python
# encoding=utf8
# pylint: disable=wrong-import-position, not-context-manager

""" The model
    Author: lipixun
    File Name: model.py
    Description:

        This script does:

            1. Define the nn model
            2. Run training
            3. Run predicting

"""

import sys
reload(sys)
sys.setdefaultencoding("utf8")

import logging

from os import makedirs
from os.path import isdir

import h5py
import tensorflow as tf
import tftrainer
import tfutils

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("preprocessor")

Path = "model"
XSize = 88
L2Beta = 1e-3
BatchSize = 512
LR=1e-3
IsTraining = tftrainer.getIsTrainingVar()

class Model(object):
    """The model object
    """
    def __init__(self):
        """Create a new Model
        """
        # Define the graph inputs
        self.X = tf.placeholder(shape=(None, XSize), dtype=tf.float32, name="x")
        self.Y = tf.placeholder(shape=(None, ), dtype=tf.float32, name="y")
        self.W = tf.placeholder(shape=(None, ), dtype=tf.float32, name="w")
        # Define layers
        with tf.variable_scope("layer1"):
            inp = tf.cond(IsTraining, true_fn=lambda: tf.nn.dropout(self.X, 0.5), false_fn=lambda: self.X)
            self.layer1W = tf.get_variable("W", shape=(XSize, 2000), initializer=tf.truncated_normal_initializer())
            self.layer1B = tf.get_variable("b", shape=(2000,))
            self.layer1 = tf.nn.tanh(tf.nn.xw_plus_b(inp, self.layer1W, self.layer1B))
        with tf.variable_scope("layer2"):
            inp = tf.cond(IsTraining, true_fn=lambda: tf.nn.dropout(self.layer1, 0.5), false_fn=lambda: self.layer1)
            self.layer2W = tf.get_variable("W", shape=(2000, 200), initializer=tf.truncated_normal_initializer())
            self.layer2B = tf.get_variable("b", shape=(200,))
            self.layer2 = tf.nn.tanh(tf.nn.xw_plus_b(inp, self.layer2W, self.layer2B))
        # Output
        with tf.variable_scope("output"):
            inp = tf.cond(IsTraining, true_fn=lambda: tf.nn.dropout(self.layer2, 0.5), false_fn=lambda: self.layer2)
            self.outputW = tf.get_variable("W", shape=(200, 1), initializer=tf.truncated_normal_initializer())
            self.outputB = tf.get_variable("b", shape=(1,))
            self.outputLogits = tf.nn.xw_plus_b(inp, self.outputW, self.outputB)
            self.output = tf.nn.sigmoid(self.outputLogits)
        # Loss & training method
        self.logitLoss = tf.reshape(self.W, shape=(-1, 1)) * tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(self.Y, shape=(-1,1)), logits=self.outputLogits)
        self.loss = tf.reduce_mean(self.logitLoss, name="loss")
        self.lossWithL2 = self.loss + L2Beta * (tf.nn.l2_loss(self.layer1W) + tf.nn.l2_loss(self.layer2W) + tf.nn.l2_loss(self.outputW))
        self.trainOp = tftrainer.AutoTrainOp(
            name="default",
            losses={"default": self.loss},
            metrics={"logloss": tftrainer.metric.TFStreamingMetric(tf.contrib.metrics.streaming_mean, "logloss", self.logitLoss)},
            optimizers=[tfutils.optimizers.clipGradientByGlobalNorm(tf.train.AdamOptimizer(learning_rate=tf.train.exponential_decay(LR, tftrainer.getGlobalStepVar(), 2000, 0.9)), self.lossWithL2, 5.0)],
        )

    def train(self, inp):
        """Train this model
        """
        # Open input data
        logger.info("Load input data: %s", inp)
        h5f = h5py.File(inp, "r")
        trainFeeder = tftrainer.HDF5Feeder(h5f, { self.X: "x", self.W: "w", self.Y: "y" }, BatchSize, shuffle=True, queue=tftrainer.SimpleFeedQueue(1000))
        testFeeder  = tftrainer.HDF5Feeder(h5f, { self.X: "test-x", self.W: "test-w", self.Y: "test-y" }, BatchSize, shuffle=True, queue=tftrainer.SimpleFeedQueue(1000))
        if not isdir(Path):
            makedirs(Path)
        # Start train
        logger.info("Train the model")
        trainer = tftrainer.Trainer(Path)
        trainer.train(trainOps=self.trainOp, validateOps=self.trainOp, trainFeeder=trainFeeder, validateFeeder=testFeeder, testFeeder=testFeeder, maxEpoch=100)

    def predict(self, inp, out):
        """Predict by this model
        """
        # Load model
        logger.info("Load model")
        trainer = tftrainer.Trainer(Path)
        # Load input data
        logger.info("Load input data: %s", inp)
        h5f = h5py.File(inp, "r")
        feeder = tftrainer.HDF5Feeder(h5f, {self.X: "x"}, BatchSize, queue=tftrainer.SimpleFeedQueue(1000))
        # Predict
        logger.info("Predict")
        outputs = trainer.run(outs=self.output, feeder=feeder)
        # Write file
        logger.info("Write file")
        with open(out, "wb") as fd:
            for output in outputs:
                for value in output:
                    print >>fd, value[0]

if __name__ == "__main__":

    from argparse import ArgumentParser

    def getArguments():
        """Get arguments
        """
        parser = ArgumentParser(description="Model")
        subParsers = parser.add_subparsers(dest="action")
        trainParser = subParsers.add_parser("train", help="Train model")
        trainParser.add_argument("-i", "--input", dest="input", required=True, help="The input data")
        predParser = subParsers.add_parser("pred", help="Predict by model")
        predParser.add_argument("-i", "--input", dest="input", required=True, help="The input data")
        predParser.add_argument("-o", "--output", dest="output", required=True, help="The output data")
        # Done
        return parser.parse_args()

    def main():
        """The main entry
        """
        args = getArguments()
        if args.action == "train":
            return Model().train(args.input)
        else:
            return Model().predict(args.input, args.output)

    main()
