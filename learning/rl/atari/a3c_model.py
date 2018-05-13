# encoding=utf8
# pylint: disable=not-context-manager,redefined-outer-name

""" The `Asynchronus Advantage Actor-Critic` version
    Author: lipixun
    File Name: a3c.py
    Description:

"""

import numpy as np
import tensorflow as tf

ValueLossFactor     = 0.5
PolicyEntropyFactor = 1e-2

class A3CNetwork(object):
    """The a3c network
    """
    def __init__(self, name, actionNums, imageHeight=210, imageWidth=160, imageDepth=3, lr=None, globalVars=None):
        """Create a new QNetwork
        """
        with tf.variable_scope(name) as scope:
            self.images = tf.placeholder(tf.float32, [None, imageHeight, imageWidth, imageDepth])
            # Apply cnn layers
            with tf.variable_scope("cnn-0"):
                cnn0 = self.conv2d(self.images, 32, [8, 8], [1, 4, 4, 1])
            with tf.variable_scope("cnn-1"):
                cnn1 = self.conv2d(cnn0, 64, [4, 4], [1, 2, 2, 1])
            with tf.variable_scope("cnn-2"):
                cnn2 = self.conv2d(cnn1, 64, [3, 3], [1, 1, 1, 1])
            cnnOut = tf.reshape(cnn2, [-1, np.prod([d.value for d in cnn2.shape[1:]])])
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
            with tf.variable_scope("train"):
                self.actions = tf.placeholder(tf.int32, [None])
                self.targetValues = tf.placeholder(tf.float32, [None])   # The discounted v(s) + r
                advantage = self.targetValues - tf.reshape(self.value, [-1])
                #
                # The loss consists of three parts: Policy loss, Value loss and Policy entropy
                #
                policyLogProba = tf.log(self.policy + 1e-6)
                # NOTE: The error == advantage only when we're using `1-step return loss`. If we're using `n-step return loss`, the error should be discounted(v(s)) - v(s)
                valueLoss = tf.where(tf.abs(advantage) > 1.0, tf.abs(advantage), tf.square(advantage))  # Huber loss
                policyLoss = tf.negative(tf.reduce_sum(tf.one_hot(self.actions, actionNums) * policyLogProba, axis=1)) * tf.stop_gradient(advantage)
                policyEntropy = tf.negative(tf.reduce_sum(self.policy * policyLogProba, axis=1))
                self.loss = tf.reduce_mean(policyLoss + ValueLossFactor * valueLoss + PolicyEntropyFactor * policyEntropy)
                # Compute and clip the gradient
                gradients = tf.gradients(self.loss, self.trainableVars)
                grads, _ = tf.clip_by_global_norm(gradients, 40.0)
                #
                # Create the update op. If `globalVars` is specified, then training will be applied to global vars. This is used to simluate `async training`
                # and if when we're using tensorflow `between graph` distributed training, there's no need to manually simulate.
                #
                self.trainer = tf.train.AdamOptimizer(learning_rate=1e-3 if lr is None else lr, use_locking=True)
                if globalVars:
                    self.updateop = self.trainer.apply_gradients(zip(grads, globalVars))
                else:
                    self.updateop = self.trainer.apply_gradients(zip(grads, self.trainableVars))

    def conv2d(self, inp, filters, ksize, strides):
        """Conv 2d
        """
        W = tf.get_variable("W", list(ksize) + [inp.shape[-1].value, filters], tf.float32, tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [filters], tf.float32, tf.zeros_initializer())
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inp, W, strides, "VALID"), b))

    def fc(self, inp, size, act=None):
        """Add a full connected layer
        """
        W = tf.get_variable("W", [inp.shape[-1].value, size], tf.float32, tf.truncated_normal_initializer(stddev=1e-2))
        b = tf.get_variable("b", [size], tf.float32, tf.zeros_initializer())
        out = tf.nn.xw_plus_b(inp, W, b)
        if act:
            out = act(out)
        return out

    def predict(self, images, session):
        """Predict
        Returns:
            tuple: A tuple of (value, policy)
        """
        return session.run([self.value, self.policy], feed_dict={self.images: images})

    def update(self, images, actions, targetValues, session):
        """Update
        Returns:
            float: The loss
        """
        _, loss = session.run([self.updateop, self.loss], feed_dict={self.images: images, self.actions: actions, self.targetValues: targetValues})
        return loss
