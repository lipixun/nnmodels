# encoding=utf8
# pylint: disable=not-context-manager

""" The version 1
    Author: lipixun
    Created Time : 2017-11-03 17:51:14

    File Name: v1.py
    Description:

        URL: https://arxiv.org/abs/1710.09829v1

"""

import tensorflow as tf

class CapsuleNetworkHyperParams(object):
    """The Hyper parameter of capsule network
    """
    def __init__(self, primaryCapsuleChannelSize=32, secondaryCapsuleInnerChannelSize=16):
        """Create a new CapsuleNetworkHyperParams
        NOTE:
            The default values are the hyper parameters which are specified in the original paper.
        """
        self.primaryCapsuleChannelSize = primaryCapsuleChannelSize
        self.secondaryCapsuleInnerChannelSize = secondaryCapsuleInnerChannelSize

class CapsuleNetwork(object):
    """Implement capsule network
    """
    def __init__(self, params):
        """Create a new CapsuleNetwork
        """
        #
        # Build the network
        #
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        # The first layer is a CNN network with 9 x 9 filter and 256 channels
        with tf.variable_scope("layer1"):
            W = tf.get_variable("W", [9, 9, 1, 256], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable("b", [256], tf.float32, tf.zeros_initializer()) # Has a bias or not? The paper doesn't clearly specified
            conv = tf.nn.bias_add(tf.nn.conv2d(tf.reshape(self.x, [-1, 28, 28, 1]), W, [1, 1, 1, 1], "VALID"), b)
            conv = tf.nn.relu(conv) # Shape [None, 20, 20, 256]
        # The second layer is a primary capsule layer
        with tf.variable_scope("layer2"):
            # ```
            # The second layer (PrimaryCapsules) is a convolutional capsule layer with 32 channels of convolutional 8D capsules (i.e. each primary capsule contains 8 convolutional units with a 9 x 9 kernel and a stride of 2).
            # ```
            capsules = self.capsule(conv, params.primaryCapsuleChannelSize)  # Shape = [BatchSize, (Cap) Channel(32 by default), 6, 6, 8]


    @classmethod
    def capsule(cls, inp, channels, convChannels=8):
        """Create new capsules

        The original paper:
            `each primary capsule contains 8 convolutional units with a 9 x 9 kernel and a stride of 2`

        NOTE:
            This method intends to build multiple capsules into one conv opt for performance consideration.

        Args:
            inp(tf.Tensor): The input tensor in shape (BatchSize, Height, Width, Channels)
            channels(int): The number of capsule channel
            convChannels(int): The number of conv channels for each capsule
        Returns:
            tf.Tensor: The output tensor in shape [BatchSize, Channels, Height, Width, Conv Channels]
        """
        with tf.variable_scope("capsule"):
            # Conv
            W = tf.get_variable("W", [9, 9, inp.shape[-1].value, channels * convChannels], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable("b", [channels * convChannels], tf.float32, tf.zeros_initializer())    # Has a bias or not? The paper doesn't clearly specified
            conv = tf.nn.bias_add(tf.nn.conv2d(inp, W, [1, 2, 2, 1], "VALID"), b) # Shape = [BatchSize, 6, 6, Channels]
            # Squash
            conv = cls.squash(conv)
            # Reshape
            return tf.transpose(tf.reshape(conv, [-1, conv.shape[1].value, conv.shape[2].value, channels, convChannels]), [0, 3, 1, 2, 4])

    @classmethod
    def squash(cls, s):
        """The so called `squashing` function that is described in the paper
        """
        with tf.variable_scope("squash"):
            # The norm square ||s|| ^ 2
            normSquare = tf.reduce_sum(tf.square(s), axis=-1, keep_dims=True)
            # The same that is described in paper
            return s * normSquare / (1 + normSquare) / tf.sqrt(normSquare)


    @classmethod
    def dynamicRouting(cls, inp, nextChannels, nextInnerChannels):
        """Create a new dynamic routing layer
        Args:
            inp(tf.Tensor): The output of a capsule in shape [BatchSize, ChannelSize, Height, Width, InnerChannelSize]
            nextChannels(int): The channel size of next capsule
            nextInnerChannels(int): The inner channel size of next capsule
        Returns:
            tf.Tensor: The input of next capsule in shape [BatchSize, NextChannels, ?]
        """
        prevChannels, prevHeight, prevWidth, prevInnerChannels = [x.value for x in inp.shape[-4:]]
        inpTotalChannels = prevChannels * prevHeight * prevWidth
        with tf.variable_scope("dynamic-routing"):
            # u(j|i) = W(ij) * u(i)
            # ```
            # W(ij) is a weight matrix between each u(i)
            # ```
            # Shape = [nextChannels(10 by default), PrevChannels(32 by default) * 6 * 6, prevInnerChannels(8 by default), nextInnerChannels(16 by default)]
            # So, default shape = [10, 1152, 8, 16]
            #
            # But for performance consideration, the actual shape of weight is [1152, 8, 16 * 10] by default
            #
            W = tf.get_variable("W", [inpTotalChannels, prevInnerChannels, nextInnerChannels * nextChannels], tf.float32, tf.random_normal_initializer())
            u = tf.matmul(tf.reshape(inp, [-1, inpTotalChannels, 1, prevInnerChannels], w))
            # The `prediction vector`. Shape = [BatchSize, NextChannels(10 by default), PrevChannels(32 by default) * 6 * 6, NextInnerChannels(16 by default)]
            u = tf.transpose(tf.reshape(u, [-1, inpTotalChannels, nextChannels, nextInnerChannels]), [0, 2, 1, 3])





            u = tf.reshape(tf.matmul(tf.reshape(inp, [-1, prevInnerChannels]), tf.reshape(W, [-1, prevInnerChannels, nextInnerChannels])), []



            b = tf.get_variable("b", [prevChannels, nextChannels], tf.float32, tf.zeros_initializer())
            tf.nn.softmax(b)
