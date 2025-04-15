from gravity_gae.layers import *
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D,Concatenate

flags = tf.app.flags
FLAGS = flags.FLAGS

"""
Disclaimer: functions and classes defined from lines 15 to 124 in this file 
come from tkipf/gae original repository on Graph Autoencoders. Functions and 
classes from line 127 correspond to Source-Target and Gravity-Inspired 
models from our paper.
"""


class Model(object):
    """ Model base class """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        # Wrapper for _build()
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GravityGAEModelAE(Model):
    """
    Gravity-Inspired Graph Autoencoder with 2-layer GCN encoder
    and Gravity-Inspired asymmetric decoder
    """
    def __init__(self, placeholders, num_features, features_nonzero,normalized_pagerank_csr, **kwargs):
        super(GravityGAEModelAE, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.normalized_pagerank_csr = normalized_pagerank_csr
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):

        self.hidden = GraphConvolutionSparse(input_dim=self.input_dim,
                                             output_dim=FLAGS.hidden,
                                             adj=self.adj,
                                             features_nonzero=self.features_nonzero,
                                             act=tf.nn.relu,
                                             dropout=self.dropout,
                                             logging=self.logging)(self.inputs)
        self.noise = gaussian_noise_layer(self.hidden, 0.1)  # 用于给 self.hidden1 输出添加高斯噪声，标准差为 0.1。结果存储在 self.noise 中

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden,
                                       output_dim=FLAGS.dimension,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.noise)
        # self.z_mean = self.z_mean + self.hidden

        self.reconstructions = GravityDecoder(act=lambda x: x,
                                              normalize=FLAGS.normalize,
                                              logging=self.logging)([self.z_mean, self.normalized_pagerank_csr])

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise






