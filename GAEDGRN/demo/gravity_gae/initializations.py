import numpy as np
import tensorflow as tf

"""
Disclaimer: the weight_variable_glorot function from this file comes from 
tkipf/gae original repository on Graph Autoencoders
"""

def weight_variable_glorot(input_dim, output_dim, name = ""):
    """
    Create a weight variable with Glorot&Bengio (AISTATS 2010) initialization
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval = -init_range,
                                maxval = init_range, dtype = tf.float32)   #  tf.random_uniform 从均匀分布中选择范围在 [-init_range, init_range] 之间的随机值，用于初始化权重矩阵的参数
    return tf.Variable(initial, name = name)