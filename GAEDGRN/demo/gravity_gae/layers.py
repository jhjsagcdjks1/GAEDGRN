from __future__ import division
from gravity_gae.initializations import weight_variable_glorot
import tensorflow as tf
from scipy.sparse import csr_matrix

flags = tf.app.flags
FLAGS = flags.FLAGS
_LAYER_UIDS = {} # Global unique layer ID dictionary for layer name assignment

"""
Disclaimer: functions and classes defined from lines 16 to 122 in this file 
come from tkipf/gae original repository on Graph Autoencoders. Functions and 
classes from line 125 correspond to Source-Target and Gravity-Inspired 
decoders from our paper.
"""

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """ Graph convolution layer """
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")  # 一个权重初始化函数，用于初始化权重矩阵
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs



class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs"""   # features_nonzero 输入特征中非零值的索引
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Symmetric inner product decoder layer"""
    def __init__(self, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


class SourceTargetInnerProductDecoder(Layer):
    """Source-Target asymmetric decoder for directed link prediction."""
    def __init__(self, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(SourceTargetInnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)      # 这行代码对输入数据进行了一定比例的丢弃操作（dropout）
        # Source vector = First half of embedding vector
        inputs_source = inputs[:, 0:int(FLAGS.dimension/2)]   # 将输入数据分割为两部分。第一部分是源向量（embedding vector 的前半部分），使用了数组切片操作，获取了输入数据的前半部分
        # Target vector = Second half of embedding vector
        inputs_target = inputs[:, int(FLAGS.dimension/2):FLAGS.dimension]  # 第二部分是目标向量（embedding vector 的后半部分），同样使用了数组切片操作，获取了输入数据的后半部分。
        # Source-Target decoding
        x = tf.matmul(inputs_source, inputs_target, transpose_b = True) # 这行代码执行了源-目标解码操作。它使用矩阵乘法（内积操作）计算了源向量和目标向量之间的乘积。通过设置 transpose_b = True，实现了输入的转置，使得目标向量在计算乘积时被转置。
        x = tf.reshape(x, [-1])      # 将计算得到的矩阵乘积结果重塑为一维张量。这一步将矩阵转换为一个扁平的向量。
        outputs = self.act(x)   # 对乘积结果应用激活函数 self.act，一般默认是 tf.nn.sigmoid。这个激活函数会对输入数据进行变换，生成最终的输出结果。
        return outputs

class GravityDecoder(Layer):
    """Gravity-Inspired asymmetric decoder for directed link prediction."""
    def __init__(self, normalize=False, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(GravityDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.normalize = normalize

    def _call(self, inputs):
        normalized_pagerank_csr = csr_matrix(inputs[1]).toarray().astype('float32') # 或者根据实际情况调整索引
        inputs = inputs[0]
        # inputs = tf.nn.dropout(inputs, 1-self.dropout)
        inputs = tf.nn.dropout(tf.cast(inputs, dtype=tf.float32), 1 - self.dropout)

        # Embedding vector = all dimensions on input except the last
        # Mass parameter = last dimension of input
        if self.normalize:
            inputs_z = tf.math.l2_normalize(inputs[:, 0:(FLAGS.dimension - 1)],
                                            axis = 1)  # 对除最后一维以外的所有维度进行 L2 归一化操作
        else:
            inputs_z = inputs[:, 0:(FLAGS.dimension - 1)]  # 提取 inputs 中的所有行，但是只保留每行的前01 FLAGS.dimension - 1 个元素
        # Get pairwise node distances in embedding
        dist = pairwise_distance(inputs_z, FLAGS.epsilon)  # 计算嵌入中成对节点的距离（即节点间的距离）
        # Get mass parameter
        inputs_mass = inputs[:,(FLAGS.dimension - 1):FLAGS.dimension]  # 通过矩阵操作获取质量参数
        if normalized_pagerank_csr is not None:
            alpha = 0.1# 权重给 Pagerank
            beta = 0.9# 权重给节点特征
            if alpha != 0:
             # 对 pagerank_scores 的每一列进行加权平均
              inputs_mass = (alpha * normalized_pagerank_csr)+(beta * inputs_mass)
              inputs_mass = tf.math.l2_normalize(inputs_mass, axis=1)
        # inputs_mass = csr_matrix(inputs_mass).astype('float32')
        mass = tf.matmul(tf.ones([tf.shape(inputs_mass)[0],1]),tf.transpose(inputs_mass))  # tf.ones([tf.shape(inputs_mass)[0], 1]) 创建一个与 inputs_mass 行数相同，列数为 1 的矩阵，所有元素都为 1。这是一个列向量。
        # Gravity-Inspired decoding  对输出进行重塑操作（将其转换为一维向量）
        outputs = mass - tf.scalar_mul(FLAGS.lamb, tf.log(dist))  # 将上述结果中的每个元素乘以 FLAGS.lamb  tf.log(dist): 计算 dist 的每个元素的自然对数（以e为底）
        outputs = tf.reshape(outputs,[-1])
        outputs = self.act(outputs)
        return outputs

class GravityInspiredDecoder(Layer):
    """Gravity-Inspired asymmetric decoder for directed link prediction."""
    def __init__(self, normalize=False, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(GravityInspiredDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        self.normalize = normalize

    def _call(self, inputs):
        # inputs = tf.nn.dropout(inputs, 1-self.dropout)
        inputs = tf.nn.dropout(tf.cast(inputs, dtype=tf.float32), 1 - self.dropout)

        # Embedding vector = all dimensions on input except the last
        # Mass parameter = last dimension of input
        if self.normalize:
            inputs_z = tf.math.l2_normalize(inputs[:,0:(FLAGS.dimension - 1)],
                                            axis = 1)  # 对除最后一维以外的所有维度进行 L2 归一化操作
        else:
            inputs_z = inputs[:, 0:(FLAGS.dimension - 1)]  # 提取 inputs 中的所有行，但是只保留每行的前 FLAGS.dimension - 1 个元素
        # Get pairwise node distances in embedding
        dist = pairwise_distance(inputs_z, FLAGS.epsilon)  # 计算嵌入中成对节点的距离（即节点间的距离）
        # Get mass parameter
        inputs_mass = inputs[:,(FLAGS.dimension - 1):FLAGS.dimension]  # 通过矩阵操作获取质量参数
        mass = tf.matmul(tf.ones([tf.shape(inputs_mass)[0],1]),tf.transpose(inputs_mass))  # tf.ones([tf.shape(inputs_mass)[0], 1]) 创建一个与 inputs_mass 行数相同，列数为 1 的矩阵，所有元素都为 1。这是一个列向量。
        # Gravity-Inspired decoding  对输出进行重塑操作（将其转换为一维向量）
        outputs = mass - tf.scalar_mul(FLAGS.lamb, tf.log(dist))  # 将上述结果中的每个元素乘以 FLAGS.lamb  tf.log(dist): 计算 dist 的每个元素的自然对数（以e为底）
        outputs = tf.reshape(outputs,[-1])
        outputs = self.act(outputs)
        return outputs

def pairwise_distance(X, epsilon):  # 计算节点对之间的成对距离
    """ Computes pairwise distances between node pairs
    :param X: n*d embedding matrix
    :param epsilon: add a small value to distances for numerical stability
    :return: n*n matrix of squared euclidean distances
    """
    # 计算每个节点的平方欧氏范数，得到 n*1 的矩阵
    x1 = tf.reduce_sum(X * X, 1, True)
    # 计算嵌入矩阵 X 之间的点积，得到 n*n 的矩阵
    x2 = tf.matmul(X, tf.transpose(X))
    # Add epsilon to distances, avoiding 0 or too small distances leading to
    # numerical instability in gravity decoder due to logarithms # 对数引起的重力解码器的数值不稳定性
    return x1 - 2 * x2 + tf.transpose(x1) + epsilon