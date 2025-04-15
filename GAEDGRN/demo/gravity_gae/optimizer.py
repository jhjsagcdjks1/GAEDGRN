import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import SGD

flags = tf.app.flags
FLAGS = flags.FLAGS

'''
Disclaimer: the OptimizerAE and OptimizerVAE classes from this file
come from tkipf/gae original repository on Graph Autoencoder
'''


class OptimizerAE(object):
    """ Optimizer for non-variational autoencoders """
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
                                                     targets = labels_sub,
                                                     pos_weight = pos_weight))
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)

        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = \
            tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.4), tf.int32),
                     tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
    def zero_grad(self):
        def zero_grad(self):
            # 获取所有模型参数
            all_vars = tf.trainable_variables()

            # 生成清零梯度的操作
            zero_ops = [var.assign(tf.zeros_like(var)) for var in all_vars]

            # 开始 TensorFlow 会话
            with tf.Session() as sess:
                # 执行清零梯度的操作
                sess.run(zero_ops)


class OptimizerAE2(object):
    """ Optimizer for non-variational autoencoders """

    def __init__(self, preds, labels, pos_weight, norm):
        self.preds = preds
        self.labels = labels
        self.pos_weight = pos_weight
        self.norm = norm

        # 构建损失函数
        self.cost = self.compute_cost()

        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.grads_and_vars = self.optimizer.compute_gradients(self.cost)
        self.opt_op = self.optimizer.apply_gradients(self.grads_and_vars)

        # 计算准确率
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(self.preds), 0.4), tf.int32),
                                           tf.cast(self.labels, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def compute_cost(self):
        # 计算加权交叉熵损失
        return self.norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=self.preds,
                                                     targets=self.labels,
                                                     pos_weight=self.pos_weight))

    def zero_grad(self):
        # 清零梯度的操作
        all_vars = tf.trainable_variables()
        zero_ops = [var.assign(tf.zeros_like(var)) for var in all_vars]
        with tf.Session() as sess:
            sess.run(zero_ops)


class OptimizerGAE(object):
    def __init__(self, preds, labels, pos_weight, norm, d_real, d_fake):
        preds_sub = preds
        labels_sub = labels

        self.real = d_real

        # Discrimminator Loss
        self.dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real), logits=self.real, name='dclreal'))

        self.dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake, name='dcfake'))
        self.dc_loss = self.dc_loss_fake + self.dc_loss_real

        # Generator loss
        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake, name='gl'))

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.generator_loss = generator_loss + self.cost
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        # 用于获取所有可训练的变量，并将它们按照名称中包含 'dc_' 或 'e_' 的模式分别存储在 dc_var 和 en_var 列表中
        all_variables = tf.trainable_variables()  # 返回所有可训练的变量的列表
        dc_var = [var for var in all_variables if
                  'dc_' in var.name]  # 通过列表推导式，从 all_variables 中筛选出名称包含 'dc_' 的变量。这通常用于鉴别器（Discriminator）相关的变量。
        en_var = [var for var in all_variables if 'e_' in var.name]  # 通过列表推导式，从 all_variables 中筛选出名称包含 'e_' 的变量。
        # 通过 dc_var 和 en_var，可以分别获得鉴别器和生成器（或编码器）中的所有可训练变量，以便在训练过程中分别应用不同的优化器或进行不同的操作。

        with tf.variable_scope(tf.get_variable_scope()):
            self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                                  beta1=0.9, name='adam1').minimize(self.dc_loss,
                                                                                                    var_list=dc_var)  # minimize(dc_loss_real, var_list=dc_var)

            self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                              beta1=0.9, name='adam2').minimize(self.generator_loss,
                                                                                                var_list=en_var)

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)


class OptimizerVAE(object):
    """ Optimizer for variational autoencoders """
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
                                                     targets = labels_sub,
                                                     pos_weight = pos_weight))
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * \
                  tf.reduce_mean(tf.reduce_sum(1 \
                                               + 2 * model.z_log_std \
                                               - tf.square(model.z_mean) \
                                               - tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = \
            tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                              tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))