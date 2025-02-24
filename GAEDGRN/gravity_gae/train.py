from __future__ import division
from __future__ import print_function


from gravity_gae.evaluation import compute_scores
from gravity_gae.input_data import load_data
from gravity_gae.model import *
from gravity_gae.optimizer import OptimizerAE, OptimizerVAE
from gravity_gae.preprocessing import *
from torch import optim
import os
import tensorflow as tf
import time
import random
import torch
from deepWalk.graph import load_edgelist_from_csr_matrix, build_deepwalk_corpus_iter, build_deepwalk_corpus
from deepWalk.skipGram import SkipGram
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import numpy as np



SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

flags = tf.app.flags
FLAGS = flags.FLAGS

# Select graph dataset
flags.DEFINE_string('dataset', '../demo', 'Name of the graph dataset')

# Select machine learning task to perform on graph
flags.DEFINE_string('task', 'task_1', 'Name of the link prediction task')

# Model parameters
flags.DEFINE_float('alpha', 0.1, '随机游走重启动的概率')
flags.DEFINE_integer('dw', 1, "whether to use deepWalk regularization, 0/1")  # 这个必须为1
flags.DEFINE_integer('walk_length', 9, 'Length of the random walk started at each node') #在每个节点开始的随机行走的长度
flags.DEFINE_integer('window_size', 3, 'Window size of skipgram model.')  # skipgram模型的窗口大小
flags.DEFINE_integer('number_walks', 5, 'Number of random walks to start at each node')  # 在每个节点开始的随机行走次数
flags.DEFINE_integer('full_number_walks', 1, 'Number of random walks from each node')
flags.DEFINE_float('lr_dw', 0.001, 'Initial learning rate for regularization.')
flags.DEFINE_integer('context', 0, "whether to use context nodes for skipgram")
flags.DEFINE_integer('ns',  0, "whether to use negative samples for skipgram")
flags.DEFINE_float('dropout', 0.1,'Dropout rate (1 - keep probability).')  # 参数的意义是 "丢弃率（1 - 保留概率）"。即，1 表示保留所有神经元，0 表示丢弃所有神经元。
flags.DEFINE_integer('epochs', 100, 'Number of epochs in training.')
flags.DEFINE_boolean('features', True, 'Include node features or not in GCN')
flags.DEFINE_float('lamb', 1.0, 'lambda parameter from Gravity AE/VAE models \
                                as introduced in section 3.5 of paper, to \
                                balance mass and proximity terms')  # 来自 Gravity AE/VAE 模型的 lambda 参如论文第 3.5 节所述，至平衡质量和邻近度项
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate (with Adam)')
flags.DEFINE_integer('hidden', 256, 'Number of units in GCN hidden laye.')
flags.DEFINE_integer('dimension', 256, 'Dimension of GCN output: \
- equal to embedding dimension for standard AE/VAE and source-target AE/VAE \
- equal to (embedding dimension - 1) for gravity-inspired AE/VAE, as the \
last dimension captures the "mass" parameter tilde{m}')
flags.DEFINE_boolean('normalize', False, 'Whether to normalize embedding \
                                          vectors of gravity models')  # 是否规范化嵌入重力模型的向量 不加感觉结果更好
flags.DEFINE_float('epsilon', 0.1, 'Add epsilon to distances computations \
                                       in gravity models, for numerical \
                                       stability')  # 将 epsilon 添加到距离计算中在重力模型中，对于数值稳定性
# Experimental setup parameters
flags.DEFINE_integer('nb_run', 10, 'Number of model run + test')
flags.DEFINE_float('prop_val', 5., 'Proportion of edges in validation set \
                                   (for Task 1)')  # 验证边的比例
flags.DEFINE_float('prop_test', 10., 'Proportion of edges in test set \
                                      (for Tasks 1 and 2)')
flags.DEFINE_float('prop_train', 100., 'Proportion of edges in test set \
                                      (for Tasks 1 and 2)')
flags.DEFINE_boolean('validation', True, 'Whether to report validation \
                                           results  at each epoch (for \
                                           Task 1)')  # 是否报告每轮的结果
flags.DEFINE_boolean('verbose', True, 'Whether to print comments details.')

# 定义数据集路径列表
dataset_paths = [
    "SIRING Dateset hESC TFs+500",
    "SIRING Dateset hESC TFs+1000"
]
# 遍历数据集路径
for dataset_path in dataset_paths:
    print(f"Paths for dataset {dataset_path} ")
    # Lists to collect average results
    mean_roc = []
    mean_ap = []
    mean_time = []

    # Load graph dataset
    if FLAGS.verbose:
        print("Loading data...")
    adj_init, features, data_name= load_data(dataset=dataset_path)
    n_nodes, feat_dim = features.shape
    # The entire training process is repeated FLAGS.nb_run times
    # for i in range(FLAGS.nb_run):

    # Edge Masking: compute Train/Validation/Test set
    if FLAGS.verbose:
        print("Masking test edges...")


    adj, val_edges, val_edges_false, test_edges, test_edges_false = \
        mask_test_edges_general_link_prediction(adj_init, FLAGS.prop_test,
                                                FLAGS.prop_val)

    # Preprocessing and initialization
    if FLAGS.verbose:
        print("Preprocessing and Initializing...")
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    # Compute number of nodes
    num_nodes = adj.shape[0]
    # If features are not used, replace feature matrix by identity matrix
    if not FLAGS.features:
        features = sp.identity(adj.shape[0])

    if not FLAGS.features:
        features = sp.identity(adj.shape[0])


    # 转移到当前网页的概率
    p = 0.90

    # 表示网页间链接关系的转移矩阵
    # 假设 adj_orig 是一个 CSR 格式的稀疏矩阵
    # 获取 CSR 矩阵的行和列索引以及数据
    rows, cols, data = sp.find(adj_orig)

    # 使用这些信息创建一个密集的 numpy 数组
    a = np.zeros_like(adj_orig.toarray(), dtype=float)
    a[rows, cols] = data

    length = a.shape[1]  # 网页数量


    # 初始化转移概率矩阵（m）
    m = np.zeros((a.shape), dtype=float)

    # 根据链接结构计算转移概率
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            # 如果一个节点没有任何入链（Dead Ends），均匀分配概率
            if a[j].sum() == 0:
                a[j] = a[j] + np.array([1 / length] * length)

            m[i][j] = a[i][j] / (a[j].sum())

    # 初始化PageRank值的矩阵
    v = np.zeros((m.shape[0], 1), dtype=float)
    for i in range(m.shape[0]):
        v[i] = float(1) / m.shape[0]

    count = 0
    ee = np.array([[1 / length] * length]).reshape(length, -1)

    # 迭代计算PageRank值，循环100次
    for i in range(100):
        # 解决Spider Traps问题，将转移矩阵加上打开其他网页的概率 (1-p)
        v = p * np.dot(m, v) + (1 - p) * ee
        count += 1
        print("第{}次迭代".format(count))

    # 输出PageRank值
    print(v)


    # 对 NumPy 数组进行乘法操作
    pagerank_array = v * 1000
    # 归一化 PageRank 值
    normalized_pagerank = pagerank_array / np.max(pagerank_array)



    # 使用 np.squeeze 去除 normalized_pagerank 的单维度
    normalized_pagerank = np.squeeze(normalized_pagerank)
    # # 将 normalized_pagerank 转换为 CSR 矩阵
    normalized_pagerank_csr = csr_matrix(normalized_pagerank).tocsr()

    # 将 normalized_pagerank 转换为 CSR 矩阵
    normalized_pagerank_csr = normalized_pagerank_csr.T

    # 将稀疏矩阵转换为密集矩阵进行广播
    normalized_pagerank_csr = normalized_pagerank_csr.toarray()
    features = features.toarray()
    alpha = 0.1
    # alpha = 0.2 # 权重给 Pagerank
    beta = 1-alpha # 权重给节点特征

    # 对 pagerank_scores 的每一列进行加权平均
    features = (alpha * normalized_pagerank_csr) + (beta * features)
    features = csr_matrix(features)


    n_nodes, feat_dim = features.shape

    # Preprocessing on node features
    features = sparse_to_tuple(features)
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }
    for i in range(FLAGS.nb_run):
        # Create model
        model = None
        model = GravityGAEModelAE(placeholders, num_features,features_nonzero,normalized_pagerank_csr)
        # Optimizer (see tkipf/gae original GAE repository for details)
        pos_weight = 9
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0]
                                                    - adj.sum()) * 2)
        with tf.name_scope('optimizer'):
            # Optimizer for Non-Variational Autoencoders
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        # Before proceeding further, make the structure for doing deepWalk    在继续进行之前，请制作deepWalk的结构
        if FLAGS.dw == 1:
            print('Using deepWalk regularization...')
            G = load_edgelist_from_csr_matrix(adj_orig,
                                                  undirected=False)  # 如果 args.dw 参数的值为 1，则打印信息 'Using deepWalk regularization...'
            print("Number of nodes: {}".format(len(G.nodes())))  # 一个 CSR（压缩稀疏行）矩阵 adj_orig 中加载一个有向图数据结构。这里禁用了图的无向性，即认为图是有向的
            num_walks = len(G.nodes()) * FLAGS.number_walks  # 计算所需的随机游走数量
            print("Number of walks: {}".format(num_walks))  # 打印随机游走的总数
            data_size = num_walks * FLAGS.walk_length  # 计算数据大小，即总的数据量
            print("Data size (walks*length): {}".format(data_size))  # 打印数据大小

        # Normalization and preprocessing on adjacency matrix
        adj_norm = preprocess_graph(adj)
        adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))
        # 创建一个包含自环的邻接矩阵，并将其转换为元组形式，以便在后续的代码中使用。

        # Initialize TF session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        if FLAGS.dw == 1:
            sg = SkipGram(FLAGS.dimension, adj.shape[0])
            optimizer_dw = optim.Adam(sg.parameters(),
                                          lr=FLAGS.lr_dw)  # 初始化了一个 Adam 优化器 optimizer_dw，用于优化 SkipGram 模型的参数。学习率为 args.lr_dw。

                # Construct the nodes for doing random walk. Doing it before since the seed is fixed  构造用于执行随机行走的节点,种子固定后再做
            nodes_in_G = list(G.nodes())  # list() 函数将这个集合转换为列表，并将其赋值给变量 nodes_in_G
            chunks = len(nodes_in_G) // FLAGS.number_walks  # 定在进行随机游走时，将节点列表分割成多少个块，以便在不同的块上并行执行随机游走算法
            random.Random().shuffle(nodes_in_G)  # 对节点列表进行随机打乱


        # Model training
        if FLAGS.verbose:
            print("Training...")
        # Flag to compute total running time
        t_start = time.time()
        hidden_emb = None
        for epoch in range(FLAGS.epochs):
                t = time.time()
                model.fit()
                opt.zero_grad()
                # 根据随机游走的路径构建节点对，以用于训练 SkipGram 模型，其中每个节点对代表了中心节点与其周围的上下文节点
                if FLAGS.dw == 1:
                    sg.train()
                    if FLAGS.full_number_walks > 0:
                        walks = build_deepwalk_corpus(G, num_paths=FLAGS.full_number_walks,
                                                      path_length=FLAGS.walk_length,
                                                      rand=random.Random(SEED))  # 返回由多个随机游走路径组成的数据集 walks
                    else:
                        walks = build_deepwalk_corpus_iter(G, num_paths=FLAGS.number_walks,
                                                           path_length=FLAGS.walk_length, alpha=FLAGS.alpha,
                                                           rand=random.Random(SEED),
                                                           chunk=epoch % chunks,
                                                           nodes=nodes_in_G)  # 生成用于 DeepWalk 算法训练的数据集，并使用生成器（yield）逐步返回每个随机游走的路径
                    for walk in walks:  # 在遍历 walks（代表每次随机游走）中的每条路径时
                        if walk == ['None']:  # 检查 walk 是否为 None
                            continue  # 如果是，则结束当前循环，继续下一个 walk
                        if FLAGS.context == 1:
                            # Construct the pairs for predicting context node  构造用于预测上下文节点的对
                            # for each node, treated as center word   对于每个节点，视为中心词
                            curr_pair = (int(walk[center_node_pos]),
                                         [])  # 对于每个节点，代码创建一个初始节点对 curr_pair，其中第一个元素是当前路径中的节点索引（int(walk[center_node_pos])），第二个元素是一个空列表 []
                            for center_node_pos in range(len(walk)):
                                # for each window position 对于每个窗口位置
                                for w in range(-FLAGS.window_size, FLAGS.window_size + 1):
                                    context_node_pos = center_node_pos + w  # 它获取相对于当前中心词位置的上下文词的位置
                                    # make soure not jump out sentence 使soure不跳出句子
                                    if context_node_pos < 0 or context_node_pos >= len(
                                            walk) or center_node_pos == context_node_pos:  # 它检查计算得到的上下文词的位置是否有效，以确保不会超出路径的边界，并且上下文词不是中心词本身
                                        continue
                                    context_node_idx = walk[
                                        context_node_pos]  # 果 args.context 不等于 1，它假设路径的第一个节点是起始节点，然后创建一个节点对 curr_pair，
                                    curr_pair[1].append(
                                        int(context_node_idx))  # 其中第一个元素是路径的第一个节点索引（int(walk[0])），第二个元素是路径中除第一个节点之外的其他节点的索引
                        else:
                            # first item in the walk is the starting node
                            curr_pair = (int(walk[0]), [int(context_node_idx) for context_node_idx in walk[
                                                                                                      1:]])  # 首先取 walk 路径的第一个节点 walk[0]，将其作为中心词，并使用 int(walk[0]) 将其索引转换为整数类型。然后，它获取 walk 路径中除了第一个节点之外的其他节点作为上下文词。
                        # 进行负采样（Negative Sampling），在训练 SkipGram 模型时生成负样本节点集合。
                        if FLAGS.ns == 1:
                            neg_nodes = []
                            pos_nodes = set(walk)
                            while len(neg_nodes) < FLAGS.walk_length - 1:  # 当负样本节点列表 neg_nodes 的长度小于 args.walk_length - 1 时，进行负采样操作
                                rand_node = random.randint(0,
                                                           n_nodes - 1)  # 每次循环迭代中，随机生成一个介于 0 到 n_nodes - 1 之间的随机节点索引 rand_node
                                if rand_node not in pos_nodes:
                                    neg_nodes.append(rand_node)  # 将 rand_node 添加到负样本节点列表 neg_nodes 中
                            neg_nodes = torch.from_numpy(
                                np.array(neg_nodes)).long()  # 将负样本节点列表转换为 PyTorch 张量，并设置为长整型类型，以便在负采样损失计算时使用。

                        # Do actual prediction
                        src_node = torch.from_numpy(np.array([curr_pair[
                                                                  0]])).long()  # 通过将 curr_pair 中的第一个元素转换为 NumPy 数组，然后使用 torch.from_numpy() 转换为 PyTorch 张量。这个张量被设置为长整型（long()）类型
                        tgt_nodes = torch.from_numpy(np.array(curr_pair[1])).long()
                        optimizer_dw.zero_grad()  # 用于清空优化器的梯度，确保每次迭代的梯度不会叠加
                        log_pos = sg(src_node, tgt_nodes, neg_sample=False)  # 通过 SkipGram 模型 sg 计算了给定中心词和上下文词的对数概率
                        if FLAGS.ns == 1:
                            loss_neg = sg(src_node, neg_nodes, neg_sample=True)
                            loss_dw = log_pos + loss_neg
                        else:
                            loss_dw = log_pos  # 如果没有负采样（args.ns 不等于 1），loss_dw 只包含正样本损失值 log_pos。
                        # loss_dw.backward(retain_graph=True)  # 用于反向传播，计算损失值对模型参数的梯度。retain_graph=True 保留计算图，以便后续可能需要多次反向传播
                        cur_dw_loss = loss_dw.item()  # 用于获取当前步骤的总损失值
                        optimizer_dw.step()  # 用于根据计算得到的梯度更新模型参数，进行一步优化
                # Construct feed dictionary
                feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                # Weight update
                outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                            feed_dict=feed_dict)  # opt.opt_op 负责执行权重更新
                avg_cost = outs[1]
                if FLAGS.verbose:
                    if FLAGS.dw == 1:
                        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),"train_loss_dw=", "{:.5f}".format(cur_dw_loss),
                              "time=", "{:.5f}".format(time.time() - t))
                    else:
                        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),"time=", "{:.5f}".format(time.time() - t))
                # Validation (implemented for Task 1 only)
                if FLAGS.validation and FLAGS.task == 'task_1':
                    feed_dict.update({placeholders['dropout']: 0})
                    emb = sess.run(model.z_mean, feed_dict=feed_dict)
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    # val_test_edges = np.vstack((val_edges, test_edges))
                    # val_test_edges_false = np.vstack((val_edges_false, test_edges_false))
                    val_roc, val_ap = compute_scores( val_edges, val_edges_false, emb)
                    print("val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))

        # Compute total running time
        mean_time.append(time.time() - t_start)
        all_time = time.time() - t_start
        # Get embedding from model
        emb = sess.run(model.z_mean, feed_dict=feed_dict)


        # Test model
        if FLAGS.verbose:
            print("Testing model...")
        # Compute ROC and AP scores on test sets
        roc_score, ap_score = compute_scores(test_edges, test_edges_false, emb)
        print(roc_score)
        # Append to list of scores over all runs
        mean_roc.append(roc_score)
        mean_ap.append(ap_score)
    # Report final results
    print("\nTest results for", FLAGS.model,
          "model on", FLAGS.dataset, "on", FLAGS.task, "\n",
          "___________________________________________________\n")


    print("AUC scores\n", mean_roc)
    AUC_scores = np.mean(mean_roc)
    AUC_std = np.std(mean_roc)
    print("Mean AUC score: ", np.mean(mean_roc),
          "\nStd of AUC scores: ", np.std(mean_roc), "\n \n")
    print("AP scores \n", mean_ap)
    AP_scores = np.mean(mean_ap)
    AP_std = np.std(mean_ap)
    print("Mean AP score: ", np.mean(mean_ap),
          "\nStd of AP scores: ", np.std(mean_ap), "\n \n")

    print("Running times\n", mean_time)
    time_mean = np.mean(mean_time)
    print("Mean running time: ", np.mean(mean_time),
          "\nStd of running time: ", np.std(mean_time), "\n \n")
    import time
    current_time = time.time()
    local_time_struct = time.localtime(current_time)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time_struct)
    column_names = ['date_name', 'dropout', 'epochs', 'learning_rate', 'hidden', 'dimension'
        ,'normalize','AUC scores1','AUC scores2','AUC scores3','AUC scores4','AUC scores5','AUC scores6','AUC scores7','AUC scores8','AUC scores9','AUC scores10','AUC mean','AUC std','AP mean','AP std','Running times','alpha']
    resultspath = "../demo/result.csv"

    new_data = [data_name,FLAGS.dropout , FLAGS.epochs, FLAGS.learning_rate, FLAGS.hidden, FLAGS.dimension, FLAGS.normalize, mean_roc, AUC_scores, AUC_std, AP_scores, AP_std, time_mean, alpha]
    if os.path.exists(resultspath):
        with open(resultspath, mode='a', newline='') as file:
            np.savetxt(file, [new_data], delimiter=',', header=' ', comments='', fmt='%s')
    else:
        np.savetxt(resultspath, [new_data], delimiter=',', header=','.join(column_names), comments='', fmt='%s')
