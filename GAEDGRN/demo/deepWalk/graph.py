"""Graph utilities."""

import logging
import tensorflow as tf
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product, permutations
from scipy.io import loadmat
from scipy.sparse import issparse
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('p', 1, "Return parameter")
flags.DEFINE_float('q', 10, 'In-out parameter')

logger = logging.getLogger("deepwalk")

__author__ = "Bryan Perozzi"
__email__ = "bperozzi@cs.stonybrook.edu"

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


class Graph(defaultdict):
    """Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""

    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def subgraph(self, nodes={}):
        subgraph = Graph()                                       # 创建一个新的空的 Graph 对象，用于存储子图的信息

        for n in nodes:                                          # 遍历传入的节点字典中的每个节点
            if n in self:                                        # 检查节点 n 是否存在于原始图中
                subgraph[n] = [x for x in self[n] if x in nodes] # 如果节点 n 存在于原始图中，那么将筛选出该节点的邻居节点，并且这些邻居节点也必须在传入的节点字典中。然后将这些邻居节点加入到 subgraph 中，构建子图

        return subgraph

    def make_undirected(self):                               # 将图变为无向图。即将所有边转换为双向边，以确保图中的每条边都是双向连接的

        t0 = time()                                          # 记录操作开始的时间

        for v in self.keys():                                # 遍历图中的节点,可能是图数据结构中存储节点的方式，返回图中的所有节点。
            for other in self[v]:                            # 对于每个节点 v，遍历与其相关联的节点 other
                if v != other:                               # 确保不处理自身到自身的边，避免重复添加相同的边
                    self[other].append(v)                    # 将当前节点 v 添加到节点 other 的连接列表中，以确保双向连接。

        t1 = time()                                          # 记录操作结束的时间
        logger.info('make_directed: added missing edges {}s'.format(t1 - t0)) # 记录操作完成时所花费的时间

        self.make_consistent()                                # 调用 make_consistent 方法，可能是用于确保图数据结构的一致性。
        return self                                           # 返回更新后的图

# 确保图数据结构中每个节点的连接（邻居节点）是唯一的且有序的，同时移除重复连接
    def make_consistent(self):
        t0 = time()                                           # 记录操作开始的时间
        for k in iterkeys(self):                              # 遍历图中的每个节点
            self[k] = list(sorted(set(self[k])))              #  针对节点 k 的连接列表，进行去重、排序操作

        t1 = time()                                           #  记录操作结束的时间
        logger.info('make_consistent: made consistent in {}s'.format(t1 - t0)) # 记录操作完成所花费的时间

        self.remove_self_loops()                               # 调用 remove_self_loops 方法，可能用于删除自循环的连接。

        return self
# 用于在图数据结构中移除自循环（self-loops），即节点连接到自身的边

    def remove_self_loops(self):

        removed = 0                                               # 初始化一个计数器，用于记录移除的自循环数量
        t0 = time()                                               # 记录操作开始的时间

        for x in self:
            if x in self[x]:                                      # 检查节点 x 是否连接到自身
                self[x].remove(x)                                 # 如果存在自循环，将节点 x 与自己的连接移除
                removed += 1                                      # 更新移除自循环的计数器

        t1 = time()

        logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1 - t0)))
        return self

# 如果在遍历过程中找到任何节点连接到自身，则返回 True，否则返回 False

    def check_self_loops(self):                       # check_self_loops 方法似乎用于检查图数据结构中是否存在自循环（self-loops）
        for x in self:                                #  遍历图中的每个节点
            for y in self[x]:                         # 对于每个节点 x 的连接节点 y 进行遍历
                if x == y:                            # 检查是否存在节点连接到自身的情况，如果连接到自身则返回True
                    return True

        return False

# 检查图数据结构中是否存在连接两个特定节点 v1 和 v2 的边，如果 v1 节点与 v2 节点直接相连，或者 v2 节点与 v1 节点直接相连，那么方法返回 True；否则返回 False。

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:  # 检查节点 v1 是否与节点 v2 直接相连，或者节点 v2 是否与节点 v1 直接相连
            return True
        return False
# 如果提供了一个节点列表，方法将返回一个字典，其中键是节点，对应的值是其度数
# 如果只提供了一个节点，则方法将返回该节点的度数，即连接到该节点的边的数量
    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        "Returns the number of nodes in the graph" # 返回图形中的节点数
        return len(self)

    def number_of_edges(self):
        "Returns the number of nodes in the graph"
        return sum([self.degree(x) for x in self.keys()]) / 2

    def number_of_nodes(self):
        "Returns the number of nodes in the graph"
        return self.order()

    # def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):    # 这个方法模拟了在图结构中的随机游走过程，控制了重新开始的概率和游走的路径长度
    #     """ Returns a truncated random walk.
    #
    #         path_length: Length of the random walk.
    #         alpha: probability of restarts.
    #         start: the start node of the random walk.
    #     """
    #     G = self               # 将当前图赋值给变量 G
    #     if start:              # 随机游走的起始节点
    #         path = [start]     # 如果提供了起始节点 start，将其作为游走的起点，否则随机选择一个节点作为起点
    #     else:
    #         # Sampling is uniform w.r.t V, and not w.r.t E
    #         path = [rand.choice(list(G.keys()))]
    #
    #     while len(path) < path_length:       # 在游走的过程中，依次执行以下步骤直到达到指定的路径长度或无法再继续游走
    #         cur = path[-1]                   # 首先，取当前路径 path 的最后一个节点作为当前节点 cur
    #         if len(G[cur]) > 0:              # 如果当前节点 cur 有邻居节点
    #             if rand.random() >= alpha:   # 若随机数大于等于 alpha，以一定概率选择一个邻居节点作为下一个节点加入路径
    #                 path.append(rand.choice(G[cur]))
    #             else:
    #                 path.append(path[0])     # 否则，以概率 alpha 重新开始游走，将路径重新回到起始节点
    #         else:
    #             break                        # 如果当前节点 cur 无邻居节点，游走中断，结束游走
    #     return [str(node) for node in path]  # 返回游走过程中的节点序列（路径）的字符串形式表示
    def random_walk(self, path_length, p=1, q=1, rand=random.Random(), start=None):
        """ 返回一个带有Node2Vec策略的截断的随机游走。

            path_length: 随机游走的长度。
            p: 回到先前节点的概率。
            q: 跳到远离先前节点的概率。
            rand: 随机数生成器。
            start: 随机游走的起始节点。
        """
        G = self
        if start:
            path = [start]
        else:
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]                        # 首先，取当前路径 path 的最后一个节点作为当前节点 cur
            if len(G[cur]) > 0:                   # 如果当前节点 cur 有邻居节点
                next_node = self._calculate_next_node(cur, path, p, q, rand)
                path.append(next_node)
            else:
                break

        return [str(node) for node in path]

    def _calculate_next_node(self, cur, path, p, q, rand):
        G = self
        neighbors = list(G[cur])           # 获取当前节点 cur 的邻居节点列表。
        if len(path) > 1:                  # 检查路径 path 的长度，如果路径长度大于1，则将先前节点 prev 设置为路径中的倒数第二个节点，否则将 prev 设置为 None
            prev = path[-2]
        else:
            prev = None
        weights = []                       # 用于存储每个邻居节点的权重
        for neighbor in neighbors:         # 循环遍历每个邻居节点 neighbor
            if neighbor == prev:           # 如果 neighbor 等于先前节点 prev，则将权重设为 1/p，表示回到先前节点的概率。
                weights.append(1 / p)
            elif neighbor in G[prev]:      # 如果 neighbor 存在于先前节点 prev 的邻居中，将权重设为 1，表示保持在相邻节点的概率。
                weights.append(1)
            else:
                weights.append(1 / q)      # 将权重设为 1/q，表示跳到远离先前节点的概率

        probabilities_sum = sum(weights)   # 计算权重的总和，用于将权重转化为概率。
        probabilities = [weight / probabilities_sum for weight in weights]  # 将权重归一化为概率，确保它们的总和为1。

        return rand.choices(neighbors, weights=probabilities)[0]     # 使用 rand.choices 方法根据计算得到的概率从邻居节点中选择一个节点，并返回选择的节点


# TODO add build_walks in here

def build_deepwalk_corpus(G, num_paths, path_length, rand=random.Random(0)):
    walks = []                         # 初始化一个空列表，用于存储生成的路径

    nodes = list(G.nodes())            # 获取图中所有节点，并存储在列表 nodes 中

    for cnt in range(num_paths):      # 循环生成指定数量的路径
        rand.shuffle(nodes)           # 打乱节点列表，以确保随机性
        for node in nodes:             # 遍历打乱后的节点列表
            walks.append(G.random_walk(path_length, p=FLAGS.p, q=FLAGS.q, rand=rand, start=node))
    # walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))  # 对每个节点执行随机游走，将生成的路径添加到 walks 中

    return walks                      # 返回由多个随机游走路径组成的数据集 walks

# 生成用于 DeepWalk 算法训练的数据集，并使用生成器（yield）逐步返回每个随机游走的路径

def build_deepwalk_corpus_iter(G, num_paths, path_length, alpha=0,
                               rand=random.Random(0), chunk=0, nodes=None):

    nodes = nodes[chunk * num_paths: (chunk + 1) * num_paths]                 # 选择并存储指定区块范围内的节点列表
    for node in nodes:   # 对选定的节点进行迭代处理
        yield G.random_walk(path_length, p=FLAGS.p, q=FLAGS.q, rand=rand, start=node)
    # yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)  # 使用随机游走方法生成每个节点的路径，并通过 yield 逐个返回生成的路径

# 该函数返回一个完全图，其中包含了指定数量的节点，并且每个节点都与其他节点直接相连。这是一个特定节点数量的完全图的表示

def clique(size):
    return from_adjlist(permutations(range(1, size + 1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
# grouper 函数能够帮助将任意可迭代对象分割成指定大小的子组，用于处理数据分组的情况，将可迭代对象按指定大小分割成子组
def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue) # 当最后一个子组不足 n 时，可以使用 padvalue 参数指定的值进行填充，使得所有子组的长度相等。

# 函数将文件中非空且不以 # 开头的行解析为邻接列表的形式，以用于表示图的连接关系
def parse_adjacencylist(f):
    adjlist = []                                             # 初始化一个空列表，用于存储邻接列表的结果
    for l in f:                                              # 遍历文件的每一行内容
        if l and l[0] != "#":                                # 检查行是否非空且不以 # 开头
            introw = [int(x) for x in l.strip().split()]     # 将字符串列表中的每个元素转换为整数
            row = [introw[0]]                                # 将字符串列表中的每个元素转换为整数
            row.extend(set(sorted(introw[1:])))              # 将排序且去重后的整数列表添加到新的行中
            adjlist.extend([row])                            # 将这一行的邻接列表信息添加到最终的邻接列表中

    return adjlist
# 这个 parse_adjacencylist_unchecked 函数与之前的 parse_adjacencylist 函数非常相似，也用于解析邻接列表文件，将文件中的数据转换为邻接列表形式。
# 然而，它没有进行排序和去重操作，而是直接将行的内容转换为整数列表，并将其添加到邻接列表。

def parse_adjacencylist_unchecked(f):
    adjlist = []                                                     # 初始化一个空列表，用于存储邻接列表的结果。
    for l in f:                                                      # 遍历文件的每一行内容
        if l and l[0] != "#":                                        # 检查行是否非空且不以 # 开头
            adjlist.extend([[int(x) for x in l.strip().split()]])    #  直接将转换后的整数列表作为子列表添加到邻接列表中

    return adjlist

# 用于从邻接列表文件加载图数据，然后创建图对象。这个函数还提供了一些可选的参数，可以选择在加载过程中将图转换为无向图、控制读取文件的块大小，并选择是否对邻接列表进行安全检查。
def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):
    if unchecked:
        parse_func = parse_adjacencylist_unchecked
        convert_func = from_adjlist_unchecked
    else:
        parse_func = parse_adjacencylist
        convert_func = from_adjlist

    adjlist = []

    t0 = time()

    total = 0
    with open(file_) as f:           # 打开文件，并使用 grouper 函数按块读取文件，并使用选定的解析函数解析这些块，然后将结果添加到 adjlist 中
        for idx, adj_chunk in enumerate(map(parse_func, grouper(int(chunksize), f))):
            adjlist.extend(adj_chunk)
            total += len(adj_chunk)  # 记录总边数和块的数量，并记录解析时间

    t1 = time()

    logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1 - t0))

    t0 = time()
    G = convert_func(adjlist)     # 计时并开始转换边缘列表为图对象
    t1 = time()

    logger.info('Converted edges to graph in {}s'.format(t1 - t0))

    if undirected:
        t0 = time()
        G = G.make_undirected()
        t1 = time()
        logger.info('Made graph undirected in {}s'.format(t1 - t0))

    return G

# 从文件中的边缘列表加载图数据，并可以选择是否将图转换为无向图
def load_edgelist(file_, undirected=True):
    G = Graph()
    with open(file_) as f:
        for l in f:                                 # 遍历文件的每一行
            x, y = l.strip().split()[:2]            # 将行内容按空格分割并取前两个元素为 x 和 y
            x = int(x)                              #  将 x 和 y 转换为整数
            y = int(y)
            G[x].append(y)                          # 将节点 x 的邻居节点 y 添加到自定义图对象 G 的邻接列表中
            if undirected:
                G[y].append(x)                      # 如果指定了无向图，也将节点 y 的邻居节点 x 添加到图对象 G 的邻接列表中

    G.make_consistent()
    return G

#  CSR 矩阵表示的边缘列表加载图数据
def load_edgelist_from_csr_matrix(adjList, undirected=False):
    """
    Added a utility function to load Graph from a sparse matrix
    - vvaibhav@cs.cmu.edu
    """
    G = Graph()
    for x, ads in enumerate(adjList.tolil().rows):   # 使用 CSR 矩阵的 tolil() 方法获取每个节点的邻接列表 enumerate(adjList.tolil().rows) 在迭代过程中返回每个节点的索引 x 以及其邻接列表 ads
        for y in ads:                                # 遍历每个节点 x 的邻居节点 y
            G[x].append(y)                           # 将节点 x 的邻居节点 y 添加到自定义图对象 G 的邻接列表中
            if undirected:
                G[y].append(x)

    G.make_consistent()                              # 调用 make_consistent 方法，确保图对象的一致性
    return G

# 从MATLAB文件中加载网络数据，并转换为自定义的图对象 Graph
def load_matfile(file_, variable_name="network", undirected=False):
    mat_varables = loadmat(file_)                  # 使用 loadmat 函数加载MATLAB文件，获取其中的变量
    mat_matrix = mat_varables[variable_name]       # 从MATLAB变量中选择指定名称的矩阵

    return from_numpy(mat_matrix, undirected)      # 调用 from_numpy 函数，将MATLAB中的矩阵转换为自定义图对象 Graph


def from_networkx(G_input, undirected=True):        # 将提供的 NetworkX 图对象转换为自定义的图对象 Graph
    G = Graph()

    for idx, x in enumerate(G_input.nodes_iter()):  # 遍历 NetworkX 图对象中的节点
        for y in iterkeys(G_input[x]):              # 遍历每个节点的邻居节点
            G[x].append(y)                          #  将 NetworkX 图中节点 x 的邻居节点 y 添加到自定义图对象 G 的邻接列表中

    if undirected:
        G.make_undirected()

    return G


def from_numpy(x, undirected=False):
    G = Graph()

    if issparse(x):                                              # 检查输入的矩阵是否为稀疏矩阵
        cx = x.tocoo()                                           # 将稀疏矩阵转换为坐标格式
        for i, j, v in zip(cx.row, cx.col, cx.data):             # 遍历坐标格式中的行、列和值。
            G[i].append(j)                                       # 将稀疏矩阵中的非零值对应的行和列信息添加到图对象 G 中的邻接列表中
    else:
        raise Exception("Dense matrices not yet supported.")

    if undirected:
        G.make_undirected()                                     # 如果指定了创建无向图，则调用 make_undirected 方法将图转换为无向图

    G.make_consistent()                                         # 调用 make_consistent 方法，确保图对象的一致性
    return G

# 根据提供的邻接列表数据创建一个图对象，对邻接节点进行排序和去重处理
def from_adjlist(adjlist):
    G = Graph()

    for row in adjlist:
        node = row[0]
        neighbors = row[1:]                       # 获取除第一个元素外的其他元素，作为节点 node 的邻居
        G[node] = list(sorted(set(neighbors)))    # 将排序和去重后的节点 node 的邻居关系添加到图对象 G 中

    return G

# 函数创建图对象，并使用提供的邻接列表数据，但不对邻接列表进行排序和去重操作
def from_adjlist_unchecked(adjlist):
    G = Graph()              # 创建一个图对象 G

    for row in adjlist:      # 遍历邻接列表中的每一行
        node = row[0]        # 获取行中的第一个元素，作为节点
        neighbors = row[1:]  # 获取除第一个元素外的其他元素，作为节点 node 的邻居
        G[node] = neighbors  # 将节点 node 和其邻居关系添加到图对象 G 中

    return G

