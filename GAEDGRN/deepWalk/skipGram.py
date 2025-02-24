import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, input_feat_dim, num_nodes):
        super(SkipGram, self).__init__()
        self.num_nodes = num_nodes
        """ word embeddings """
        self.word_embedding = torch.nn.Embedding(num_nodes, input_feat_dim)   # : 定义了一个词嵌入的 Embedding 层，其参数是节点数 num_nodes 和词嵌入的维度 input_feat_dim
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        torch.nn.init.xavier_uniform_(self.word_embedding.weight)             # 使用 Xavier 均匀分布初始化方法对词嵌入的权重进行初始化
        """ context embeddings"""
        self.context_embedding = torch.nn.Embedding(num_nodes, input_feat_dim) # 定义了一个上下文嵌入的 Embedding 层，其参数同样是节点数 num_nodes 和词嵌入的维度 input_feat_dim
        # initialize the weights with xavier uniform (Glorot, X. & Bengio, Y. (2010))
        torch.nn.init.xavier_uniform_(self.context_embedding.weight)           # 同样使用 Xavier 均匀分布初始化方法对上下文嵌入的权重进行初始化

    def get_input_layer(self, word_idx):
        x = torch.zeros(self.num_nodes).float()                              # 创建一个长度为 self.num_nodes 的全零张量，数据类型为 float，用于表示输入向量
        x[word_idx] = 1.0                                                    #  将索引为 word_idx 的位置设为 1.0。这一步将输入向量中特定索引位置上的值设置为 1.0，而其他位置保持为 0。
        return x

    def forward(self, node, context_positions, neg_sample=False):

        embed_word = self.word_embedding(node)  # 1 * emb_size                                       # 用词嵌入模型得到节点对应的嵌入表示
        embed_context = self.context_embedding(context_positions)  # n * emb_size                    # 获取上下文位置的嵌入表示
        score = torch.matmul(embed_context, embed_word.transpose(dim0=1, dim1=0))  # score = n * 1   # 计算得分，通常表示节点与上下文之间的相关性

        # following is an example of something you can only do in a framework that allows
        # dynamic graph creation
        if neg_sample:                                                                   # 如果 neg_sample 参数为 True，则将 score 取反
            score = -1 * score
        obj = -1 * torch.sum(F.logsigmoid(score))                                        # 返回计算得到的损失值 obj

        return obj
