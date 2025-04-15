import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix

# 读取 CSV 文件
file_path = '../Benchmark Dataset/Non-Specific Dataset/hESC/TFs+500/BL--ExpressionData.csv'  # 基因表达矩阵的 CSV 文件路径
df = pd.read_csv(file_path)

# 提取基因名称（第一行）
gene_names = df.columns.tolist()

# 去掉第一行和第一列
data_matrix = df.iloc[1:, 1:].values

# 转置矩阵（基因作为行，样本作为列）
transposed_matrix = data_matrix.T

# 将矩阵转换为稀疏矩阵（CSR 格式）
sparse_expression_matrix = csr_matrix(transposed_matrix)

# 获取文件路径的前三个文件夹名称
folder_path = os.path.dirname(file_path)
folder_names = folder_path.split(os.sep)[-3:]

# 拼接文件名
npz_filename = f"{'_'.join(folder_names)}.npz"

# 读取 BL-network 文件（假设为 CSV 格式，每行一个基因对）
bl_network_file = './Benchmark Dataset/Non-Specific Dataset/hESC/TFs+500/BL--network.csv'  # 替换为你的 BL-network 文件路径
bl_network = pd.read_csv(bl_network_file, header=None)

# 构建邻接矩阵
num_genes = len(gene_names)
adjacency_matrix = np.zeros((num_genes, num_genes))

# 创建基因到索引的映射
gene_to_index = {gene: idx for idx, gene in enumerate(gene_names)}

# 遍历 BL-network 文件中的每对基因
for _, row in bl_network.iterrows():
    gene1 = row[0]
    gene2 = row[1]
    
    # 检查基因是否在基因列表中
    if gene1 in gene_to_index and gene2 in gene_to_index:
        idx1 = gene_to_index[gene1]
        idx2 = gene_to_index[gene2]
        
        # 设置邻接矩阵的值为 1
        adjacency_matrix[idx1, idx2] = 1
        adjacency_matrix[idx2, idx1] = 1  # 如果是无向图

# 将邻接矩阵转换为稀疏矩阵（CSR 格式）
sparse_adjacency_matrix = csr_matrix(adjacency_matrix)

# 保存为 .npz 文件
np.savez(
    npz_filename,
    expr_data=sparse_expression_matrix.data,
    expr_indices=sparse_expression_matrix.indices,
    expr_indptr=sparse_expression_matrix.indptr,
    expr_shape=sparse_expression_matrix.shape,
    adj_data=sparse_adjacency_matrix.data,
    adj_indices=sparse_adjacency_matrix.indices,
    adj_indptr=sparse_adjacency_matrix.indptr,
    adj_shape=sparse_adjacency_matrix.shape
)

print(f"稀疏矩阵已保存为 {npz_filename}")