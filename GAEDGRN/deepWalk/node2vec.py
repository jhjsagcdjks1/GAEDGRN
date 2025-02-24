import pandas as pd

# 要写入的列表数据
my_list = ['a', 'b', 'c', 'd']

# 创建空的DataFrame
df = pd.DataFrame(columns=['col1', 'col2', 'col3', 'col4', 'col5'])

# 将列表数据写入第五列
df['col5'] = my_list

# 将DataFrame写入CSV文件
df.to_csv('data.csv', index=False)
