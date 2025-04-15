import logging
from io import open
from os import path
from time import time
from multiprocessing import cpu_count
import random
from concurrent.futures import ProcessPoolExecutor
from collections import Counter

from six.moves import zip

from . import graph

logger = logging.getLogger("deepwalk")

__current_graph = None

# speed up the string encoding
__vertex2str = None

def count_words(file):
  """ Counts the word frequences in a list of sentences.   #统计句子列表中的单词频率
  Note:
    This is a helper function for parallel execution of `Vocabulary.from_text`
    method.
  """
  c = Counter()                             # 创建了一个空的 Counter 对象，用于存储单词和它们的频率。
  with open(file, 'r') as f:                # 打开文件，通过文件句柄 f 进行读取操作。
    for l in f:                             # 对文件中的每一行进行迭代处理
      words = l.strip().split()             # 针对每一行文本，去除首尾空白符后使用空格分割成单词列表
      c.update(words)                       # 使用 Counter 对象的 update 方法，统计当前行中单词的出现次数，将其添加到 c 中
  return c


def count_textfiles(files, workers=1):                         # 用于处理多个文本文件并计算它们中单词的频率
  c = Counter()                                                # 创建了一个空的 Counter 对象，用于存储所有文件中单词的频率
  with ProcessPoolExecutor(max_workers=workers) as executor:   # 创建一个进程池执行器（ProcessPoolExecutor），并使用 max_workers 指定最大工作进程数。这里是为了并行处理文件
    for c_ in executor.map(count_words, files):                # 对于文件列表中的每个文件路径，使用 count_words 函数并行地进行处理
      c.update(c_)                                             # c.update(c_): 对每个文件返回的单词频率统计（Counter 对象 c_）进行累积更新到最终的 Counter 对象 c 中
  return c


def count_lines(f):                                             # 用于统计文件中的行数
  if path.isfile(f):                                            # 检查指定的文件路径 f 是否存在并且是一个文件（不是目录或其他类型的文件）
    num_lines = sum(1 for line in open(f))                      # 如果文件存在，它使用 open(f) 打开文件，并且通过迭代文件的每一行，计算行数并将其求和。这里使用了一个生成器表达式来计算文件中行的数量。
    return num_lines
  else:
    return 0
# 生成随机游走序列，并将其写入磁盘文件中 据参数生成图的随机游走序列，并将这些序列写入指定的文件中。这通常在图数据挖掘、机器学习中用于学习节点嵌入（embedding）或图表示学习等任务中使用。
def _write_walks_to_disk(args):
  num_paths, path_length, alpha, rand, f = args                  # 从传递的参数中解包获取随机游走所需的参数
  G = __current_graph                                            # 获取一个名为 __current_graph 的全局图对象
  t_0 = time()                                                   # 记录函数开始执行的时间
  with open(f, 'w') as fout:                                     # 使用打开文件的上下文管理器，打开一个文件以供写入
    for walk in graph.build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length,
                                                 alpha=alpha, rand=rand):  # 对于 G 图中通过指定参数生成的随机游走序列（使用 build_deepwalk_corpus_iter 方法生成），对每个生成的随机游走进行操作
      fout.write(u"{}\n".format(u" ".join(v for v in walk)))      # 将每个生成的随机游走（walk）以空格分隔的形式写入文件
  logger.debug("Generated new file {}, it took {} seconds".format(f, time() - t_0)) # 记录生成新文件的时间消耗
  return f

# 执行并行生成随机游走，并将其写入磁盘文件。
def write_walks_to_disk(G, filebase, num_paths, path_length, alpha=0, rand=random.Random(0), num_workers=cpu_count(),
                        always_rebuild=True):                      # 如果设置为 True，总是重新生成随机游走序列；否则，检查文件大小并仅在需要时重建序列。
  global __current_graph
  __current_graph = G                                              # 设置 __current_graph 全局变量为传入的图对象 G
  files_list = ["{}.{}".format(filebase, str(x)) for x in list(range(num_paths))] # 创建文件名列表 files_list，形式为 filebase.0, filebase.1, ... , filebase.num_paths。
  expected_size = len(G)                                           # 计算预期文件大小 expected_size 为图的节点数量
  args_list = []                                                   # 创建一个空列表 args_list 用于存储并行执行 _write_walks_to_disk 函数所需的参数，以及一个空列表 files 用于存储生成的文件名
  files = []

  if num_paths <= num_workers:                                     # 计算每个工作进程要处理的路径数，并创建路径分配列表 paths_per_worker
    paths_per_worker = [1 for x in range(num_paths)]
  else:
    paths_per_worker = [len(list(filter(lambda z: z!= None, [y for y in x])))
                        for x in graph.grouper(int(num_paths / num_workers)+1, range(1, num_paths+1))]
# 目的是为每个工作进程分配路径数，其中路径数是非 None 元素的个数。这是通过 grouper 函数对路径进行分组，以便在并行处理中进行分配，并通过 filter 和 len 函数计算非 None 元素的数量。
# 这种方式可用于在工作进程之间平均分配路径数，以使每个进程的工作负载尽可能平衡。
  with ProcessPoolExecutor(max_workers=num_workers) as executor:  # 创建并行执行器的上下文管理器
    for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_worker):
      if always_rebuild or size != (ppw*expected_size):           # 对于每个文件，检查是否需要重建随机游走序列
        args_list.append((ppw, path_length, alpha, random.Random(rand.randint(0, 2**31)), file_))  # 如果需要重新生成随机游走序列，将相关参数添加到 args_list 列表中，以便后续使用。
      else:
        files.append(file_)                                        # 如果文件的行数与路径数乘以预期的图节点数一致，表示不需要重新生成，因此将文件名添加到 files 列表中。

  with ProcessPoolExecutor(max_workers=num_workers) as executor:   # 这是使用 ProcessPoolExecutor 创建并行执行器的上下文管理器，用于并行执行任务
    for file_ in executor.map(_write_walks_to_disk, args_list):    # 对于 args_list 中的每个参数集合，使用 executor.map 并行执行 _write_walks_to_disk 函数。
      files.append(file_)                                          # 将生成的文件名添加到 files 列表中。

  return files

# 这个类是一个迭代器，适合用于处理随机游走数据。通过迭代文件列表中的文件内容，将随机游走序列转化为单词列表，以便进行后续的处理或分析。
class WalksCorpus(object):
  def __init__(self, file_list):
    self.file_list = file_list
  def __iter__(self):
    for file in self.file_list:
      with open(file, 'r') as f:
        for line in f:
          yield line.split()
# 在 __iter__ 方法中的 yield 关键字用于产生一个生成器。每次调用 __iter__ 方法时，生成器会产生一个包含随机游走序列的单词列表。这个生成器可以被用于迭代处理大量的随机游走序列数据，而不会一次性加载整个数据到内存中。

# 提供了一个逐行读取文件内容并将其转换为单词列表的方法。它以生成器的形式返回每个文件中的单词列表，可以逐行处理文件内容，避免一次性加载整个文件到内存中，适用于处理大型文本文件或需要逐行处理的情况。
def combine_files_iter(file_list):
  for file in file_list:
    with open(file, 'r') as f:
      for line in f:
        yield line.split()