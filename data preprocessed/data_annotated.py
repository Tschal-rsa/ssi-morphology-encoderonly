"""Functions and constants related to Datasets.

MAX_LENGTH          maximum length in tokens of a sequence.
SOS_token           start of sentence token
EOS_token           end of sentence token
INPUT_WORD_TO_IDX   mapping from an input token to an index
OUTPUT_WORD_TO_IDX  mapping from an output token to an index

mc_reduce           reduce an output sequence to a more compact form
mc_expand           inverse of the above

collate_fn          Create a batch from a list of input records

encode_string       Convert a string to a Torch Tensor, using a mapping
decode_string       Convert a Torch Tensor to a string, using a mapping

Dataset wrappers:
    HebrewWords     a pytroch Dataset arount the hebrew bible, returns words.

"""
# 导入部分
import collections  # 导入collections模块，提供了如defaultdict等特殊容器数据类型
import os  # 导入os模块，用于文件路径操作
import re  # 导入re模块，用于正则表达式操作

from sklearn.utils import shuffle  # 导入数据打乱函数
from sklearn.model_selection import train_test_split  # 导入用于分割训练集和测试集的函数
import torch  # 导入PyTorch库
from torch.utils.data import Dataset  # 导入PyTorch的Dataset基类
from torch.nn.utils.rnn import pad_sequence  # 导入序列填充函数
from config import device, MC_PREFIXES, PAD_IDX, SOS_token, EOS_token, TRAIN_DATA_FOLDER  # 从配置文件导入常量


class DataReader:
    """
    读取输入和输出文件中的数据，
    并将数据按序列长度分组为连续的经文段。
    将数据分为训练集、验证集和测试集。
    
    创建长度为sequence_length的部分重叠文本序列。
    确保训练集、验证集和测试集之间没有（部分）重叠非常重要。
    同时，不同书卷的文本之间也没有重叠。
    """
    def __init__(self, 
                 input_filename: str,  # 输入文件名
                 output_filename: str,  # 输出文件名
                 sequence_length: int,  # 序列长度
                 val_plus_test_size: float,  # 验证集加测试集的比例
                 INPUT_WORD_TO_IDX: dict,  # 输入词到索引的映射字典
                 OUTPUT_WORD_TO_IDX: dict):  # 输出词到索引的映射字典
        # 初始化实例变量            
        self.input_filename = os.path.join(TRAIN_DATA_FOLDER, input_filename)  # 组合完整的输入文件路径
        self.output_filename = os.path.join(TRAIN_DATA_FOLDER, output_filename)  # 组合完整的输出文件路径
        self.sequence_length = sequence_length  # 保存序列长度
        self.val_plus_test_size = val_plus_test_size  # 保存验证集和测试集的总比例
        self.INPUT_WORD_TO_IDX = INPUT_WORD_TO_IDX  # 保存输入词到索引的映射
        self.OUTPUT_WORD_TO_IDX = OUTPUT_WORD_TO_IDX  # 保存输出词到索引的映射
        
        # 尝试读取输入文件
        try:
            with open(self.input_filename, 'r') as f:
                input_verses = f.readlines()  # 读取所有行
        except FileNotFoundError as err:
            print(err)  # 如果文件不存在，打印错误信息
        
        # 尝试读取输出文件
        try:
            with open(self.output_filename, 'r') as f:
                output_verses = f.readlines()  # 读取所有行
        except FileNotFoundError as err:
            print(err)  # 如果文件不存在，打印错误信息

        # 确保输入和输出文件有相同数量的行
        assert len(input_verses) == len(output_verses)
        
        # 创建两个默认字典，用于按书卷存储输入和输出单词
        all_input_words_per_book = collections.defaultdict(list)
        all_output_words_per_book = collections.defaultdict(list)
        
        # 处理每一行数据
        for i in range(len(input_verses)):
            # 分割输入行：书卷、章、节、文本
            bo, ch, ve, text = tuple(input_verses[i].strip().split('\t'))
            # 分割输出行：书卷、章、节、输出
            bo, ch, ve, output = tuple(output_verses[i].strip().split('\t'))

            # 按空格分割文本为单词列表
            input_words = text.split()
            # 将下划线替换为"_ _"然后分割，确保下划线被视为独立标记
            output_words = output.replace("_", "_ _").split()
            
            # 检查输入和输出单词数量是否匹配
            if (len(input_words) == len(output_words)):
                # 匹配则添加到对应书卷的列表中
                all_input_words_per_book[bo].append(input_words)
                all_output_words_per_book[bo].append(output_words)
            else:
                # 不匹配则打印错误信息
                print(f"Encoding issue with {bo} {ch} {ve} : mismatch in number of words")
                print(input_words)
                print(output_words)

        # 对输入和输出单词按书卷进行分组
        all_input_words_per_book_grouped = self.group_verses(all_input_words_per_book)
        all_output_words_per_book_grouped = self.group_verses(all_output_words_per_book)

        # 扁平化分组后的嵌套列表结构
        all_input_words = self.flatten_inner_lists(all_input_words_per_book_grouped)
        all_output_words = self.flatten_inner_lists(all_output_words_per_book_grouped)

        # 创建滑动窗口文本序列
        all_input_seq_lists, all_output_seq_lists = self.make_rolling_window_strings(all_input_words, all_output_words)

        # 分割数据集：首先分出训练集和"验证+测试"集
        self.X_train, X_val_test, self.y_train, y_val_test = train_test_split(
            all_input_seq_lists, all_output_seq_lists, 
            test_size=self.val_plus_test_size, random_state=42
        )
        # 将"验证+测试"集再分为验证集和测试集
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_val_test, y_val_test, 
            test_size=0.5, random_state=11
        )
        
        # 扁平化列表结构，使每个数据集成为简单的列表
        self.flatten_list_of_lists()

        
    def group_verses(self, all_words_per_book: dict):
        """
        将经文按固定长度分组。
        这样可以保证每个序列都具有期望的长度，
        并且文本可以在不重叠的情况下进行分割。
        输入：
            all_words_per_book: 字典，键是书名，值是列表的列表。
            每个列表包含一节经文的单词。
        输出：
            grouped_verses_dict: 字典，键是书名，值是列表的列表的列表，
            与输入类似，但增加了一层，其中经文按sequence_length分组。
        """
        grouped_verses_dict = {}  # 创建一个新字典存储分组结果
        
        # 遍历每本书的经文列表
        for book, verse_list in all_words_per_book.items():
        
            # 如果该书的经文数量小于等于序列长度，整本书作为一个组
            if len(verse_list) <= self.sequence_length:
                grouped_verses_dict[book] = [verse_list]
            else:
                # 否则，按固定长度将经文分组
                grouped_verses = [verse_list[idx:idx+self.sequence_length] for idx in range(0, len(verse_list), self.sequence_length)]
                # 如果最后一组不足序列长度，将其与前一组合并
                if len(grouped_verses[-1]) < self.sequence_length:
                    grouped_verses[-1] += grouped_verses.pop(-1)
                grouped_verses_dict[book] = grouped_verses
            
        return grouped_verses_dict
        
    def make_rolling_window_strings(self, all_input_words, all_output_words):
        """
        创建固定长度的字符串序列。
        移除包含特殊标记的序列：
        - 包含 '*' 的序列（ketiv/qere情况，这是希伯来圣经中的文本变体）
        - 输出中包含 '_' 的序列（通常是地名）
        """
        # 创建两个空列表存储生成的序列
        all_input_seq_lists = []
        all_output_seq_lists = []
        
        # 遍历每本书的单词列表
        for bo in all_input_words.keys():
            for word_list_input, word_list_output in zip(all_input_words[bo], all_output_words[bo]):
                input_seq_list = []
                output_seq_list = []
                # 使用滑动窗口创建固定长度的序列
                for word_idx in range(len(word_list_input) - self.sequence_length + 1):
                    # 将单词连接成字符串
                    input_seq = ' '.join(word_list_input[word_idx:word_idx + self.sequence_length])
                    output_seq = ' '.join(word_list_output[word_idx:word_idx + self.sequence_length])

                    # 过滤掉包含特殊标记的序列
                    if '*' not in input_seq and '_' not in output_seq:
                        input_seq_list.append(input_seq)
                        output_seq_list.append(output_seq)
                        # 更新字符到索引的映射字典
                        self.make_char2idx_dicts(input_seq, output_seq)

                # 如果当前组生成了有效序列，将其添加到结果列表
                if input_seq_list:
                    all_input_seq_lists.append(input_seq_list)
                    all_output_seq_lists.append(output_seq_list)
                    
        return all_input_seq_lists, all_output_seq_lists
        
    def make_char2idx_dicts(self, input_seq, output_seq):
        """
        更新输入和输出字符到索引的映射字典。
        对于每个新出现的字符，分配一个唯一的索引。
        """
        # 处理输入序列中的每个字符
        for char in input_seq:
            if char not in self.INPUT_WORD_TO_IDX:
                self.INPUT_WORD_TO_IDX[char] = len(self.INPUT_WORD_TO_IDX)
        # 处理输出序列中的每个字符
        for char in output_seq:
            if char not in self.OUTPUT_WORD_TO_IDX:
               self.OUTPUT_WORD_TO_IDX[char] = len(self.OUTPUT_WORD_TO_IDX)
        
    @staticmethod
    def flatten_inner_lists(data_dict):
        """
        扁平化嵌套列表结构中的最内层。
        """
        new_data_dict = {}  # 创建新字典存储扁平化结果
        # 遍历原字典的每个条目
        for bo, w_list in data_dict.items():
            flattened_list = []
            # 遍历每个子列表
            for sub_list in w_list:
                # 使用列表推导式扁平化最内层列表
                flattened_list.append([word for subsublist in sub_list for word in subsublist])
            new_data_dict[bo] = flattened_list
            
        return new_data_dict
        
    def flatten_list_of_lists(self):
        """
        扁平化训练集、验证集和测试集的嵌套列表结构。
        将每个数据集转换为简单的列表，其中每个元素是一个序列。
        """
        # 使用列表推导式扁平化各个数据集
        self.X_train = [item for sublist in self.X_train for item in sublist]
        self.y_train = [item for sublist in self.y_train for item in sublist]
        self.X_val = [item for sublist in self.X_val for item in sublist]
        self.y_val = [item for sublist in self.y_val for item in sublist]
        self.X_test = [item for sublist in self.X_test for item in sublist]
        self.y_test = [item for sublist in self.y_test for item in sublist]


class HebrewWords(Dataset):
    """PyTorch的希伯来圣经文本数据集包装器。按单词处理。"""

    def __init__(self, 
                 input_data: list,  # 包含文本序列的列表
                 output_data: list,  # 包含输出序列的列表
                 INPUT_WORD_TO_IDX: dict,  # 输入字符到索引的映射字典
                 OUTPUT_WORD_TO_IDX: dict):  # 输出字符到索引的映射字典
        """
        参数：
            input_data: 包含文本序列(字符串)的列表
            output_data: 包含文本序列(字符串)的列表
            INPUT_WORD_TO_IDX, OUTPUT_WORD_TO_IDX: 用于字符到整数的转换字典
        文件格式为每行一节经文，使用制表符分隔的元数据：书卷 章 节 文本
        注意：输出使用mc_reduce函数进行简化处理
        数据集中的每个样本包含以下字段：
            text: 原始文本字符串
            output: 输出字符串
            encoded_text: 编码后的文本张量
            encoded_output: 编码后的输出张量
        """
        # 保存输入参数为实例变量
        self.input_data = input_data
        self.output_data = output_data
        self.INPUT_WORD_TO_IDX = INPUT_WORD_TO_IDX
        self.OUTPUT_WORD_TO_IDX = OUTPUT_WORD_TO_IDX

    def __len__(self):
        """返回数据集的大小"""
        return len(self.input_data)

    def __getitem__(self, idx):
        """获取指定索引的样本"""
        # 如果索引是张量，转换为Python列表
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取输入文本和处理后的输出文本
        text_input = self.input_data[idx]
        text_output = mc_reduce(self.output_data[idx])

        # 创建并返回包含多个字段的样本字典
        sample = {
                "text": text_input,  # 原始输入文本
                "encoded_text": encode_string(text_input, self.INPUT_WORD_TO_IDX, add_sos=False, add_eos=True),  # 编码后的输入文本
                "output": text_output,  # 处理后的输出文本
                "encoded_output": encode_string(text_output, self.OUTPUT_WORD_TO_IDX, add_sos=True, add_eos=True)  # 编码后的输出文本
                }

        return sample


def collate_transformer_fn(batch):
    """将数据样本整合成批次张量的函数，适用于Transformer模型"""
    # 创建空列表存储源和目标批次
    src_batch, tgt_batch = [], []
    
    # 遍历批次中的每个样本
    for sample in batch:
        src_batch.append(sample['encoded_text'])  # 添加编码后的输入文本
        tgt_batch.append(sample['encoded_output'])  # 添加编码后的输出文本

    # 对序列进行填充，使同一批次中的所有序列长度相同
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def encode_string(seq: str, d: dict, add_sos=False, add_eos=True):
    """
    将字符串转换为使用给定字典的索引张量。

    如果add_sos为True，在句子开头添加SOS_token（默认为False）
    如果add_eos为True（默认），在句子结尾添加EOS_token
    """
    idxs = []  # 创建空列表存储索引
    
    # 如果需要，添加开始标记
    if add_sos:
        idxs.append(SOS_token)

    # 将每个字符转换为对应的索引
    for w in seq:
        idxs.append(d[w])

    # 如果需要，添加结束标记
    if add_eos:
        idxs.append(EOS_token)
        
    # 返回长整型张量，包含所有字符的索引
    return torch.tensor(idxs, dtype=torch.long, device=device)


def decode_string(t, d:dict, strip_sos=True, strip_eos=True):
    """
    将索引张量转换回字符串，使用给定的字典。

    如果strip_eos为True（默认），从字符串中移除所有EOS_token
    如果strip_sos为True（默认），从字符串中移除所有SOS_token
    """
    # 创建索引到字符的反向映射字典
    inv_d = {v: k for k, v in d.items()}
    seq = ""
    
    # 遍历张量中的每个元素
    for c in list(t):
        # 如果是张量，转换为Python数值
        if isinstance(c, torch.Tensor):
            c = c.item()

        # 如果是结束标记且需要移除，则跳过
        if strip_eos and c == EOS_token:
            continue

        # 如果是开始标记且需要移除，则跳过
        if strip_sos and c == SOS_token:
            continue

        # 将索引转换为字符并连接到结果字符串
        seq = seq + inv_d[c]
    return seq


def mc_reduce(s: str) -> str:
    """
    将输出简化为最小形式。

    简化包括移除所有双重标记前缀中的最左侧标记，
    以及元音模式标记中冗余的冒号。
    """
    # 对于每个前缀标记，使用正则表达式移除左侧标记
    for c in MC_PREFIXES:
        s = re.sub(f'{c}([^{c}]*{c})', r'\1', s)
    # 移除所有冒号
    return s.replace(':', '')


def mc_expand(s: str) -> str:
    """
    这个函数撤销简化操作。搜索模式中的连字符
    确保我们限制在单个分析单词内。
    """
    # 在所有字母序列前添加冒号
    s = re.sub(r'([a-z]+)', r':\1', s)
    # 对前缀标记进行正则表达式转义
    r = re.sub('(.)', r'\\\1', ''.join(MC_PREFIXES))
    # 对每个前缀标记，在其出现的地方添加额外的标记
    for c in MC_PREFIXES:
        s = re.sub(f'([^-{r}]*{c})', f'{c}\\1', s)
    return s
        
        
class DataMerger:
    """
    DataMerger类用于模型同时在两个数据集（希伯来文和叙利亚文）
    上训练时合并数据。
    """
    def __init__(self, 
                 input1: list,  # 第一个数据集的输入列表
                 input2: list,  # 第二个数据集的输入列表
                 output1: list,  # 第一个数据集的输出列表
                 output2: list):  # 第二个数据集的输出列表
        # 保存输入参数为实例变量
        self.input1 = input1
        self.input2 = input2
        self.output1 = output1
        self.output2 = output2

    def merge_data(self):
        """合并两个数据集的输入和输出列表"""
        # 简单地连接两个列表
        self.input_data = self.input1 + self.input2
        self.output_data = self.output1 + self.output2
        return self.input_data, self.output_data

    def shuffle_data(self, input_data, output_data):
        """打乱数据集，保持输入和输出的对应关系"""
        # 使用sklearn的shuffle函数同时打乱两个列表，保持它们的对应关系
        input_data_shuffled, output_data_shuffled = shuffle(input_data, output_data, random_state=0)
        return input_data_shuffled, output_data_shuffled 