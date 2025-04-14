import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class SyriacDataset(Dataset):
    def __init__(self, csv_file, max_length=128):
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length
        
        # 创建字符到索引的映射
        self.char_to_idx = {
            '>': 0, 'B': 1, 'G': 2, 'D': 3, 'H': 4, 'W': 5, 'Z': 6,
            'X': 7, 'V': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
            'S': 14, '<': 15, 'P': 16, 'Y': 17, 'Q': 18, 'R': 19,
            'C': 20, 'T': 21, ' ': 22
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        # 处理数据，按句子分组
        self.sentences = []
        current_sentence = []
        current_labels = []
        
        def split_sentence(sentence, labels):
            """智能分割句子，尽量在空格处分割"""
            if len(sentence) <= self.max_length:
                return [(sentence, labels)]
            
            segments = []
            start = 0
            
            while start < len(sentence):
                # 找到下一个分割点（在max_length范围内的最后一个空格）
                end = min(start + self.max_length, len(sentence))
                if end < len(sentence):
                    # 在max_length范围内从后向前找最后一个空格
                    for i in range(end, start, -1):
                        if sentence[i] == ' ' and sentence[i-1] == ' ':
                            end = i
                            break
                
                # 如果找不到合适的空格，就强制在max_length处分割
                if end == start + self.max_length:
                    # 向前找最近的单个空格
                    for i in range(end, start, -1):
                        if sentence[i] == ' ':
                            end = i
                            break
                
                segments.append((sentence[start:end], labels[start:end]))
                start = end
            
            return segments
        
        for _, row in self.data.iterrows():
            if pd.isna(row['input']) or (row['input'] == ' ' and len(current_sentence) > 0 and current_sentence[-1] == ' '):
                if current_sentence:
                    # 使用智能分割函数处理当前句子
                    segments = split_sentence(current_sentence, current_labels)
                    self.sentences.extend(segments)
                    current_sentence = []
                    current_labels = []
            else:
                current_sentence.append(row['input'])
                # 修改标签处理逻辑
                output_value = row['output']
                if isinstance(output_value, str):
                    # 如果输出包含多个值，取第一个值
                    output_value = output_value.split(',')[0]
                output_value = int(output_value)  # 将字符串转换为整数
                current_labels.append(output_value)  # 直接使用原始值，不做限制
        
        if current_sentence:
            # 处理最后一个句子
            segments = split_sentence(current_sentence, current_labels)
            self.sentences.extend(segments)
        
        print(f"Total sentences found: {len(self.sentences)}")
        
        # 统计序列长度
        lengths = [len(sentence) for sentence, _ in self.sentences]
        print(f"\nSequence length statistics:")
        print(f"Min length: {min(lengths)}")
        print(f"Max length: {max(lengths)}")
        print(f"Mean length: {np.mean(lengths):.2f}")
        print(f"Median length: {np.median(lengths):.2f}")
        print(f"95th percentile: {np.percentile(lengths, 95):.2f}")
        
        if len(self.sentences) > 0:
            print(f"\nSample sentence: {''.join(self.sentences[0][0])}")
            print(f"Sample labels: {self.sentences[0][1]}")
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence, labels = self.sentences[idx]
        
        # 将字符转换为索引
        sentence_indices = [self.char_to_idx[char] for char in sentence]
        
        # 添加padding，但只到实际需要的长度
        if len(sentence_indices) < self.max_length:
            sentence_indices.extend([self.char_to_idx[' ']] * (self.max_length - len(sentence_indices)))
        else:
            sentence_indices = sentence_indices[:self.max_length]
        
        # 确保标签长度与句子长度匹配
        if len(labels) < self.max_length:
            labels.extend([0] * (self.max_length - len(labels)))
        else:
            labels = labels[:self.max_length]
        
        return {
            'input_ids': torch.tensor(sentence_indices, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        } 