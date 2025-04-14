import torch
import random
from model import TransformerClassifier
from dataset import SyriacDataset
import re
import pandas as pd

# 加载模型
def load_model(model_path, device):
    # 修改num_classes为129以匹配保存的模型
    model = TransformerClassifier(num_classes=129)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# 加载patterns.csv
def load_patterns(pattern_file):
    patterns_df = pd.read_csv(pattern_file)
    return patterns_df

def label_to_symbol(label, patterns_df):
    """将数字标签转换为对应的符号形式"""
    if label == 0:
        return ''
    try:
        symbol = patterns_df[patterns_df['编号'] == label]['符号形式'].values[0]
        return symbol
    except:
        return str(label)

def format_aligned_output(input_text, pred_symbols):
    """将输入文本和预测标注对齐显示"""
    # 确保输入文本和预测标注长度一致
    min_len = min(len(input_text), len(pred_symbols))
    input_text = input_text[:min_len]
    pred_symbols = pred_symbols[:min_len]
    
    # 创建对齐的输出
    pred_line = ''.join([f"{char}{pred}" for char, pred in zip(input_text, pred_symbols)])
    
    return pred_line

# 加载文本文件并分割成句子
def load_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 使用正则表达式分割文本，保留章节标记
    sentences = []
    current_sentence = []
    current_line_number = None
    
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # 如果是章节标记，单独处理
        if line.startswith('Chapter'):
            if current_sentence:
                sentences.append((''.join(current_sentence), current_line_number))  # 保存句子和行号
                current_sentence = []
                current_line_number = None
            sentences.append((line, None))
        else:
            # 处理普通文本行，提取行号
            parts = re.split(r'^(\d+)\s+', line)  # 提取行首的数字
            if len(parts) > 1:
                current_line_number = parts[1]
                line = parts[2]
            current_sentence.extend(list(line.strip()))  # 将每行转换为字符列表
    
    if current_sentence:
        sentences.append((''.join(current_sentence), current_line_number))
    
    return sentences

# 将文本转换为模型输入
def text_to_tensor(text, char_to_idx, max_length=128):
    # 将字符转换为索引
    indices = [char_to_idx.get(c, char_to_idx[' ']) for c in text]
    
    # 添加padding
    if len(indices) < max_length:
        indices.extend([char_to_idx[' ']] * (max_length - len(indices)))
    else:
        indices = indices[:max_length]
    
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # 添加batch维度

def split_sentence(sentence, max_length=50):
    """
    Intelligently split a sentence into segments that don't exceed max_length.
    Each segment will end with two spaces to maintain consistency with training data.
    """
    words = sentence.split()
    segments = []
    current_segment = []
    current_length = 0
    
    for word in words:
        # Check if adding this word would exceed max_length
        if current_length + len(word) + len(current_segment) > max_length and current_segment:
            # Join current segment and add two spaces
            segments.append(' '.join(current_segment) + '  ')
            current_segment = []
            current_length = 0
        
        current_segment.append(word)
        current_length += len(word)
    
    # Add any remaining words
    if current_segment:
        segments.append(' '.join(current_segment) + '  ')
    
    return segments if segments else [sentence + '  ']  # Ensure we return at least one segment

def main(input_sentence=None):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    model = load_model('best_model.pth', device)
    
    # 加载patterns
    patterns_df = load_patterns('patterns.csv')
    
    # 准备字符映射
    char_to_idx = {
        '>': 0, 'B': 1, 'G': 2, 'D': 3, 'H': 4, 'W': 5, 'Z': 6,
        'X': 7, 'V': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13,
        'S': 14, '<': 15, 'P': 16, 'Y': 17, 'Q': 18, 'R': 19,
        'C': 20, 'T': 21, ' ': 22
    }
    
    if input_sentence is None:
        # 加载文本
        sentences = load_sentences('isaiah_transcription.txt')
        
        # 过滤掉章节标记，只选择实际的文本行
        text_sentences = [s for s in sentences if not s[0].startswith('Chapter')]
        if not text_sentences:
            print("No valid sentences found!")
            return
        
        # 随机选择一句话
        random_sentence, line_number = random.choice(text_sentences)
        sentence_to_process = random_sentence
        print(f"\nSelected text (Line {line_number}):")
    else:
        sentence_to_process = input_sentence
        line_number = None
        print(f"\nInput text: {sentence_to_process}")
        
        # 直接处理整个输入句子，不进行分段
        sentence_segments = [sentence_to_process]
    
    # 使用与训练时相同的分割逻辑处理句子（如果不是直接输入）
    if input_sentence is None:
        sentence_segments = split_sentence(sentence_to_process)
        segment = random.choice(sentence_segments)
        print(f"Processing segment: {segment}")
    else:
        segment = sentence_to_process
        print(f"Processing input: {segment}")
    
    # 转换为模型输入
    input_tensor = text_to_tensor(segment, char_to_idx)
    input_tensor = input_tensor.to(device)
    
    # 进行预测
    with torch.no_grad():
        output = model(input_tensor)
        predictions = torch.argmax(output, dim=-1)
    
    # 将预测结果转换为符号形式
    pred_symbols = [label_to_symbol(p.item(), patterns_df) for p in predictions[0]]
    
    # 创建对齐的输出
    pred_line = format_aligned_output(segment, pred_symbols)
    print(f"Pred:  {pred_line}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run best model prediction')
    parser.add_argument('--input', type=str, help='Input sentence to predict')
    args = parser.parse_args()
    main(args.input)

if __name__ == "__main__":
    main()
