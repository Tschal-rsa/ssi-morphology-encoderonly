import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import math
import torch.nn as nn
import pandas as pd
import Levenshtein  # 添加Levenshtein距离计算

from dataset import SyriacDataset
from model import TransformerClassifier

def decode_sequence(indices, dataset):
    """将索引序列解码为叙利亚文字符"""
    return ''.join([dataset.idx_to_char[idx.item()] for idx in indices])

def get_device():
    """获取可用的设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def calculate_pos_weight(dataset):
    """计算类别权重"""
    # 首先统计所有出现的类别
    class_counts = {}
    total_samples = 0
    
    for _, labels in dataset.sentences:
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
            total_samples += 1
    
    # 获取最大类别数
    max_class = max(class_counts.keys())
    
    # 计算权重（使用类别频率的倒数）
    weights = torch.zeros(max_class + 1)
    for i in range(max_class + 1):
        if i in class_counts:
            weights[i] = total_samples / (len(class_counts) * class_counts[i])
        else:
            weights[i] = 1.0
    
    return weights

def calculate_levenshtein_distance(predictions, labels, patterns_df):
    """计算预测序列和标签序列之间的Levenshtein距离"""
    pred_symbols = [label_to_symbol(p.item(), patterns_df) for p in predictions]
    label_symbols = [label_to_symbol(l.item(), patterns_df) for l in labels]
    return Levenshtein.distance(''.join(pred_symbols), ''.join(label_symbols))

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, num_epochs=5):
    best_val_loss = float('inf')
    max_grad_norm = 1.0
    
    # 读取patterns.csv
    patterns_df = pd.read_csv('patterns.csv')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        train_total = 0
        train_zero_correct = 0
        train_nonzero_correct = 0
        train_nonzero_exact = 0
        train_zero_total = 0
        train_nonzero_total = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].long().to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            
            batch_size, seq_len, num_classes = outputs.size()
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            train_loss += loss.item()
            predictions = outputs.argmax(dim=-1)
            
            # 计算各种指标
            zero_mask = (labels == 0)
            nonzero_mask = (labels > 0)
            
            # 零猜成零的正确率
            train_zero_correct += ((predictions == 0) & zero_mask).sum().item()
            train_zero_total += zero_mask.sum().item()
            
            # 非零猜成非零的正确率
            train_nonzero_correct += ((predictions > 0) & nonzero_mask).sum().item()
            train_nonzero_total += nonzero_mask.sum().item()
            
            # 非零准确猜对的正确率
            train_nonzero_exact += ((predictions == labels) & nonzero_mask).sum().item()
            
            train_total += labels.numel()
            
            # 每100个batch显示一个样本
            if batch_idx % 100 == 0:
                print(f"\nBatch {batch_idx} Sample:")
                sample_idx = 0
                input_seq = input_ids[sample_idx]
                label_seq = labels[sample_idx * seq_len:(sample_idx + 1) * seq_len]
                pred_seq = predictions[sample_idx * seq_len:(sample_idx + 1) * seq_len]
                
                input_text = decode_sequence(input_seq, train_loader.dataset.dataset)
                label_symbols = [label_to_symbol(l.item(), patterns_df) for l in label_seq]
                pred_symbols = [label_to_symbol(p.item(), patterns_df) for p in pred_seq]
                
                label_line, pred_line = format_aligned_output(input_text, label_symbols, pred_symbols)
                
                print(f"Input: {input_text}")
                print(f"Label: {label_line}")
                print(f"Pred:  {pred_line}")
                print(f"Loss: {loss.item():.4f}")
                
                if label_symbols != pred_symbols:
                    print("\nDifferences:")
                    for i, (l, p) in enumerate(zip(label_symbols, pred_symbols)):
                        if l != p:
                            print(f"Position {i}: Label='{l}', Pred='{p}'")
        
        train_loss = train_loss / len(train_loader)
        train_zero_acc = train_zero_correct / train_zero_total if train_zero_total > 0 else 0
        train_nonzero_acc = train_nonzero_correct / train_nonzero_total if train_nonzero_total > 0 else 0
        train_nonzero_exact_acc = train_nonzero_exact / train_nonzero_total if train_nonzero_total > 0 else 0
        train_overall_acc = (train_zero_correct + train_nonzero_exact) / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_total = 0
        val_zero_correct = 0
        val_nonzero_correct = 0
        val_nonzero_exact = 0
        val_zero_total = 0
        val_nonzero_total = 0
        val_levenshtein_distance = 0
        
        print(f"\n{'='*50}")
        print("Validation Phase")
        print(f"{'='*50}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].long().to(device)
                
                outputs = model(input_ids)
                
                batch_size, seq_len, num_classes = outputs.size()
                outputs = outputs.view(-1, num_classes)
                labels = labels.view(-1)
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = outputs.argmax(dim=-1)
                
                # 计算各种指标
                zero_mask = (labels == 0)
                nonzero_mask = (labels > 0)
                
                # 零猜成零的正确率
                val_zero_correct += ((predictions == 0) & zero_mask).sum().item()
                val_zero_total += zero_mask.sum().item()
                
                # 非零猜成非零的正确率
                val_nonzero_correct += ((predictions > 0) & nonzero_mask).sum().item()
                val_nonzero_total += nonzero_mask.sum().item()
                
                # 非零准确猜对的正确率
                val_nonzero_exact += ((predictions == labels) & nonzero_mask).sum().item()
                
                # 计算Levenshtein距离
                val_levenshtein_distance += calculate_levenshtein_distance(predictions, labels, patterns_df)
                
                val_total += labels.numel()
                
                # 每50个batch显示一个验证样本
                if batch_idx % 50 == 0:
                    print(f"\nValidation Batch {batch_idx} Sample:")
                    sample_idx = 0
                    input_seq = input_ids[sample_idx]
                    label_seq = labels[sample_idx * seq_len:(sample_idx + 1) * seq_len]
                    pred_seq = predictions[sample_idx * seq_len:(sample_idx + 1) * seq_len]
                    
                    input_text = decode_sequence(input_seq, val_loader.dataset.dataset)
                    label_symbols = [label_to_symbol(l.item(), patterns_df) for l in label_seq]
                    pred_symbols = [label_to_symbol(p.item(), patterns_df) for p in pred_seq]
                    
                    label_line, pred_line = format_aligned_output(input_text, label_symbols, pred_symbols)
                    
                    print(f"Input: {input_text}")
                    print(f"Label: {label_line}")
                    print(f"Pred:  {pred_line}")
                    print(f"Loss: {loss.item():.4f}")
                    
                    if label_symbols != pred_symbols:
                        print("\nDifferences:")
                        for i, (l, p) in enumerate(zip(label_symbols, pred_symbols)):
                            if l != p:
                                print(f"Position {i}: Label='{l}', Pred='{p}'")
        
        val_loss = val_loss / len(val_loader)
        val_zero_acc = val_zero_correct / val_zero_total if val_zero_total > 0 else 0
        val_nonzero_acc = val_nonzero_correct / val_nonzero_total if val_nonzero_total > 0 else 0
        val_nonzero_exact_acc = val_nonzero_exact / val_nonzero_total if val_nonzero_total > 0 else 0
        val_overall_acc = (val_zero_correct + val_nonzero_exact) / val_total
        val_levenshtein_distance = val_levenshtein_distance / len(val_loader)
        
        # 计算零与非零的比例
        train_zero_ratio = train_zero_total / train_total
        train_nonzero_ratio = train_nonzero_total / train_total
        val_zero_ratio = val_zero_total / val_total
        val_nonzero_ratio = val_nonzero_total / val_total
        
        print(f"\n{'='*50}")
        print("\nEpoch Results:")
        print(f"Training Metrics:")
        print(f"1. Zero/Non-zero Ratio: {train_zero_ratio:.4f}/{train_nonzero_ratio:.4f}")
        print(f"2. Zero to Zero Accuracy: {train_zero_acc:.4f}")
        print(f"3. Non-zero to Non-zero Accuracy: {train_nonzero_acc:.4f}")
        print(f"4. Non-zero Exact Match Accuracy: {train_nonzero_exact_acc:.4f}")
        print(f"5. Overall Accuracy: {train_overall_acc:.4f}")
        print(f"6. Loss: {train_loss:.4f}")
        
        print(f"\nValidation Metrics:")
        print(f"1. Zero/Non-zero Ratio: {val_zero_ratio:.4f}/{val_nonzero_ratio:.4f}")
        print(f"2. Zero to Zero Accuracy: {val_zero_acc:.4f}")
        print(f"3. Non-zero to Non-zero Accuracy: {val_nonzero_acc:.4f}")
        print(f"4. Non-zero Exact Match Accuracy: {val_nonzero_exact_acc:.4f}")
        print(f"5. Overall Accuracy: {val_overall_acc:.4f}")
        print(f"6. Loss: {val_loss:.4f}")
        print(f"7. Overall Levenshtein Distance: {val_levenshtein_distance:.4f}")
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("New best model saved!")

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    将数据集划分为训练集、验证集和测试集
    
    Args:
        dataset: 完整数据集
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例之和必须为1"
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(seed)
    
    # 计算每个集合的大小
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # 划分数据集
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"数据集划分情况：")
    print(f"总数据量: {total_size}")
    print(f"训练集: {len(train_dataset)} ({train_ratio*100:.1f}%)")
    print(f"验证集: {len(val_dataset)} ({val_ratio*100:.1f}%)")
    print(f"测试集: {len(test_dataset)} ({test_ratio*100:.1f}%)")
    
    # 检查数据分布
    print("\n数据分布检查：")
    check_distribution(dataset, train_dataset, "训练集")
    check_distribution(dataset, val_dataset, "验证集")
    check_distribution(dataset, test_dataset, "测试集")
    
    return train_dataset, val_dataset, test_dataset

def check_distribution(full_dataset, subset, name):
    """检查数据子集的分布情况"""
    # 获取子集的索引
    indices = subset.indices
    
    # 统计正样本比例
    total_ones = 0
    total_samples = 0
    
    for idx in indices:
        _, labels = full_dataset.sentences[idx]
        total_ones += sum(labels)
        total_samples += len(labels)
    
    pos_ratio = total_ones / total_samples if total_samples > 0 else 0
    print(f"{name} 正样本比例: {pos_ratio:.4f}")

class CustomLoss(nn.Module):
    def __init__(self, zero_mistake_weight=2.5):
        super().__init__()
        self.base_criterion = nn.CrossEntropyLoss(reduction='none')
        self.zero_mistake_weight = zero_mistake_weight
        
    def forward(self, outputs, labels):
        # 计算基础的交叉熵损失
        base_loss = self.base_criterion(outputs, labels)  # [batch_size * seq_len]
        
        # 获取预测的类别
        predictions = outputs.argmax(dim=-1)  # [batch_size * seq_len]
        
        # 创建惩罚权重
        # 当真实标签非零但预测为零时，增加惩罚
        weights = torch.ones_like(base_loss)
        zero_mistakes = (predictions == 0) & (labels > 0)
        weights[zero_mistakes] = self.zero_mistake_weight
        
        weighted_loss = (base_loss * weights).mean()
        return weighted_loss

def label_to_symbol(label, patterns_df):
    """Convert numeric label to corresponding symbol form"""
    if label == 0:
        return ''
    try:
        symbol = patterns_df[patterns_df['编号'] == label]['符号形式'].values[0]
        return symbol
    except:
        return str(label)

def format_aligned_output(input_text, label_symbols, pred_symbols):
    """Align input text and annotations for display"""
    # 确保输入文本和标注长度一致
    min_len = min(len(input_text), len(label_symbols), len(pred_symbols))
    input_text = input_text[:min_len]
    label_symbols = label_symbols[:min_len]
    pred_symbols = pred_symbols[:min_len]
    
    # 创建对齐的输出
    label_line = ''.join([f"{char}{label}" for char, label in zip(input_text, label_symbols)])
    pred_line = ''.join([f"{char}{pred}" for char, pred in zip(input_text, pred_symbols)])
    
    return label_line, pred_line

def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model performance on test set"""
    model.eval()
    test_loss = 0
    test_total = 0
    test_zero_correct = 0
    test_nonzero_correct = 0
    test_nonzero_exact = 0
    test_zero_total = 0
    test_nonzero_total = 0
    test_levenshtein_distance = 0
    
    # 读取patterns.csv
    patterns_df = pd.read_csv('patterns.csv')
    
    print(f"\n{'='*50}")
    print("Testing Phase")
    print(f"{'='*50}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].long().to(device)
            
            outputs = model(input_ids)
            
            batch_size, seq_len, num_classes = outputs.size()
            outputs = outputs.view(-1, num_classes)
            labels = labels.view(-1)
            
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            predictions = outputs.argmax(dim=-1)
            
            # 计算各种指标
            zero_mask = (labels == 0)
            nonzero_mask = (labels > 0)
            
            # 零猜成零的正确率
            test_zero_correct += ((predictions == 0) & zero_mask).sum().item()
            test_zero_total += zero_mask.sum().item()
            
            # 非零猜成非零的正确率
            test_nonzero_correct += ((predictions > 0) & nonzero_mask).sum().item()
            test_nonzero_total += nonzero_mask.sum().item()
            
            # 非零准确猜对的正确率
            test_nonzero_exact += ((predictions == labels) & nonzero_mask).sum().item()
            
            # 计算Levenshtein距离
            test_levenshtein_distance += calculate_levenshtein_distance(predictions, labels, patterns_df)
            
            test_total += labels.numel()
            
            # 每50个batch显示一个测试样本
            if batch_idx % 50 == 0:
                print(f"\nTest Batch {batch_idx} Sample:")
                sample_idx = 0
                input_seq = input_ids[sample_idx]
                label_seq = labels[sample_idx * seq_len:(sample_idx + 1) * seq_len]
                pred_seq = predictions[sample_idx * seq_len:(sample_idx + 1) * seq_len]
                
                input_text = decode_sequence(input_seq, test_loader.dataset.dataset)
                label_symbols = [label_to_symbol(l.item(), patterns_df) for l in label_seq]
                pred_symbols = [label_to_symbol(p.item(), patterns_df) for p in pred_seq]
                
                label_line, pred_line = format_aligned_output(input_text, label_symbols, pred_symbols)
                
                print(f"Input: {input_text}")
                print(f"Label: {label_line}")
                print(f"Pred:  {pred_line}")
                print(f"Loss: {loss.item():.4f}")
                
                if label_symbols != pred_symbols:
                    print("\nDifferences:")
                    for i, (l, p) in enumerate(zip(label_symbols, pred_symbols)):
                        if l != p:
                            print(f"Position {i}: Label='{l}', Pred='{p}'")
    
    test_loss = test_loss / len(test_loader)
    test_zero_acc = test_zero_correct / test_zero_total if test_zero_total > 0 else 0
    test_nonzero_acc = test_nonzero_correct / test_nonzero_total if test_nonzero_total > 0 else 0
    test_nonzero_exact_acc = test_nonzero_exact / test_nonzero_total if test_nonzero_total > 0 else 0
    test_overall_acc = (test_zero_correct + test_nonzero_exact) / test_total
    test_levenshtein_distance = test_levenshtein_distance / len(test_loader)
    
    # 计算零与非零的比例
    test_zero_ratio = test_zero_total / test_total
    test_nonzero_ratio = test_nonzero_total / test_total
    
    print("\nTest Results:")
    print(f"1. Zero/Non-zero Ratio: {test_zero_ratio:.4f}/{test_nonzero_ratio:.4f}")
    print(f"2. Zero to Zero Accuracy: {test_zero_acc:.4f}")
    print(f"3. Non-zero to Non-zero Accuracy: {test_nonzero_acc:.4f}")
    print(f"4. Non-zero Exact Match Accuracy: {test_nonzero_exact_acc:.4f}")
    print(f"5. Overall Accuracy: {test_overall_acc:.4f}")
    print(f"6. Loss: {test_loss:.4f}")
    print(f"7. Overall Levenshtein Distance: {test_levenshtein_distance:.4f}")
    
    return test_loss, test_zero_acc, test_nonzero_acc, test_nonzero_exact_acc, test_overall_acc, test_levenshtein_distance

def main():
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")
    
    # 创建数据集
    dataset = SyriacDataset('training.csv', max_length=128)
    print(f"Dataset size: {len(dataset)} sentences")
    
    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return
    
    # 统计所有出现的类别
    all_labels = set()
    for _, labels in dataset.sentences:
        all_labels.update(labels)
    
    # 获取最大类别数
    max_label = max(all_labels)
    num_classes = max_label + 1
    
    print("\nClass Statistics:")
    print(f"Number of unique classes: {len(all_labels)}")
    print(f"Maximum class value: {max_label}")
    print(f"Total number of classes: {num_classes}")
    print(f"All classes: {sorted(list(all_labels))}")
    
    # 使用新的划分函数
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False,
        num_workers=4
    )
    
    # 计算类别权重
    class_weights = calculate_pos_weight(dataset)
    print(f"\nClass weights: {class_weights}")
    
    # 创建模型
    model = TransformerClassifier(num_classes=num_classes).to(device)
    print(f"\nNumber of model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # 定义优化器和损失函数
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True, min_lr=1e-6)
    criterion = CustomLoss(zero_mistake_weight=2.0).to(device)
    
    # 训练模型
    train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device)
    
    # 在测试集上评估模型
    print("\nStarting model evaluation on test set...")
    evaluate_model(model, test_loader, criterion, device)

if __name__ == '__main__':
    main() 